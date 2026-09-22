#!/usr/bin/env python3
"""Score a capability-regression run: source (llama.cpp) vs MLX output.

Consumes the two JSONL files produced by run_eval.py and emits:

* ``report.json`` — full machine-readable results (committed)
* ``report.md``   — per-question outputs, verdicts, and summary metrics (committed)

``--rescore-report`` re-evaluates an existing ``report.json`` under the
current ``questions.json`` checks without re-running the models: verdicts
are recomputed from the embedded per-item outputs, while generation-time
fields (anomalies, blocking flags, timings) are preserved from the original
run and a ``rescoring`` provenance block records what changed.

Gating rules (strict):

* Runtime parameters (``max_tokens``, ``temp``) are read from the *actual
  run records* — never from questions.json — and both sides must agree
  exactly, otherwise scoring refuses to run.
* Anomalies are reported separately per type (truncated / repetition_loop /
  empty_output / garbled / replacement_chars / control_chars).
* Any blocking anomaly marks the question incorrect for that side — an
  answer that contains the right keyword but then loops into the token cap
  is NOT an instruction success.
* Format constraints are strict: ``number`` requires exactly one number,
  ``all_colors`` requires exactly the three requested colors and no other
  color words, ``one_word`` / ``yes_no`` must match the whole final text.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import unicodedata
from pathlib import Path

NEGATIVES = ["不", "别", "无", "no", "not", "false", "否", "错", "wrong", "non"]
AFFIRMATIVES = ["是", "对", "是的", "yes", "true", "correct", "right", "的确"]

# Anomaly types that invalidate an answer for scoring. Reported separately
# per type; any one of them forces the verdict to incorrect.
BLOCKING_ANOMALIES = {
    "truncated",
    "repetition_loop",
    "empty_output",
    "garbled",
    "replacement_chars",
    "control_chars",
}

# Color lexicon for the strict "exactly three colors" check: the three
# required groups below, plus every other color word that would make the
# answer contain MORE than exactly three colors.
REQUIRED_COLOR_GROUPS = (("red", "红"), ("green", "绿"), ("blue", "蓝"))
EXTRA_COLORS = (
    "yellow", "黄", "purple", "紫", "orange", "橙", "pink", "粉",
    "brown", "棕", "black", "黑", "white", "白", "gray", "grey", "灰",
    "gold", "金", "silver", "银", "violet", "cyan", "beige", "teal",
)


def extract_final(raw: str) -> str:
    """Post-<think> final text, template junk stripped."""
    text = raw
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"<\|[^>]*\|>", "", text)
    return text.strip()


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE)
    return s


def printable_ratio(s: str) -> float:
    body = [c for c in s if not c.isspace()]
    if not body:
        return 1.0
    ok = sum(1 for c in body if c.isprintable() and c != "\ufffd")
    return ok / len(body)


def tokens_alnum(s: str) -> list[str]:
    return re.findall(r"[0-9a-z]+|[\u4e00-\u9fff]+", s.lower())


# ---------------------------------------------------------------------------
# anomaly detection (each type reported separately)
# ---------------------------------------------------------------------------


def is_repetition_loop(final: str) -> bool:
    """Detect degenerate temp-0 loops in the final text.

    Heuristics, in order: an immediately repeated tail block, a single
    non-space segment repeating >= 5 times, a >= 4x repeated last line, or
    a degenerate single-character run.
    """
    s = re.sub(r"\s+", " ", final.strip())
    if len(s) < 64:
        return False
    for k in range(8, 65):  # tail is an immediate repeat of the block before it
        if len(s) >= 2 * k and s[-k:] == s[-2 * k:-k]:
            return True
    for seg in set(re.findall(r"\S{10,}", s)):
        if s.count(seg) >= 5:
            return True
    lines = [ln.strip() for ln in final.strip().splitlines() if ln.strip()]
    if len(lines) >= 4 and len(set(lines[-4:])) == 1:
        return True
    return bool(re.search(r"(.)\1{29,}", s))


def detect_anomalies(record: dict, final: str) -> list[str]:
    """Anomalies for one run record, each type reported separately."""
    raw = record.get("raw_output", "")
    flags: list[str] = []
    if not raw.strip() or not final.strip():
        flags.append("empty_output")
    if final:
        if printable_ratio(final) < 0.85:
            flags.append("garbled")
        if is_repetition_loop(final):
            flags.append("repetition_loop")
    if raw.count("\ufffd") >= 3:
        flags.append("replacement_chars")
    if sum(1 for c in raw if ord(c) < 32 and c not in "\n\t\r") >= 3:
        flags.append("control_chars")
    if record.get("truncated_suspect") or record.get("truncated"):
        flags.append("truncated")
    return flags


# ---------------------------------------------------------------------------
# checks (strict format constraints)
# ---------------------------------------------------------------------------


def eval_check(check: dict, final: str) -> bool:
    n = norm(final)
    kind = check["type"]
    if kind == "open":
        return True
    if kind == "contains":
        return any(norm(p) in n for p in check["patterns"])
    if kind == "number":
        # exactly one number in the whole final answer, and it is the value
        nums = re.findall(r"-?\d+(?:\.\d+)?", final)
        return len(nums) == 1 and abs(float(nums[0]) - float(check["value"])) < 1e-9
    if kind == "affirm":
        toks = tokens_alnum(final)
        joined = "".join(toks)
        if any(x in joined for x in NEGATIVES):
            return not check.get("affirmative", True)
        return check.get("affirmative", True) and any(
            x in joined for x in AFFIRMATIVES)
    if kind == "token":
        toks = tokens_alnum(final)
        return any(p in toks for p in check["patterns"])
    if kind == "one_word":
        words = [w for w in re.split(r"\s+", final.strip()) if w]
        return len(words) == 1 and norm(words[0]) in (
            [norm(p) for p in check["patterns"]])
    if kind == "yes_no":
        bare = norm(final)
        return bare in ("yes", "no") and bare == check["target"]
    if kind == "all_colors":
        # exactly the three requested colors: all present, no other color word
        if any(norm(w) in n for w in EXTRA_COLORS):
            return False
        return all(any(norm(w) in n for w in group)
                   for group in REQUIRED_COLOR_GROUPS)
    if kind == "max_chars":
        return len(n) <= check["max_chars"] and any(
            norm(p) in n for p in check["patterns"])
    raise ValueError(f"unknown check type: {kind}")


def is_correct(checks: list[dict], final: str) -> bool:
    return any(eval_check(c, final) for c in checks)


def score_side(record: dict, checks: list[dict]) -> tuple[bool, str, list[str]]:
    """Score one run record: (correct, final_text, blocking_anomalies).

    Blocking anomalies force ``correct=False`` regardless of keyword matches:
    a truncated or looping output never counts as an instruction success.
    """
    final = extract_final(record.get("raw_output", ""))
    anomalies = detect_anomalies(record, final)
    blocked = sorted(set(anomalies) & BLOCKING_ANOMALIES)
    correct = not blocked and is_correct(checks, final)
    return correct, final, blocked


# ---------------------------------------------------------------------------
# recorded runtime parameters (never taken from questions.json)
# ---------------------------------------------------------------------------


def recorded_params(records: dict[str, dict]) -> set[tuple]:
    return {
        (r.get("max_tokens"), r.get("temp"))
        for r in records.values()
    }


def assert_param_consistency(
    src: dict[str, dict], mlx: dict[str, dict]
) -> tuple[int, float]:
    """Both sides must have uniform, identical recorded max_tokens/temp."""
    src_params, mlx_params = recorded_params(src), recorded_params(mlx)
    if len(src_params) != 1 or len(mlx_params) != 1 or src_params != mlx_params:
        raise SystemExit(
            "runtime parameter mismatch across run records: "
            f"source={sorted(src_params)} mlx={sorted(mlx_params)}; "
            "reports must compare runs with identical recorded max_tokens/temp "
            "(re-run the eval; questions.json defaults are not evidence)"
        )
    max_tokens, temp = src_params.pop()
    return int(max_tokens), float(temp)


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in
            (json.loads(line) for line in path.read_text().splitlines()
             if line.strip())}


def _strip_trailing_ws(text: str) -> str:
    """Cosmetic only (report display): drop trailing spaces per line."""
    return re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)


def score_item(q: dict, rs: dict, rm: dict) -> dict:
    """Score one question from its two run records into a report item."""
    ans_s, fs, block_s = score_side(rs, q["checks"])
    ans_m, fm, block_m = score_side(rm, q["checks"])
    return {
        "id": q["id"], "lang": q["lang"], "category": q["category"],
        "prompt": q["prompt"], "gold": q["answer"],
        "source_output": fs, "mlx_output": fm,
        "source_correct": ans_s, "mlx_correct": ans_m,
        "source_blocked": block_s, "mlx_blocked": block_m,
        "source_anomalies": detect_anomalies(rs, fs),
        "mlx_anomalies": detect_anomalies(rm, fm),
        "agree": norm(fs) == norm(fm),
        "source_gen_s": rs.get("gen_s"), "mlx_gen_s": rm.get("gen_s"),
        "source_eval_s": rs.get("eval_s"), "source_load_s": rs.get("load_s"),
    }


def rescore_item(q: dict, item: dict) -> dict:
    """Re-evaluate a stored report item under the current check set.

    Generation-time fields (outputs, anomaly lists, blocking flags, timings)
    are properties of the original run and are preserved; only the
    rule-based correctness and surface agreement are recomputed.
    """
    fs, fm = item["source_output"], item["mlx_output"]
    updated = dict(item)
    updated["source_correct"] = (
        not item["source_blocked"] and is_correct(q["checks"], fs))
    updated["mlx_correct"] = (
        not item["mlx_blocked"] and is_correct(q["checks"], fm))
    updated["agree"] = norm(fs) == norm(fm)
    return updated


def build_report(questions: list[dict], per_q: list[dict], *, title: str,
                 source_desc: str, mlx_desc: str, note: str,
                 max_tokens: int, temp: float,
                 rescoring: dict | None = None) -> tuple[dict, str]:
    """Aggregate scored items into (report dict, report markdown)."""
    by_id = {r["id"]: r for r in per_q}
    scored = [q for q in questions if q["checks"][0]["type"] != "open"]
    open_qs = [q for q in questions if q["checks"][0]["type"] == "open"]

    def acc(side: str, qs=scored) -> float:
        return sum(1 for q in qs if by_id[q["id"]][f"{side}_correct"]) / len(qs)

    def anom_rate(side: str) -> float:
        return (sum(1 for q in questions if by_id[q["id"]][f"{side}_anomalies"])
                / len(questions))

    def anom_breakdown(side: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for q in questions:
            for flag in by_id[q["id"]][f"{side}_anomalies"]:
                counts[flag] = counts.get(flag, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    acc_s, acc_m = acc("source"), acc("mlx")
    agree_rate = sum(1 for r in per_q if r["agree"]) / len(per_q)
    verdict_agree = (sum(1 for q in scored
                         if by_id[q["id"]]["source_correct"]
                         == by_id[q["id"]]["mlx_correct"]) / len(scored))
    flips = [q["id"] for q in scored
             if by_id[q["id"]]["source_correct"] != by_id[q["id"]]["mlx_correct"]]

    cats = sorted({q["category"] for q in scored})
    cat_rows = [{
        "category": c,
        "source_correct": sum(1 for q in scored
                              if q["category"] == c and by_id[q["id"]]["source_correct"]),
        "mlx_correct": sum(1 for q in scored
                           if q["category"] == c and by_id[q["id"]]["mlx_correct"]),
        "n": sum(1 for q in scored if q["category"] == c),
    } for c in cats]

    def gen_stats(side: str) -> dict:
        vals = [r[f"{side}_gen_s"] for r in per_q if r[f"{side}_gen_s"] is not None]
        return {"total_s": round(sum(vals), 1),
                "mean_s": round(statistics.mean(vals), 2),
                "median_s": round(statistics.median(vals), 2)} if vals else {}

    summary = {
        "n_questions": len(questions),
        "n_scored": len(scored),
        "n_open": len(open_qs),
        "recorded_max_tokens": max_tokens,
        "recorded_temp": temp,
        # gated accuracy: blocking anomalies (truncation, repetition loops,
        # empty/garbled output) count as incorrect; accuracy == clean accuracy
        "source_accuracy": round(acc_s, 4),
        "mlx_accuracy": round(acc_m, 4),
        "accuracy_delta_mlx_minus_source": round(acc_m - acc_s, 4),
        "answer_agreement_rate": round(agree_rate, 4),
        "verdict_agreement_rate": round(verdict_agree, 4),
        "verdict_flips": flips,
        "source_anomaly_rate": round(anom_rate("source"), 4),
        "mlx_anomaly_rate": round(anom_rate("mlx"), 4),
        "source_anomaly_breakdown": anom_breakdown("source"),
        "mlx_anomaly_breakdown": anom_breakdown("mlx"),
        "per_category": cat_rows,
        "source_timing_s": gen_stats("source"),
        "mlx_timing_s": gen_stats("mlx"),
        "source_desc": source_desc,
        "mlx_desc": mlx_desc,
        "note": note,
    }
    if rescoring is not None:
        summary["rescoring"] = rescoring

    report = {"title": title, "summary": summary, "per_question": per_q}

    rescore_line = None
    if rescoring is not None:
        changed = rescoring["changed_verdicts"]
        changed_txt = ", ".join(
            f"{c['id']} (src {c['source'][0]}->{c['source'][1]}, "
            f"mlx {c['mlx'][0]}->{c['mlx'][1]})"
            for c in changed) or "none"
        prev = rescoring["previous"]

        def pct(value) -> str:
            return "n/a" if value is None else f"{value:.1%}"

        rescore_line = (
            f"- rescoring: {rescoring['date']} — {rescoring['reason']}; "
            f"changed verdicts: {changed_txt}; previous summary: "
            f"src {pct(prev.get('source_accuracy'))} / "
            f"mlx {pct(prev.get('mlx_accuracy'))}, "
            f"verdict agreement {pct(prev.get('verdict_agreement_rate'))}, "
            f"flips {prev.get('verdict_flips')}")

    lines = [f"# {title}", "",
             f"- source: {source_desc}",
             f"- MLX: {mlx_desc}",
             (f"- protocol: identical chat template semantics (GGUF-embedded "
              f"template vs the tokenizer files derived from it), single user "
              f"turn with no system prompt on both sides, temp {temp:g}, "
              f"max_tokens {max_tokens} (recorded in every run record and "
              f"asserted identical across sides), thinking enabled (template "
              f"default) on both sides, scored over the post-<think> text"),
             ("- scoring gate: truncated / repetition-loop / empty / garbled "
              "outputs count as incorrect regardless of contained keywords; "
              "format checks are strict (single number, exactly three colors, "
              "one word, yes/no)"),
             *([f"- note: {note}"] if note else []),
             *([rescore_line] if rescore_line else []),
             "",
             "## Summary", "",
             "| metric | source | MLX |",
             "|---|---|---|",
             f"| gated accuracy ({len(scored)} scored) | {acc_s:.1%} | {acc_m:.1%} |",
             f"| accuracy delta (MLX − source) | | {acc_m - acc_s:+.1%} |",
             (f"| anomaly rate ({len(questions)} items) "
              f"| {anom_rate('source'):.1%} | {anom_rate('mlx'):.1%} |"),
             (f"| total gen time (s) | {summary['source_timing_s'].get('total_s')} "
              f"| {summary['mlx_timing_s'].get('total_s')} |"),
             "",
             (f"Answer agreement rate (surface form): **{agree_rate:.1%}** · "
              f"verdict agreement on scored items: **{verdict_agree:.1%}** "
              f"(flips: {', '.join(flips) if flips else 'none'})"), "",
             "Anomaly breakdown (items affected, per type):", "",
             "| type | source | MLX |", "|---|---|---|"]
    for flag in sorted(set(summary["source_anomaly_breakdown"])
                       | set(summary["mlx_anomaly_breakdown"])):
        lines.append(
            f"| {flag} | {summary['source_anomaly_breakdown'].get(flag, 0)} "
            f"| {summary['mlx_anomaly_breakdown'].get(flag, 0)} |")
    lines += ["", "| category | n | source | MLX |", "|---|---|---|---|"]
    for c in cat_rows:
        lines.append(f"| {c['category']} | {c['n']} "
                     f"| {c['source_correct']}/{c['n']} | {c['mlx_correct']}/{c['n']} |")
    lines += ["", "## Per-question results", ""]
    for r in per_q:
        def mark(ok: bool, flags: list[str]) -> str:
            return ("⚠" + ",".join(flags)) if flags else ("✓" if ok else "✗")
        lines += [(f"### {r['id']} — src {mark(r['source_correct'], r['source_blocked'])}"
                   f" / mlx {mark(r['mlx_correct'], r['mlx_blocked'])}"
                   f" / agree {'yes' if r['agree'] else 'NO'}"),
                  f"- gold: {r['gold']}",
                  f"- source ({r['source_gen_s']}s):",
                  "```", _strip_trailing_ws(r["source_output"]) or "(empty)", "```",
                  f"- mlx ({r['mlx_gen_s']}s):",
                  "```", _strip_trailing_ws(r["mlx_output"]) or "(empty)", "```", ""]
    return report, "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=None,
                    help="llama.cpp JSONL (fresh scoring runs)")
    ap.add_argument("--mlx", type=Path, default=None, help="MLX JSONL (fresh scoring runs)")
    ap.add_argument("--rescore-report", type=Path, default=None,
                    help="rescoring mode: re-evaluate an existing report.json "
                         "under the current questions.json checks, reusing the "
                         "embedded per-item outputs; exclusive with --source/--mlx")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--source-desc", default=None)
    ap.add_argument("--mlx-desc", default=None)
    ap.add_argument("--allow-partial", action="store_true",
                    help="score only questions present in both JSONLs "
                         "(smoke tests); default requires full coverage")
    ap.add_argument("--note", default=None,
                    help="free-form note recorded in the report (e.g. "
                         "system conditions during the run); in rescoring "
                         "mode this is the rescoring reason")
    args = ap.parse_args()

    qdata = json.loads(args.questions.read_text())
    questions = qdata["questions"]

    if args.rescore_report:
        if args.source or args.mlx:
            raise SystemExit("--rescore-report is exclusive with --source/--mlx")
        report = json.loads(args.rescore_report.read_text())
        stored = {r["id"]: r for r in report["per_question"]}
        missing = [q["id"] for q in questions if q["id"] not in stored]
        if missing and not args.allow_partial:
            raise SystemExit(f"missing records for: {missing[:5]} ... ({len(missing)} total)")
        questions = [q for q in questions if q["id"] in stored]
        old_summary = report["summary"]
        max_tokens = int(old_summary["recorded_max_tokens"])
        temp = float(old_summary["recorded_temp"])
        note = old_summary.get("note", "")  # preserved; --note is the reason
        changed = []
        per_q = []
        for q in questions:
            old = stored[q["id"]]
            new = rescore_item(q, old)
            if (old["source_correct"], old["mlx_correct"]) != (
                    new["source_correct"], new["mlx_correct"]):
                changed.append({
                    "id": q["id"],
                    "source": [old["source_correct"], new["source_correct"]],
                    "mlx": [old["mlx_correct"], new["mlx_correct"]],
                })
            per_q.append(new)
        rescoring = {
            "date": time.strftime("%Y-%m-%d"),
            "reason": args.note or ("questions/checks updated; verdicts "
                                    "recomputed from embedded outputs"),
            "changed_verdicts": changed,
            "previous": {k: old_summary.get(k) for k in (
                "source_accuracy", "mlx_accuracy", "answer_agreement_rate",
                "verdict_agreement_rate", "verdict_flips")},
        }
        title = args.title or report["title"]
        source_desc = args.source_desc or old_summary.get("source_desc") or ""
        mlx_desc = args.mlx_desc or old_summary.get("mlx_desc") or ""
    else:
        if not (args.source and args.mlx):
            raise SystemExit(
                "--source and --mlx are required unless --rescore-report is given")
        if not (args.title and args.source_desc and args.mlx_desc):
            raise SystemExit(
                "--title/--source-desc/--mlx-desc are required for a fresh scoring run")
        src = load_jsonl(args.source)
        mlx = load_jsonl(args.mlx)
        missing = [q["id"] for q in questions
                   if q["id"] not in src or q["id"] not in mlx]
        if missing and not args.allow_partial:
            raise SystemExit(f"missing records for: {missing[:5]} ... ({len(missing)} total)")
        questions = [q for q in questions if q["id"] in src and q["id"] in mlx]
        # hard gate: parameters come from the run records and must match exactly
        max_tokens, temp = assert_param_consistency(src, mlx)
        per_q = [score_item(q, src[q["id"]], mlx[q["id"]]) for q in questions]
        rescoring = None
        note = args.note or ""
        title, source_desc, mlx_desc = args.title, args.source_desc, args.mlx_desc

    report, md = build_report(
        questions, per_q, title=title, source_desc=source_desc,
        mlx_desc=mlx_desc, note=note, max_tokens=max_tokens, temp=temp,
        rescoring=rescoring)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.out_dir / "report.md").write_text(md)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    main()
