#!/usr/bin/env python3
"""Score a capability-regression run: source (llama.cpp) vs MLX output.

Consumes the two JSONL files produced by run_eval.py and emits:

* ``report.json`` — full machine-readable results (committed)
* ``report.md``   — per-question outputs, verdicts, and summary metrics (committed)

Metrics: per-side accuracy (scored items only), accuracy delta, answer
agreement rate, anomaly rate (empty / garbled / suspect truncation), and
generation timing. Open items are screened for anomalies only, never scored.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from pathlib import Path

NEGATIVES = ["不", "别", "无", "no", "not", "false", "否", "错", "wrong", "non"]
AFFIRMATIVES = ["是", "对", "是的", "yes", "true", "correct", "right", "的确"]


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


def eval_check(check: dict, final: str) -> bool:
    n = norm(final)
    kind = check["type"]
    if kind == "open":
        return True
    if kind == "contains":
        return any(norm(p) in n for p in check["patterns"])
    if kind == "number":
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", final)]
        return any(abs(x - float(check["value"])) < 1e-9 for x in nums)
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
        return all(any(norm(w) in n for w in group)
                   for group in (("red", "红"), ("green", "绿"), ("blue", "蓝")))
    if kind == "max_chars":
        return len(n) <= check["max_chars"] and any(
            norm(p) in n for p in check["patterns"])
    raise ValueError(f"unknown check type: {kind}")


def is_correct(checks: list[dict], final: str) -> bool:
    return any(eval_check(c, final) for c in checks)


def detect_anomalies(raw: str, final: str, truncated_suspect: bool) -> list[str]:
    flags = []
    if not raw.strip():
        flags.append("empty_raw")
    elif not final:
        flags.append("empty_after_think_extract")
    if final and printable_ratio(final) < 0.85:
        flags.append("garbled")
    if raw.count("\ufffd") >= 3:
        flags.append("replacement_chars")
    if sum(1 for c in raw if ord(c) < 32 and c not in "\n\t\r") >= 3:
        flags.append("control_chars")
    if truncated_suspect:
        flags.append("truncated_suspect")
    return flags


def load_jsonl(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in
            (json.loads(line) for line in path.read_text().splitlines()
             if line.strip())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True,
                    help="llama.cpp JSONL")
    ap.add_argument("--mlx", type=Path, required=True, help="MLX JSONL")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--source-desc", required=True)
    ap.add_argument("--mlx-desc", required=True)
    ap.add_argument("--allow-partial", action="store_true",
                    help="score only questions present in both JSONLs "
                         "(smoke tests); default requires full coverage")
    ap.add_argument("--note", default="",
                    help="free-form note recorded in the report (e.g. "
                         "system conditions during the run)")
    args = ap.parse_args()

    qdata = json.loads(args.questions.read_text())
    questions = qdata["questions"]
    src = load_jsonl(args.source)
    mlx = load_jsonl(args.mlx)
    missing = [q["id"] for q in questions
               if q["id"] not in src or q["id"] not in mlx]
    if missing and not args.allow_partial:
        raise SystemExit(f"missing records for: {missing[:5]} ... ({len(missing)} total)")
    questions = [q for q in questions if q["id"] in src and q["id"] in mlx]

    per_q = []
    scored = [q for q in questions if q["checks"][0]["type"] != "open"]
    open_qs = [q for q in questions if q["checks"][0]["type"] == "open"]

    for q in questions:
        rs, rm = src[q["id"]], mlx[q["id"]]
        fs, fm = extract_final(rs["raw_output"]), extract_final(rm["raw_output"])
        truncated_s = bool(rs.get("truncated_suspect"))
        truncated_m = bool(rm.get("truncated_suspect"))
        ans_s, ans_m = is_correct(q["checks"], fs), is_correct(q["checks"], fm)
        item = {
            "id": q["id"], "lang": q["lang"], "category": q["category"],
            "prompt": q["prompt"], "gold": q["answer"],
            "source_output": fs, "mlx_output": fm,
            "source_correct": ans_s, "mlx_correct": ans_m,
            "agree": norm(fs) == norm(fm),
            "source_anomalies": detect_anomalies(rs["raw_output"], fs, truncated_s),
            "mlx_anomalies": detect_anomalies(rm["raw_output"], fm, truncated_m),
            "source_gen_s": rs.get("gen_s"), "mlx_gen_s": rm.get("gen_s"),
            "source_eval_s": rs.get("eval_s"), "source_load_s": rs.get("load_s"),
        }
        per_q.append(item)

    def acc(side: str, qs=scored) -> float:
        by_id = {r["id"]: r for r in per_q}
        return sum(1 for q in qs if by_id[q["id"]][f"{side}_correct"]) / len(qs)

    def anom_rate(side: str) -> float:
        by_id = {r["id"]: r for r in per_q}
        return (sum(1 for q in questions if by_id[q["id"]][f"{side}_anomalies"])
                / len(questions))

    acc_s, acc_m = acc("source"), acc("mlx")
    agree_rate = sum(1 for r in per_q if r["agree"]) / len(per_q)
    by_id = {r["id"]: r for r in per_q}
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
        "source_accuracy": round(acc_s, 4),
        "mlx_accuracy": round(acc_m, 4),
        "accuracy_delta_mlx_minus_source": round(acc_m - acc_s, 4),
        "answer_agreement_rate": round(agree_rate, 4),
        "verdict_agreement_rate": round(verdict_agree, 4),
        "verdict_flips": flips,
        "source_anomaly_rate": round(anom_rate("source"), 4),
        "mlx_anomaly_rate": round(anom_rate("mlx"), 4),
        "per_category": cat_rows,
        "source_timing_s": gen_stats("source"),
        "mlx_timing_s": gen_stats("mlx"),
        "source_desc": args.source_desc,
        "mlx_desc": args.mlx_desc,
        "note": args.note,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"title": args.title, "summary": summary, "per_question": per_q}
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    lines = [f"# {args.title}", "",
             f"- source: {args.source_desc}",
             f"- MLX: {args.mlx_desc}",
             f"- protocol: identical chat template semantics (GGUF-embedded "
             f"template vs the tokenizer files derived from it), single user "
             f"turn with no system prompt on both sides, temp 0, "
             f"max_tokens {qdata['max_tokens']}, thinking enabled (template "
             f"default) on both sides, scored over the post-<think> text",
             *( [f"- note: {args.note}"] if args.note else [] ),
             "",
             "## Summary", "",
             "| metric | source | MLX |",
             "|---|---|---|",
             f"| accuracy ({len(scored)} scored) | {acc_s:.1%} | {acc_m:.1%} |",
             f"| accuracy delta (MLX − source) | | {acc_m - acc_s:+.1%} |",
             f"| anomaly rate ({len(questions)} items) "
             f"| {anom_rate('source'):.1%} | {anom_rate('mlx'):.1%} |",
             f"| total gen time (s) | {summary['source_timing_s'].get('total_s')} "
             f"| {summary['mlx_timing_s'].get('total_s')} |", "",
             f"Answer agreement rate (surface form): **{agree_rate:.1%}** · "
             f"verdict agreement on scored items: **{verdict_agree:.1%}** "
             f"(flips: {', '.join(flips) if flips else 'none'})", "",
             "| category | n | source | MLX |", "|---|---|---|---|"]
    for c in cat_rows:
        lines.append(f"| {c['category']} | {c['n']} "
                     f"| {c['source_correct']}/{c['n']} | {c['mlx_correct']}/{c['n']} |")
    lines += ["", "## Per-question results", ""]
    for r in per_q:
        def mark(ok: bool, flags: list[str]) -> str:
            return ("⚠" + ",".join(flags)) if flags else ("✓" if ok else "✗")
        lines += [f"### {r['id']} — src {mark(r['source_correct'], r['source_anomalies'])}"
                  f" / mlx {mark(r['mlx_correct'], r['mlx_anomalies'])}"
                  f" / agree {'yes' if r['agree'] else 'NO'}",
                  f"- gold: {r['gold']}",
                  f"- source ({r['source_gen_s']}s):",
                  "```", r["source_output"] or "(empty)", "```",
                  f"- mlx ({r['mlx_gen_s']}s):",
                  "```", r["mlx_output"] or "(empty)", "```", ""]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    main()
