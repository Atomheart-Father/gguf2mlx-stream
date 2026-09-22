#!/usr/bin/env python3
"""Score ARC benchmark runs and apply the promotion gate.

Metrics per side: accuracy (extracted letter == answer key), anomaly rate
(no letter / empty / repetition loop / truncation-without-letter), clean
accuracy (correct AND anomaly-free), letter-only compliance, timing.

Gate (initial, for promoting a target-bits default): a candidate's clean
accuracy must not be more than 5 percentage points below the source's, and
its anomaly rate must not exceed the source's by more than 2 percentage
points. Candidates failing the gate are labeled "experimental" — never a
recommended configuration.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

GATE_CLEAN_ACCURACY_PP = 5.0
GATE_ANOMALY_PP = 2.0


def anomalies(rec: dict) -> list[str]:
    """Blocking anomalies for a letter-answer task.

    ``repetition_loop`` blocks only when the run also hit the token cap
    (looped into the cap): a correct letter that was emitted before the
    output degenerated into an endless loop is not a clean success —
    mirroring the capability-eval rule that keyword-then-loop is not an
    instruction success. Truncation without a letter means no answer was
    produced at all.
    """
    flags: list[str] = []
    if not rec.get("raw_output", "").strip():
        flags.append("empty_output")
    if rec.get("letter") is None:
        flags.append("no_letter")
    if rec.get("repetition") and rec.get("truncated"):
        flags.append("repetition_loop")
    if rec.get("truncated") and rec.get("letter") is None:
        flags.append("truncated")
    return flags


def load_records(path: Path, name: str) -> dict[str, dict]:
    """Load one side's records: strict JSONL — duplicate IDs are rejected."""
    records: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        qid = rec.get("id")
        if qid is None:
            raise SystemExit(f"{name}: record without 'id' in {path}")
        if qid in records:
            raise SystemExit(f"{name}: duplicate record id {qid!r} in {path}")
        records[qid] = rec
    if not records:
        raise SystemExit(f"{name}: no records in {path}")
    return records


def assert_side_consistency(
    sides: dict[str, dict[str, dict]],
) -> tuple[int, float]:
    """All sides must cover the same questions under the same run parameters.

    Rejects: missing questions on any side, differing answer keys, differing
    prompt hashes (the rendered prompt is part of the protocol), and
    differing temp / max_tokens run parameters.
    """
    id_sets = {name: set(recs) for name, recs in sides.items()}
    reference = next(iter(sides))
    for name, ids in id_sets.items():
        missing = id_sets[reference] - ids
        extra = ids - id_sets[reference]
        if missing or extra:
            raise SystemExit(
                f"question coverage mismatch: {reference} vs {name} — "
                f"missing from {name}: {sorted(missing)[:5]} "
                f"(+{len(missing)}), unexpected extra: {sorted(extra)[:5]} "
                f"(+{len(extra)}); every side must complete the same subset"
            )
    for qid in sorted(id_sets[reference]):
        keys = {name: recs[qid].get("answer_key") for name, recs in sides.items()}
        if len(set(map(str, keys.values()))) != 1:
            raise SystemExit(f"{qid}: answer key differs across sides: {keys}")
        hashes = {name: recs[qid].get("prompt_sha256") for name, recs in sides.items()}
        if len(set(map(str, hashes.values()))) != 1:
            raise SystemExit(
                f"{qid}: prompt hash differs across sides: {hashes}; "
                "both engines must receive the identical rendered prompt"
            )
    for param in ("temp", "max_tokens"):
        values = {
            name: sorted({str(r.get(param)) for r in recs.values()})
            for name, recs in sides.items()
        }
        if any(len(v) != 1 for v in values.values()):
            raise SystemExit(f"{param} is not uniform within every side: {values}")
        if len({v[0] for v in values.values()}) != 1:
            raise SystemExit(f"{param} differs across sides: {values}")
    ref_recs = next(iter(sides.values()))
    max_tokens = int(float(next(iter(ref_recs.values()))["max_tokens"]))
    temp = float(next(iter(ref_recs.values()))["temp"])
    return max_tokens, temp


def score_side(records_path: Path) -> dict:
    recs = [json.loads(line) for line in records_path.read_text().splitlines()
            if line.strip()]
    n = len(recs)
    if not n:
        raise SystemExit(f"empty records: {records_path}")
    correct = sum(1 for r in recs
                  if r.get("letter") == r.get("answer_key"))
    anom = [a for r in recs for a in anomalies(r)]
    anom_items = sum(1 for r in recs if anomalies(r))
    gen = [r["gen_s"] for r in recs if r.get("gen_s") is not None]
    breakdown: dict[str, int] = {}
    for r in recs:
        for a in anomalies(r):
            breakdown[a] = breakdown.get(a, 0) + 1
    return {
        "n": n,
        "accuracy": correct / n,
        "clean_accuracy": sum(1 for r in recs
                              if r.get("letter") == r.get("answer_key")
                              and not anomalies(r)) / n,
        "anomaly_rate": anom_items / n,
        "anomaly_breakdown": dict(sorted(breakdown.items(),
                                         key=lambda kv: -kv[1])),
        "anomaly_events": len(anom),
        # letter-only compliance: a valid letter was produced AND nothing else
        "letter_only_compliance": sum(1 for r in recs
                                      if r.get("letter") is not None
                                      and not r.get("extra_text")) / n,
        "median_gen_s": round(statistics.median(gen), 3) if gen else None,
        "total_gen_s": round(sum(gen), 1) if gen else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", action="append", required=True,
                    metavar="NAME=RECORDS.jsonl",
                    help="named side; repeat for every candidate")
    ap.add_argument("--meta", action="append", default=[],
                    metavar="NAME=META.json",
                    help="optional conversion metrics per side "
                         "(output_gib, peak_rss_gib, elapsed_s)")
    ap.add_argument("--source-side", required=True,
                    help="name of the side the gate compares against")
    ap.add_argument("--title", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    sides: dict[str, dict] = {}
    raw_sides: dict[str, dict[str, dict]] = {}
    paths: dict[str, Path] = {}
    for spec in args.side:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--side expects NAME=PATH, got {spec!r}")
        paths[name] = Path(path)
        raw_sides[name] = load_records(paths[name], name)
    max_tokens, temp = assert_side_consistency(raw_sides)
    for name, path in paths.items():
        sides[name] = score_side(path)
    if args.source_side not in sides:
        raise SystemExit(f"source side {args.source_side!r} not among sides")
    metas: dict[str, dict] = {}
    for spec in args.meta:
        name, _, path = spec.partition("=")
        metas[name] = json.loads(Path(path).read_text())

    src = sides[args.source_side]
    verdicts = {}
    for name, s in sides.items():
        if name == args.source_side:
            verdicts[name] = "source (reference)"
            continue
        acc_gap_pp = (src["clean_accuracy"] - s["clean_accuracy"]) * 100
        anom_gap_pp = (s["anomaly_rate"] - src["anomaly_rate"]) * 100
        ok = (acc_gap_pp <= GATE_CLEAN_ACCURACY_PP
              and anom_gap_pp <= GATE_ANOMALY_PP)
        verdicts[name] = ("PASS gate" if ok else
                          "FAIL gate — experimental result only, not a "
                          "recommended configuration")
        s["gate"] = {"clean_accuracy_gap_pp": round(acc_gap_pp, 2),
                     "anomaly_rate_gap_pp": round(anom_gap_pp, 2),
                     "pass": ok}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "title": args.title,
        "source_side": args.source_side,
        "recorded_max_tokens": max_tokens,
        "recorded_temp": temp,
        "gate": {"max_clean_accuracy_drop_pp": GATE_CLEAN_ACCURACY_PP,
                 "max_anomaly_rate_increase_pp": GATE_ANOMALY_PP},
        "sides": sides,
        "conversion_meta": metas,
        "verdicts": verdicts,
    }
    (args.out_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    lines = [f"# {args.title}", "",
             (f"Protocol evidence: {len(raw_sides)} side(s) × "
              f"{len(next(iter(raw_sides.values())))} questions, identical "
              f"question coverage, answer keys, prompt hashes; recorded "
              f"temp {temp:g}, max_tokens {max_tokens}."), "",
             (f"Gate: candidate clean accuracy ≥ source − "
              f"{GATE_CLEAN_ACCURACY_PP:g} pp AND anomaly rate ≤ source + "
              f"{GATE_ANOMALY_PP:g} pp. Failing candidates are experimental "
              f"results, never recommended configurations."), "",
             "| side | accuracy | clean accuracy | anomaly rate | letter-only | median gen (s) | size | conv RSS | conv time | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in sides.items():
        m = metas.get(name, {})
        lines.append(
            f"| {name} | {s['accuracy']:.1%} | {s['clean_accuracy']:.1%} "
            f"| {s['anomaly_rate']:.1%} | {s['letter_only_compliance']:.1%} "
            f"| {s['median_gen_s']} "
            f"| {m.get('output_gib', '—')} | {m.get('peak_rss_gib', '—')} "
            f"| {m.get('elapsed_s', '—')} | {verdicts[name]} |")
    lines += ["", "Anomaly breakdown per side:", ""]
    for name, s in sides.items():
        lines.append(f"- **{name}**: {s['anomaly_breakdown'] or 'none'}")
    (args.out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdicts": verdicts,
                      "clean_accuracy": {k: v["clean_accuracy"]
                                         for k, v in sides.items()},
                      "anomaly_rate": {k: v["anomaly_rate"]
                                       for k, v in sides.items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    main()
