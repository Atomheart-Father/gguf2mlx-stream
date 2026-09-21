"""Tensor-level paired-oracle comparison: our conversion vs a reference MLX repo.

Primary use: prove BF16-GGUF -> unquantized-MLX conversion semantics by
comparing every tensor against the official mlx-community BF16 export of the
same base model. Also usable for any (ours, reference) pair at other bit
widths with looser expectations.

Comparison semantics
--------------------
* Name resolution: exact match first; then a fallback that strips the
  ``language_model.`` prefix on either side (VLM-style vs text-only naming).
* Stats per matched tensor: shape/dtype checks, max abs diff, mean abs diff,
  relative L2, cosine similarity, NaN/Inf counts. Both sides are widened to
  float32 for the arithmetic.
* Bit-exactness is *expected* for pure copy tensors (identical bf16 payloads);
  arithmetic transforms (e.g. ``A_log = log(-x)``) may differ in the last
  ulp. The report records both ``bit_exact`` and float stats; interpretation
  happens at report level, not here.
* Config comparison: ``--config-report`` additionally diffs the two
  ``config.json`` files field-by-field (recursive), which validates emitted
  runtime metadata, not tensor bytes.

Usage::

    python -m research.paired_oracle.compare_bf16 \
        --ours  <cache>/out/qwen35-0.8b-bf16-ours \
        --reference <cache>/mlx/qwen35-0.8b-instruct-bf16 \
        --report <cache>/reports/bf16_oracle_report.json \
        [--md <cache>/reports/bf16_oracle_report.md] [--config-report]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

_LANGUAGE_PREFIX = "language_model."


def load_model_tensors(model_dir: Path) -> dict[str, mx.array]:
    tensors: dict[str, mx.array] = {}
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors shards under {model_dir}")
    for shard in shards:
        for name, value in mx.load(str(shard)).items():
            tensors[name] = value
    return tensors


def resolve_names(
    ours: dict[str, mx.array], reference: dict[str, mx.array]
) -> tuple[dict[str, str], list[str], list[str]]:
    """Map reference-side names to ours-side names.

    Returns (mapping reference->ours, unmatched_reference, unmatched_ours).
    """
    mapping: dict[str, str] = {}
    used: set[str] = set()
    unmatched_ref: list[str] = []
    for name, value in reference.items():
        if name in ours:
            mapping[name] = name
            used.add(name)
            continue
        stripped = name.removeprefix(_LANGUAGE_PREFIX)
        alt = _LANGUAGE_PREFIX + name if not name.startswith(_LANGUAGE_PREFIX) else stripped
        if alt in ours:
            mapping[name] = alt
            used.add(alt)
            continue
        unmatched_ref.append(name)
    unmatched_ours = sorted(set(ours) - used)
    return mapping, unmatched_ref, unmatched_ours


def _scalar_stats(a: mx.array, b: mx.array) -> dict:
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    diff = af - bf
    abs_diff = mx.abs(diff)
    rel_l2 = mx.sqrt(mx.sum(diff * diff) / mx.sum(bf * bf)) if mx.sum(bf * bf) > 0 else float("nan")
    dot = mx.sum(af * bf)
    na = mx.sqrt(mx.sum(af * af))
    nb = mx.sqrt(mx.sum(bf * bf))
    cos = dot / (na * nb) if na > 0 and nb > 0 else float("nan")
    # Semantic-equality views. Normalizing both sides to a common storage
    # format separates *structural/transform* errors (which survive the cast)
    # from storage-rounding artifacts (which do not):
    # - after bf16 cast: reference-format view; f16-storage artifacts of our
    #   side (subnormal flush of tiny values) remain visible.
    # - after f16 cast: our-format view; when this is exact, the reference
    #   carries no information our output lost.
    abf = af.astype(mx.bfloat16).astype(mx.float32)
    bbf = bf.astype(mx.bfloat16).astype(mx.float32)
    a16 = af.astype(mx.float16).astype(mx.float32)
    b16 = bf.astype(mx.float16).astype(mx.float32)
    return {
        "max_abs": float(mx.max(abs_diff)),
        "mean_abs": float(mx.mean(abs_diff)),
        "rel_l2": float(rel_l2),
        "cosine": float(cos),
        "nan_ours": int(mx.sum(mx.isnan(af))),
        "nan_ref": int(mx.sum(mx.isnan(bf))),
        "inf_ours": int(mx.sum(mx.isinf(af))),
        "inf_ref": int(mx.sum(mx.isinf(bf))),
        "equal_after_bf16_cast": bool(mx.array_equal(abf, bbf)),
        "max_abs_after_bf16_cast": float(mx.max(mx.abs(abf - bbf))),
        "equal_after_f16_cast": bool(mx.array_equal(a16, b16)),
        "max_abs_after_f16_cast": float(mx.max(mx.abs(a16 - b16))),
        "flushed_to_zero_ours": int(mx.sum((bf != 0) & (a16 == 0))),
    }


def compare_tensor(name: str, ours: mx.array, reference: mx.array) -> dict:
    rec: dict = {
        "shape_ours": list(ours.shape),
        "shape_ref": list(reference.shape),
        "dtype_ours": str(ours.dtype),
        "dtype_ref": str(reference.dtype),
    }
    if list(ours.shape) != list(reference.shape):
        rec["status"] = "shape_mismatch"
        return rec
    bit_exact = bool(
        str(ours.dtype) == str(reference.dtype) and mx.array_equal(ours, reference)
    )
    rec["bit_exact"] = bit_exact
    rec.update(_scalar_stats(ours, reference))
    rec["status"] = "ok"
    return rec


def compare_config(ours_dir: Path, reference_dir: Path) -> dict:
    def flatten(prefix: str, value: object, out: dict) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                flatten(f"{prefix}.{k}" if prefix else k, v, out)
        else:
            out[prefix] = value

    ours_cfg = json.loads((ours_dir / "config.json").read_text())
    ref_cfg = json.loads((reference_dir / "config.json").read_text())
    f_ours: dict = {}
    f_ref: dict = {}
    flatten("", ours_cfg, f_ours)
    flatten("", ref_cfg, f_ref)
    only_ours = sorted(set(f_ours) - set(f_ref))
    only_ref = sorted(set(f_ref) - set(f_ours))
    differing = sorted(
        k for k in set(f_ours) & set(f_ref) if f_ours[k] != f_ref[k]
    )
    return {
        "only_ours": {k: f_ours[k] for k in only_ours},
        "only_reference": {k: f_ref[k] for k in only_ref},
        "differing": {k: {"ours": f_ours[k], "reference": f_ref[k]} for k in differing},
        "counts": {
            "ours": len(f_ours),
            "reference": len(f_ref),
            "common": len(set(f_ours) & set(f_ref)),
            "differing": len(differing),
        },
    }


def tensor_class(name: str) -> str:
    """Coarse class for expected-difference interpretation."""
    if "A_log" in name:
        return "a_log_transform"
    if name.endswith(("norm.weight", "norm.bias", "dt_bias", "conv1d.weight")) and (
        "linear_attn" in name or "norm" in name
    ):
        return "norm_or_small"
    return "copy"


def write_md(report: dict, path: Path) -> None:
    s = report["summary"]
    lines = [
        "# BF16 paired-oracle tensor comparison",
        "",
        f"- ours: `{report['ours_dir']}`",
        f"- reference: `{report['reference_dir']}`",
        f"- matched tensors: **{s['matched']}** "
        f"(bit-exact: **{s['bit_exact']}**, "
        f"equal after bf16 cast: **{s['equal_after_bf16_cast']}**, "
        f"equal after f16 cast: **{s['equal_after_f16_cast']}**, "
        f"values our f16 flushed to zero: {s['values_flushed_to_zero_ours']})",
        f"- unmatched reference: {s['unmatched_reference']}",
        f"- unmatched ours: {s['unmatched_ours']}",
        f"- shape mismatches: {s['shape_mismatch']}",
        "",
        "| tensor | class | bit_exact | max_abs | rel_l2 | cosine | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, rec in sorted(report["tensors"].items()):
        if rec.get("status") != "ok":
            lines.append(
                f"| {name} | - | - | - | - | - | {rec.get('status')} |"
            )
            continue
        note = ""
        if rec["max_abs"] > 0 and rec["max_abs"] < 1e-2:
            note = "ulp-level"
        if rec["nan_ours"] or rec["nan_ref"] or rec["inf_ours"] or rec["inf_ref"]:
            note = "NON-FINITE"
        lines.append(
            f"| {name} | {tensor_class(name)} | {rec['bit_exact']} "
            f"| {rec['max_abs']:.3g} | {rec['rel_l2']:.3g} "
            f"| {rec['cosine']:.9f} | {note} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--md")
    parser.add_argument("--config-report", action="store_true",
                        help="also diff config.json semantics")
    args = parser.parse_args(argv)

    ours_dir = Path(args.ours).expanduser()
    ref_dir = Path(args.reference).expanduser()
    print(f"loading ours from {ours_dir} ...")
    ours = load_model_tensors(ours_dir)
    print(f"loading reference from {ref_dir} ...")
    reference = load_model_tensors(ref_dir)

    mapping, unmatched_ref, unmatched_ours = resolve_names(ours, reference)
    print(f"matched {len(mapping)} tensors "
          f"(unmatched ref={len(unmatched_ref)}, ours={len(unmatched_ours)})")

    tensors: dict[str, dict] = {}
    for ref_name, our_name in sorted(mapping.items()):
        tensors[ref_name] = compare_tensor(ref_name, ours[our_name], reference[ref_name])
        if tensors[ref_name]["status"] == "shape_mismatch":
            print(f"SHAPE MISMATCH {ref_name}: "
                  f"{tensors[ref_name]['shape_ours']} vs {tensors[ref_name]['shape_ref']}")

    bit_exact = sum(1 for r in tensors.values() if r.get("bit_exact"))
    eq_bf16 = sum(
        1 for r in tensors.values()
        if r.get("status") == "ok" and r.get("equal_after_bf16_cast")
    )
    eq_f16 = sum(
        1 for r in tensors.values()
        if r.get("status") == "ok" and r.get("equal_after_f16_cast")
    )
    flushed = sum(
        int(r.get("flushed_to_zero_ours", 0)) for r in tensors.values()
    )
    report = {
        "ours_dir": str(ours_dir),
        "reference_dir": str(ref_dir),
        "summary": {
            "matched": len(mapping),
            "bit_exact": bit_exact,
            "equal_after_bf16_cast": eq_bf16,
            "equal_after_f16_cast": eq_f16,
            "values_flushed_to_zero_ours": flushed,
            "unmatched_reference": unmatched_ref,
            "unmatched_ours": unmatched_ours,
            "shape_mismatch": sum(
                1 for r in tensors.values() if r.get("status") == "shape_mismatch"
            ),
        },
        "tensors": tensors,
    }
    if args.config_report:
        report["config"] = compare_config(ours_dir, ref_dir)

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.md:
        write_md(report, Path(args.md))
    print(f"report: {out_path}"
          + (f" / {args.md}" if args.md else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
