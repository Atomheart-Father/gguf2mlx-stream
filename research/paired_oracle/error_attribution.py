"""Per-module quantization error attribution against the same-family BF16.

For an MLX quantized model directory this dequantizes every quantized
tensor (using the per-tensor bits/group_size recorded in config.json)
and compares against the corresponding tensor of a same-family BF16
reference: relative L2 per module class, aggregated. This ranks module
classes by quantization sensitivity — the empirical driver for mixed
profile design.

Usage::

    python -m research.paired_oracle.error_attribution \
        --quantized <quant_model_dir> --bf16 <bf16_reference_dir> \
        --report <out.json> [--md <out.md>]

The two directories must share the same tensor naming (same base model
family). Name normalization strips a leading ``language_model.`` on
either side, so text-only and VLM-style exports compare directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

from .profile_extract import classify, extract as extract_profile
from .compare_bf16 import load_model_tensors, _LANGUAGE_PREFIX

_DEQUANT_SHARD_CACHE: dict[str, tuple] = {}


def _dequantized_tensors(model_dir: Path) -> dict[str, mx.array]:
    """Load every tensor, dequantizing per the config.json quantization map."""
    cfg = json.loads((model_dir / "config.json").read_text())
    quant = cfg.get("quantization", {})
    global_bits = quant.get("bits")
    global_group = quant.get("group_size", 64)
    per_tensor_params: dict[str, tuple[int, int]] = {}
    for key, value in quant.items():
        if key in ("bits", "group_size", "mode") or not isinstance(value, dict):
            continue
        if "bits" in value:
            per_tensor_params[key] = (
                int(value["bits"]),
                int(value.get("group_size", global_group)),
            )

    tensors: dict[str, mx.array] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        for name, value in mx.load(str(shard)).items():
            tensors[name] = value

    # group packed/scales/biases: "X.weight" packs; siblings are "X.scales"/"X.biases"
    out: dict[str, mx.array] = {}
    consumed: set[str] = set()
    for name in list(tensors):
        if name.endswith(".scales") or name.endswith(".biases"):
            consumed.add(name)
            continue
        if name in consumed:
            continue
        value = tensors[name]
        module_stem = name.removesuffix(".weight")
        scales = tensors.get(f"{module_stem}.scales")
        biases = tensors.get(f"{module_stem}.biases")
        if scales is None or biases is None:
            out[name] = value
            continue
        consumed.update((f"{module_stem}.scales", f"{module_stem}.biases"))
        bits, group = per_tensor_params.get(
            module_stem, (int(global_bits), int(global_group))
        )
        out[name] = mx.dequantize(value, scales, biases, group, bits)
    return out


def _canon(name: str) -> str:
    return name.removeprefix(_LANGUAGE_PREFIX)


def attribute(quant_dir: Path, bf16_dir: Path) -> dict:
    profile = extract_profile(quant_dir)
    quant_tensors = _dequantized_tensors(quant_dir)
    bf16_tensors = load_model_tensors(bf16_dir)

    by_bf16 = {_canon(n): (n, t) for n, t in bf16_tensors.items()}

    class_sq: dict[str, float] = defaultdict(float)
    class_ref: dict[str, float] = defaultdict(float)
    class_count: dict[str, int] = defaultdict(int)
    per_tensor_stats: dict[str, dict] = {}
    unmatched: list[str] = []

    for name, q in quant_tensors.items():
        canon = _canon(name)
        hit = by_bf16.get(canon)
        if hit is None:
            unmatched.append(name)
            continue
        ref_name, ref = hit
        if q.shape != ref.shape:
            per_tensor_stats[ref_name] = {"status": "shape_mismatch"}
            continue
        cls = classify(canon)
        qf = q.astype(mx.float32)
        rf = ref.astype(mx.float32)
        diff = qf - rf
        sq = float(mx.sum(diff * diff))
        ref_sq = float(mx.sum(rf * rf))
        class_sq[cls] += sq
        class_ref[cls] += ref_sq
        class_count[cls] += 1
        rel = sq / ref_sq if ref_sq > 0 else float("nan")
        per_tensor_stats[ref_name] = {
            "class": cls,
            "rel_l2_sq": rel,
            "rel_l2": rel**0.5 if ref_sq > 0 else float("nan"),
            "status": "ok",
        }

    summary = {}
    for cls in sorted(class_sq):
        rel_sq = class_sq[cls] / class_ref[cls] if class_ref[cls] else float("nan")
        summary[cls] = {
            "rel_l2": rel_sq**0.5,
            "n_tensors": class_count[cls],
        }
    summary = dict(
        sorted(summary.items(), key=lambda kv: kv[1]["rel_l2"], reverse=True)
    )
    return {
        "quantized_dir": str(quant_dir),
        "bf16_dir": str(bf16_dir),
        "profile_global": {
            "bits": profile["global_bits"],
            "group_size": profile["global_group_size"],
        },
        "class_error_summary": summary,
        "per_tensor": per_tensor_stats,
        "unmatched": unmatched,
    }


def write_md(report: dict, path: Path) -> None:
    lines = [
        "# Per-module quantization error attribution",
        "",
        f"- quantized: `{report['quantized_dir']}`",
        f"- bf16 reference: `{report['bf16_dir']}`",
        "",
        "| module class | rel L2 error | n tensors |",
        "|---|---|---|",
    ]
    for cls, info in report["class_error_summary"].items():
        lines.append(f"| {cls} | {info['rel_l2']:.4e} | {info['n_tensors']} |")
    if report["unmatched"]:
        lines += ["", f"Unmatched tensors: {len(report['unmatched'])}"]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantized", required=True)
    parser.add_argument("--bf16", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--md")
    args = parser.parse_args(argv)

    report = attribute(Path(args.quantized).expanduser(), Path(args.bf16).expanduser())
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.md:
        write_md(report, Path(args.md))
    print("worst classes by rel L2 error:")
    for cls, info in list(report["class_error_summary"].items())[:12]:
        print(f"  {cls:45s} {info['rel_l2']:.4e}  (n={info['n_tensors']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
