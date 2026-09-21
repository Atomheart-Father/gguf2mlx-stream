"""Extract the quantization profile of an MLX safetensors model directory.

mlx-lm records quantization parameters in ``config.json``: a global
``bits``/``group_size``/``mode`` plus *inline* per-tensor entries keyed by
module path (``{"...layers.3.mlp.down_proj": {"bits": 4, ...}}``). This tool
aggregates that record into a human-readable profile and, for mixed models,
reverse-engineers the classification rule by correlating bit width with
module class and tensor size.

Usage::

    python -m research.paired_oracle.profile_extract \
        --model <dir> [--model <dir> ...] --report <out.json> [--md <out.md>]
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _safetensors_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for shard in sorted(path.glob("*.safetensors")):
        with shard.open("rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        for key, meta in header.items():
            if isinstance(meta, dict) and "shape" in meta:
                shapes[key] = tuple(meta["shape"])
    return shapes


_LAYER_RE = re.compile(r"^(?:language_model\.)?model\.layers\.(\d+)\.(.+)$")


def classify(name: str) -> str:
    """Coarse module class used for profile aggregation.

    Accepts names with or without the ``language_model.`` prefix.
    """
    m = _LAYER_RE.match(name)
    if m:
        rest = m.group(2)
        rest = re.sub(r"\.weight$", "", rest)
        return f"layer.{rest}"
    name = re.sub(r"\.weight$", "", name)
    return name


def extract(model_dir: Path) -> dict:
    cfg = json.loads((model_dir / "config.json").read_text())
    quant = cfg.get("quantization", {})
    global_bits = quant.get("bits")
    global_group = quant.get("group_size")
    mode = quant.get("mode")
    shapes = _safetensors_shapes(model_dir)

    per_tensor: dict[str, dict] = {}
    classes: dict[str, Counter] = defaultdict(Counter)
    class_bits: dict[str, set] = defaultdict(set)
    for key, value in quant.items():
        if key in ("bits", "group_size", "mode"):
            continue
        if not isinstance(value, dict) or "bits" not in value:
            continue
        bits = int(value["bits"])
        group = int(value.get("group_size", global_group or 0))
        cls = classify(key)
        shape = shapes.get(f"{key}.weight", ())
        numel = 1
        for d in shape:
            numel *= d
        per_tensor[key] = {
            "bits": bits,
            "group_size": group,
            "class": cls,
            "shape": list(shape),
            "numel": numel,
        }
        classes[cls][bits] += 1
        class_bits[cls].add(bits)

    # tensors present but NOT quantized (norms, A_log, dt_bias, ...)
    quantized_prefixes = set(per_tensor)
    unquantized = sorted(
        k for k in shapes
        if k not in quantized_prefixes
        and not k.endswith((".scales", ".biases"))
    )

    return {
        "model_dir": str(model_dir),
        "global_bits": global_bits,
        "global_group_size": global_group,
        "mode": mode,
        "n_quantized": len(per_tensor),
        "n_unquantized": len(unquantized),
        "unquantized_examples": unquantized[:8],
        "class_summary": {
            cls: {
                "bits_histogram": dict(counter),
                "distinct_bits": sorted(class_bits[cls]),
            }
            for cls, counter in sorted(classes.items())
        },
        "per_tensor": per_tensor,
    }


def summarize_md(models: list[dict]) -> str:
    lines = ["# Official MLX quantization profiles", ""]
    for rec in models:
        lines += [
            f"## `{Path(rec['model_dir']).name}`",
            "",
            f"- global: bits={rec['global_bits']} group_size={rec['global_group_size']} "
            f"mode={rec['mode']}",
            f"- quantized tensors: {rec['n_quantized']} | unquantized: {rec['n_unquantized']} "
            f"(e.g. {rec['unquantized_examples'][:4]})",
            "",
            "| module class | bits histogram | distinct bits |",
            "|---|---|---|",
        ]
        for cls, info in rec["class_summary"].items():
            hist = ", ".join(f"{b}b x{n}" for b, n in sorted(info["bits_histogram"].items()))
            lines.append(f"| {cls} | {hist} | {info['distinct_bits']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--md")
    args = parser.parse_args(argv)

    models = [extract(Path(m).expanduser()) for m in args.model]
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(models, indent=2, sort_keys=True) + "\n")
    if args.md:
        Path(args.md).write_text(summarize_md(models))
    for rec in models:
        print(f"{Path(rec['model_dir']).name}: global bits={rec['global_bits']} "
              f"quantized={rec['n_quantized']} unquantized={rec['n_unquantized']} "
              f"distinct classes={len(rec['class_summary'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
