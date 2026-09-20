"""Command line interface.

    gguf2mlx-stream inspect        model.gguf
    gguf2mlx-stream list-ops
    gguf2mlx-stream validate-config configs/qwen3_5.yaml
    gguf2mlx-stream convert        model.gguf --arch-config configs/qwen3_5.yaml --output ./out [options]
    gguf2mlx-stream verify         model.gguf ./out --arch-config configs/qwen3_5.yaml [options]

All commands are non-interactive and scriptable. ``convert --dry-run``
prints the compiled conversion plan without touching any tensor data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config.schema import load_arch_config
from .errors import Gguf2MlxError
from .ops import all_ops
from .planner import plan_conversion
from .quantize import SUPPORTED_BITS
from .runner import ConversionRunner, QuantSettings
from .source.gguf import GGUFSource
from .verifier import load_test, verify_conversion


def _add_common_convert_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("gguf", help="source GGUF file")
    p.add_argument("--arch-config", required=True, help="architecture YAML config")
    p.add_argument(
        "--source-config",
        default=None,
        help="optional reference config.json (HF/text config) used for dims and output config",
    )
    p.add_argument(
        "--tokenizer-source",
        default=None,
        help="directory to copy tokenizer files from (default: GGUF's directory)",
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    source = GGUFSource(args.gguf)
    if args.json:
        print(json.dumps(source.summary(), indent=2))
        return 0
    s = source.summary()
    print(f"file        : {s['path']}")
    print(f"arch        : {s['architecture']}")
    print(f"tensors     : {s['n_tensors']} ({s['total_tensor_bytes'] / 2**30:.2f} GiB)")
    print("quant types : " + ", ".join(f"{k}×{v}" for k, v in s["tensors_by_type"].items()))
    if args.metadata:
        print(source.dump_metadata_json())
    if args.tensors:
        print(f"\n{'tensor':<48} {'shape (out,in)':<24} {'type':<8} size")
        for t in source.iter_infos():
            shape = "x".join(str(d) for d in t.hf_shape)
            print(f"{t.name:<48} {shape:<24} {t.qtype.name:<8} {t.n_bytes / 2**20:.2f} MiB")
    return 0


def cmd_list_ops(args: argparse.Namespace) -> int:
    for name, spec in sorted(all_ops().items()):
        stream = "chunk-safe" if spec.streaming else "whole-tensor"
        print(f"{name:<28} [{spec.kind}, {stream}] {spec.summary}")
        if spec.inputs_doc != "x":
            print(f"{'':<28} inputs: {spec.inputs_doc}")
        for pname, pdesc in spec.params:
            print(f"{'':<28} - {pname}: {pdesc}")
    return 0


def cmd_validate_config(args: argparse.Namespace) -> int:
    config = load_arch_config(args.config)
    n_rules = len(config.rules)
    n_drop = sum(1 for r in config.rules if r.drop)
    print(f"OK {args.config}")
    print(f"  architecture : {config.architecture.id} (aliases: {config.architecture.aliases})")
    print(f"  rules        : {n_rules} ({n_drop} drop rules)")
    print(f"  dims declared: {len(config.dims)}")
    if config.output.model_type:
        print(f"  output       : model_type={config.output.model_type} "
              f"nest={config.output.nest_config_under!r}")
    return 0


def _load_ref_config(path: str | None) -> dict | None:
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def cmd_convert(args: argparse.Namespace) -> int:
    config = load_arch_config(args.arch_config)
    source = GGUFSource(args.gguf)
    ref = _load_ref_config(args.source_config)
    tokenizer_source = args.tokenizer_source or os.path.dirname(os.path.abspath(args.gguf))

    plan = plan_conversion(config, source, ref_config=ref)
    if args.dry_run:
        print("\n".join(plan.summary_lines()))
        return 0

    quant = QuantSettings(
        bits=None if args.no_quantize else args.bits,
        group_size=args.group_size,
        mode=args.mode,
    )
    runner = ConversionRunner(
        plan,
        source,
        args.output,
        quant=quant,
        ref_config=ref,
        tokenizer_source=tokenizer_source,
        chunk_elements=args.chunk_mb * 2**20 // 4,
        log=print if not args.quiet else (lambda _msg: None),
        max_shard_bytes=int(args.max_shard_gb * 2**30) if args.max_shard_gb else None,
    )
    stats = runner.run()
    if not args.quiet:
        print(
            f"[stats] output={stats.output_bytes / 2**30:.2f} GiB, shards={stats.n_shards}, "
            f"time={stats.elapsed_s:.0f}s, peak RSS={stats.peak_rss_gib:.2f} GiB"
        )
        if args.report_json:
            with open(args.report_json, "w") as f:
                json.dump(
                    {
                        "output_bytes": stats.output_bytes,
                        "n_tensors": stats.n_tensors,
                        "n_shards": stats.n_shards,
                        "elapsed_s": stats.elapsed_s,
                        "peak_rss_gib": stats.peak_rss_gib,
                        "estimated_output_bytes": plan.est_output_bytes,
                        "n_jobs": len(plan.jobs),
                        "n_dropped": len(plan.dropped),
                        "dims": dict(plan.dims),
                    },
                    f,
                    indent=2,
                )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    config = load_arch_config(args.arch_config)
    source = GGUFSource(args.gguf)
    ref = _load_ref_config(args.source_config)
    plan = plan_conversion(config, source, ref_config=ref)

    report = verify_conversion(
        plan,
        source,
        args.output_dir,
        bits=args.bits,
        group_size=args.group_size,
        tolerance=args.tolerance,
        verify_all_quantized=args.all,
    )
    for line in report.details[: args.max_details]:
        print(line)
    if len(report.details) > args.max_details:
        print(f"... {len(report.details) - args.max_details} more OK checks")
    if report.failures:
        for f in report.failures:
            print(f"FAIL {f}", file=sys.stderr)
        print(
            f"verify: {len(report.failures)} failure(s); "
            f"{report.checked_numeric} numeric, {report.checked_shapes} shape, "
            f"{report.checked_finite} finite checks ran",
            file=sys.stderr,
        )
        return 1
    print(
        f"verify: ALL OK ({report.checked_numeric} numeric, {report.checked_shapes} shape, "
        f"{report.checked_finite} finite checks)"
    )
    if args.load_test:
        text = load_test(args.output_dir, prompt=args.prompt, max_tokens=args.max_tokens)
        print(f"load-test generation: {text[:200]!r}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gguf2mlx-stream",
        description="Declarative bounded-memory GGUF -> MLX-LM streaming transcoder",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="show GGUF metadata and tensor inventory")
    p.add_argument("gguf")
    p.add_argument("--tensors", action="store_true", help="list every tensor")
    p.add_argument("--metadata", action="store_true", help="dump all metadata")
    p.add_argument("--json", action="store_true", help="machine-readable summary")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("list-ops", help="list registered transformation operators")
    p.set_defaults(fn=cmd_list_ops)

    p = sub.add_parser("validate-config", help="validate an architecture config")
    p.add_argument("config")
    p.set_defaults(fn=cmd_validate_config)

    p = sub.add_parser("convert", help="convert GGUF to an MLX-LM checkpoint")
    _add_common_convert_args(p)
    p.add_argument("--output", "-o", required=True, help="output model directory")
    p.add_argument("--bits", type=int, default=4, choices=SUPPORTED_BITS)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--mode", default="affine", choices=("affine",))
    p.add_argument("--no-quantize", action="store_true", help="write float16 weights")
    p.add_argument("--dry-run", action="store_true", help="print plan and exit")
    p.add_argument("--max-shard-gb", type=float, default=None,
                   help="override shard size limit from the config")
    p.add_argument("--chunk-mb", type=int, default=512,
                   help="dequantization chunk size in MiB (default 512)")
    p.add_argument("--quiet", "-q", action="store_true")
    p.add_argument("--report-json", default=None, help="write run statistics JSON")
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("verify", help="verify an MLX-LM output against its source GGUF")
    _add_common_convert_args(p)
    p.add_argument("output_dir")
    p.add_argument("--bits", type=int, default=None, choices=SUPPORTED_BITS,
                   help="quantization bits (default: read from output config.json)")
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--tolerance", type=float, default=None)
    p.add_argument("--all", action="store_true", help="numerically check every quantized tensor")
    p.add_argument("--load-test", action="store_true", help="run mlx_lm.load() + short generation")
    p.add_argument("--prompt", default="What is 2+2? Answer with just the number.")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--max-details", type=int, default=40)
    p.set_defaults(fn=cmd_verify)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except Gguf2MlxError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
