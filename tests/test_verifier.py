"""Verifier credibility tests.

Guarantees under test (release P1):

* by default, EVERY quantized output tensor is numerically recomputed from
  the source GGUF — a tampered later tensor of an already-seen rule is
  caught (the old first-per-rule sampling would have missed it; sampling is
  now an explicit ``--sampled`` opt-in);
* the index and every shard's actual key set agree bidirectionally: no
  unindexed tensor inside a shard, no unindexed shard file, no stale index
  entry;
* quantization parameters (bits, group_size, mode) are read from the
  output's config.json and validated; explicit CLI values that conflict
  with the output metadata fail.
"""

import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
from conftest import write_minimal_tokenizer  # noqa: E402

from test_pipeline_synthetic import build_fixture_gguf  # noqa: E402

from gguf2mlx_stream.cli import main as cli_main  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QWEN35_YAML = str(ROOT / "configs" / "qwen3_5.yaml")


def _convert(tmp_path: Path) -> tuple[Path, Path]:
    """Convert the synthetic qwen35 fixture (2 layers, Q6/Q4 globals)."""
    gguf_path, _, _, _ = build_fixture_gguf(tmp_path)
    out_dir = tmp_path / "out"
    rc = cli_main([
        "convert", str(gguf_path), "--arch-config", QWEN35_YAML,
        "--output", str(out_dir),
        "--tokenizer-source", write_minimal_tokenizer(tmp_path / "tokenizer"),
        "--bits", "4", "--quiet",
    ])
    assert rc == 0, "conversion failed"
    return gguf_path, out_dir


def _rewrite_shard(out_dir: Path, fname: str, mutate) -> None:
    path = out_dir / fname
    tensors = load_file(str(path))
    mutate(tensors)
    save_file(tensors, str(path))


def _find_shard(out_dir: Path, key: str) -> str:
    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    return index["weight_map"][key]


def test_verify_passes_untouched_output(tmp_path):
    gguf_path, out_dir = _convert(tmp_path)
    # default: no explicit quant args, full numeric coverage
    assert cli_main(["verify", str(gguf_path), str(out_dir),
                     "--arch-config", QWEN35_YAML]) == 0


def test_tampered_later_layer_tensor_is_caught(tmp_path):
    """A later large tensor of a rule whose first instance was already
    verified must still be checked: full coverage is the default."""
    gguf_path, out_dir = _convert(tmp_path)
    # layer 1 ffn_gate: same rule as layer 0's ffn_gate, packed uint32
    victim = "language_model.model.layers.1.mlp.gate_proj.weight"
    shard = _find_shard(out_dir, victim)

    def tamper(tensors):
        arr = tensors[victim].copy()
        arr[0, 0] = arr[0, 0] ^ np.uint32(0xFFFF)  # corrupt the packed codes
        tensors[victim] = arr

    _rewrite_shard(out_dir, shard, tamper)

    rc = cli_main(["verify", str(gguf_path), str(out_dir),
                   "--arch-config", QWEN35_YAML, "--max-details", "0"])
    assert rc == 1, "tampered tensor must fail verify"
    # the report names the victim, not just any failure
    # (rc==1 path prints FAIL lines to stderr; re-run via the API for text)
    from gguf2mlx_stream.planner import plan_conversion
    from gguf2mlx_stream.source.gguf import GGUFSource
    from gguf2mlx_stream.builtin import resolve_arch_config
    from gguf2mlx_stream.verifier import verify_conversion
    cfg, _ = resolve_arch_config(QWEN35_YAML)
    source = GGUFSource(str(gguf_path))
    report = verify_conversion(plan_conversion(cfg, source), source, str(out_dir))
    assert not report.ok
    assert any(victim in f for f in report.failures)


def test_extra_key_inside_shard_is_rejected(tmp_path):
    gguf_path, out_dir = _convert(tmp_path)
    shard = _find_shard(out_dir, "language_model.model.norm.weight")

    def inject(tensors):
        tensors["sneaky.extra.key"] = np.zeros((2, 2), np.float32)

    _rewrite_shard(out_dir, shard, inject)
    rc = cli_main(["verify", str(gguf_path), str(out_dir),
                   "--arch-config", QWEN35_YAML, "--max-details", "0"])
    assert rc == 1
    from gguf2mlx_stream.planner import plan_conversion
    from gguf2mlx_stream.source.gguf import GGUFSource
    from gguf2mlx_stream.builtin import resolve_arch_config
    from gguf2mlx_stream.verifier import verify_conversion
    cfg, _ = resolve_arch_config(QWEN35_YAML)
    source = GGUFSource(str(gguf_path))
    report = verify_conversion(plan_conversion(cfg, source), source, str(out_dir))
    assert any("sneaky.extra.key" in f and "unindexed" in f for f in report.failures)


def test_unindexed_shard_file_is_rejected(tmp_path):
    gguf_path, out_dir = _convert(tmp_path)
    src_shard = out_dir / _find_shard(out_dir, "language_model.model.norm.weight")
    rogue = out_dir / "model-99999-of-99999.safetensors"
    save_file({"ghost.weight": np.zeros((2, 2), np.float32)}, str(rogue))
    _ = src_shard  # referenced only to keep the fixture explicit

    from gguf2mlx_stream.planner import plan_conversion
    from gguf2mlx_stream.source.gguf import GGUFSource
    from gguf2mlx_stream.builtin import resolve_arch_config
    from gguf2mlx_stream.verifier import verify_conversion
    cfg, _ = resolve_arch_config(QWEN35_YAML)
    source = GGUFSource(str(gguf_path))
    report = verify_conversion(plan_conversion(cfg, source), source, str(out_dir))
    assert any("not referenced by the index" in f for f in report.failures)


def test_verify_fails_on_bits_conflict_with_config_json(tmp_path):
    gguf_path, out_dir = _convert(tmp_path)
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    assert cfg["quantization"]["bits"] == 4
    cfg["quantization"]["bits"] = 6
    cfg_path.write_text(json.dumps(cfg))
    # explicit --bits disagrees with the (edited) output metadata
    assert cli_main(["verify", str(gguf_path), str(out_dir),
                     "--arch-config", QWEN35_YAML, "--bits", "4"]) == 2
    # no explicit value: metadata says 6, weights are 4-bit -> numeric garbage
    assert cli_main(["verify", str(gguf_path), str(out_dir),
                     "--arch-config", QWEN35_YAML]) == 1


def test_verify_fails_on_group_size_conflict(tmp_path):
    gguf_path, out_dir = _convert(tmp_path)
    assert cli_main(["verify", str(gguf_path), str(out_dir),
                     "--arch-config", QWEN35_YAML, "--group-size", "32"]) == 2


def test_verify_fails_on_mode_conflict(tmp_path):
    gguf_path, out_dir = _convert(tmp_path)
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["quantization"]["mode"] = "mxfp4"
    cfg_path.write_text(json.dumps(cfg))
    assert cli_main(["verify", str(gguf_path), str(out_dir),
                     "--arch-config", QWEN35_YAML, "--mode", "affine"]) == 2


def test_sampled_opt_in_still_passes_untouched_output(tmp_path):
    gguf_path, out_dir = _convert(tmp_path)
    assert cli_main(["verify", str(gguf_path), str(out_dir),
                     "--arch-config", QWEN35_YAML, "--sampled"]) == 0
