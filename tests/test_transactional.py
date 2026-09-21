"""Transactional output and explicit-config validation tests.

Guarantees under test:
- a failed conversion never leaves a partial (or replaced) model directory
  at the output path, and no staging directories survive;
- an existing non-empty output directory is never replaced without an
  explicit overwrite;
- a config that cannot produce a required config.json field fails loudly
  instead of silently relying on mlx-lm defaults.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest import write_gguf, write_minimal_tokenizer  # noqa: E402

from gguf2mlx_stream.config.schema import load_arch_config  # noqa: E402
from gguf2mlx_stream.errors import ConversionError, PlanError  # noqa: E402
from gguf2mlx_stream.planner import plan_conversion  # noqa: E402
from gguf2mlx_stream.runner import ConversionRunner, QuantSettings  # noqa: E402
from gguf2mlx_stream.source.gguf import GGUFSource  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QWEN35_YAML = ROOT / "configs" / "qwen3_5.yaml"


def _tiny_qwen35_gguf(tmp_path: Path) -> Path:
    hidden, inter, vocab = 16, 24, 32
    rng = np.random.default_rng(5)

    def mat(r, c):
        return (rng.standard_normal((r, c)) * 0.05).astype(np.float32)

    def vec(n):
        return (rng.standard_normal((n,)) * 0.05).astype(np.float32)
    tensors = [("token_embd.weight", mat(vocab, hidden)),
               ("output_norm.weight", np.abs(vec(hidden)) + 1.0),
               ("output.weight", mat(vocab, hidden))]
    # layer 0: GDN layer; layer 1: full attention (interval 2)
    tensors += [
        ("blk.0.attn_norm.weight", np.abs(vec(hidden)) + 1.0),
        ("blk.0.post_attention_norm.weight", np.abs(vec(hidden)) + 1.0),
        ("blk.0.ffn_gate.weight", mat(inter, hidden)),
        ("blk.0.ffn_up.weight", mat(inter, hidden)),
        ("blk.0.ffn_down.weight", mat(hidden, inter)),
        ("blk.0.attn_qkv.weight", mat(64, hidden)),
        ("blk.0.attn_gate.weight", mat(32, hidden)),
        ("blk.0.ssm_alpha.weight", mat(8, hidden)),
        ("blk.0.ssm_beta.weight", mat(8, hidden)),
        ("blk.0.ssm_out.weight", mat(hidden, 32)),
        ("blk.0.ssm_conv1d.weight", mat(64, 2)),
        ("blk.0.ssm_dt.bias", vec(8)),
        ("blk.0.ssm_a", -np.abs(vec(8)) - 0.1),
        ("blk.0.ssm_norm.weight", np.abs(vec(4)) + 1.0),
        ("blk.1.attn_norm.weight", np.abs(vec(hidden)) + 1.0),
        ("blk.1.post_attention_norm.weight", np.abs(vec(hidden)) + 1.0),
        ("blk.1.ffn_gate.weight", mat(inter, hidden)),
        ("blk.1.ffn_up.weight", mat(inter, hidden)),
        ("blk.1.ffn_down.weight", mat(hidden, inter)),
        ("blk.1.attn_q.weight", mat(2 * 4 * 8, hidden)),
        ("blk.1.attn_k.weight", mat(2 * 8, hidden)),
        ("blk.1.attn_v.weight", mat(2 * 8, hidden)),
        ("blk.1.attn_output.weight", mat(hidden, 4 * 8)),
        ("blk.1.attn_q_norm.weight", np.abs(vec(8)) + 1.0),
        ("blk.1.attn_k_norm.weight", np.abs(vec(8)) + 1.0),
    ]
    return Path(write_gguf(
        tmp_path / "tiny.gguf", arch="qwen35",
        metadata={
            "embedding_length": hidden, "block_count": 2,
            "nextn_predict_layers": 0,
            "attention.head_count": 4, "attention.head_count_kv": 2,
            "attention.key_length": 8,
            "attention.layer_norm_rms_epsilon": 1e-5,
            "feed_forward_length": inter, "context_length": 512,
            "full_attention_interval": 2,
            "rope.freq_base": 1e6, "rope.dimension_count": 2,
            "qwen35.ssm.group_count": 4, "qwen35.ssm.state_size": 4,
            "qwen35.ssm.time_step_rank": 8, "qwen35.ssm.inner_size": 32,
            "qwen35.ssm.conv_kernel": 2,
        },
        f32_tensors=tensors,
    ))


def _runner(gguf: Path, out: Path, config_path: Path | None = None,
            tokenizer_source: Path | None = None) -> ConversionRunner:
    cfg = load_arch_config(str(config_path or QWEN35_YAML))
    source = GGUFSource(str(gguf))
    plan = plan_conversion(cfg, source)
    return ConversionRunner(
        plan, source, str(out),
        quant=QuantSettings(bits=None),
        tokenizer_source=str(tokenizer_source) if tokenizer_source else write_minimal_tokenizer(out.parent / "tokenizer"),
        log=lambda _: None,
    )


def test_failed_conversion_leaves_no_partial_output(tmp_path, monkeypatch):
    gguf = _tiny_qwen35_gguf(tmp_path)
    out = tmp_path / "model-out"

    real_convert = ConversionRunner._convert_job
    calls = {"n": 0}

    def failing(self, job, writer):
        calls["n"] += 1
        if calls["n"] == 2:  # fail on the second tensor
            raise ConversionError("injected failure")
        return real_convert(self, job, writer)

    monkeypatch.setattr(ConversionRunner, "_convert_job", failing)
    with pytest.raises(ConversionError, match="injected failure"):
        _runner(gguf, out).run()

    assert not out.exists(), "output path must not appear on failure"
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name or ".old-" in p.name]
    assert not leftovers, f"staging leftovers: {leftovers}"
    assert calls["n"] >= 2


def test_existing_output_needs_overwrite(tmp_path):
    gguf = _tiny_qwen35_gguf(tmp_path)
    out = tmp_path / "model-out"
    _runner(gguf, out).run()
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["weight_map"]

    sentinel = out / "sentinel.txt"
    sentinel.write_text("previous output")

    with pytest.raises(ConversionError, match="overwrite"):
        _runner(gguf, out).run()
    assert sentinel.exists(), "previous output must be untouched without --overwrite"

    _runner(gguf, out).run(overwrite=True)
    assert not sentinel.exists(), "overwrite must replace the previous output"
    assert (out / "config.json").is_file()
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name or ".old-" in p.name]
    assert not leftovers


def test_empty_output_dir_is_writable_without_overwrite(tmp_path):
    gguf = _tiny_qwen35_gguf(tmp_path)
    out = tmp_path / "model-out"
    out.mkdir()
    _runner(gguf, out).run()
    assert (out / "config.json").is_file()


def test_missing_required_config_field_fails(tmp_path):
    gguf = _tiny_qwen35_gguf(tmp_path)
    out = tmp_path / "model-out"

    cfg = load_arch_config(str(QWEN35_YAML))
    # break one required field: point it at a metadata key that does not exist
    broken = cfg.output.text_config | {"full_attention_interval": "gguf:not.a.real.key"}
    from gguf2mlx_stream.config.schema import OutputSpec
    cfg = type(cfg)(
        path=cfg.path, architecture=cfg.architecture, dims=cfg.dims, rules=cfg.rules,
        coverage=cfg.coverage,
        output=OutputSpec(**{**cfg.output.__dict__, "text_config": broken}),
        unmatched_policy=cfg.unmatched_policy,
    )
    source = GGUFSource(str(gguf))
    plan = plan_conversion(cfg, source)
    runner = ConversionRunner(plan, source, str(out),
                              quant=QuantSettings(bits=None),
                              tokenizer_source=write_minimal_tokenizer(tmp_path / "tokenizer"),
                              log=lambda _: None)
    with pytest.raises((ConversionError, PlanError)):
        runner.run()
    assert not out.exists()


# ---------------------------------------------------------------------------
# tokenizer output contract (P1): no tokenizer, no successful conversion
# ---------------------------------------------------------------------------

def test_missing_tokenizer_files_fail_and_preserve_previous_output(tmp_path):
    gguf = _tiny_qwen35_gguf(tmp_path)
    out = tmp_path / "model-out"
    _runner(gguf, out).run()
    sentinel = out / "sentinel.txt"
    sentinel.write_text("previous output")

    empty = tmp_path / "no-tokenizer"
    empty.mkdir()
    with pytest.raises(ConversionError, match="tokenizer"):
        _runner(gguf, out, tokenizer_source=empty).run(overwrite=True)

    assert sentinel.exists(), "previous output must be preserved on tokenizer failure"
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name or ".old-" in p.name]
    assert not leftovers, f"staging leftovers: {leftovers}"


def test_tokenizer_config_json_is_required(tmp_path):
    gguf = _tiny_qwen35_gguf(tmp_path)
    out = tmp_path / "model-out"

    partial = tmp_path / "vocab-only"
    write_minimal_tokenizer(partial)
    (partial / "tokenizer_config.json").unlink()

    with pytest.raises(ConversionError, match="tokenizer_config.json"):
        _runner(gguf, out, tokenizer_source=partial).run()

    assert not out.exists()
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name or ".old-" in p.name]
    assert not leftovers


def test_invalid_tokenizer_json_fails(tmp_path):
    gguf = _tiny_qwen35_gguf(tmp_path)
    out = tmp_path / "model-out"

    broken = tmp_path / "broken-tokenizer"
    write_minimal_tokenizer(broken)
    (broken / "tokenizer.json").write_text("{not json")

    with pytest.raises(ConversionError, match="tokenizer.json"):
        _runner(gguf, out, tokenizer_source=broken).run()

    assert not out.exists()


def test_valid_tokenizer_produces_loadable_contract_files(tmp_path):
    gguf = _tiny_qwen35_gguf(tmp_path)
    out = tmp_path / "model-out"
    _runner(gguf, out).run()
    from gguf2mlx_stream.writer import check_tokenizer_output
    check_tokenizer_output(str(out))  # must not raise
    assert (out / "tokenizer.json").is_file()
    assert (out / "tokenizer_config.json").is_file()
