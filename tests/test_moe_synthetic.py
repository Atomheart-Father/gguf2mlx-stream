"""Synthetic MoE pipeline test: 3-D quantization -> shard -> reload -> verify.

Exercises the general N-D quantization semantics end to end on a tiny
fixture (no multi-GB checkpoint required):

* a rank-3 expert tensor is quantized along its last axis with every
  leading axis preserved (chunked streaming over the flattened outer
  GGUF rows, never a whole-tensor FP32 materialization);
* per-rule bits/group_size overrides travel from the architecture config
  through the runner into config.json (mlx-lm's per-key override format)
  and are re-checked by the verifier;
* a GGUF metadata *array* (`rope.dimension_sections`) is read safely and
  passed through declaratively into the output config;
* chunk boundaries are exercised with a tiny chunk_elements setting, and
  the chunked path is proven bounded by spying on the source reader.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gguf2mlx_stream.config.schema import arch_config_from_dict
from gguf2mlx_stream.planner import plan_conversion
from gguf2mlx_stream.quantize import dequantize_weights
from gguf2mlx_stream.runner import ConversionRunner, QuantSettings
from gguf2mlx_stream.source.gguf import GGUFSource
from gguf2mlx_stream.verifier import verify_conversion

from conftest import write_gguf, write_minimal_tokenizer

N_EXPERTS, EXPERT_TOKENS, HIDDEN = 2, 4, 64  # expert tensor (2, 4, 64)
VOCAB, INTER = 8, 32


def _moe_config_raw() -> dict:
    """Minimal qwen35moe-shaped config exercising all MoE semantics."""
    return {
        "architecture": {"id": "synthetic_moe", "gguf_arch": "synmoe"},
        "dims": {
            "hidden_size": "gguf:embedding_length",
            "n_layers": "gguf:block_count",
            "total_blocks": "gguf:block_count",
            "nextn_predict_layers": 0,
            "vocab_size": "gguf:vocab_size",
            "n_experts": "gguf:expert_count",
            "inter": "gguf:feed_forward_length",
        },
        "rules": [
            {
                "name": "embeddings",
                "match": "token_embd\\.weight",
                "dest": "embed.weight",
            },
            {
                "name": "router",
                "match": "blk\\.(?P<n>\\d+)\\.ffn_gate_inp\\.weight",
                "dest": "layers.{n}.mlp.gate.weight",
                "expect_shape": ["n_experts", "hidden_size"],
                "bits": 8,
                "group_size": 64,
            },
            {
                "name": "experts",
                "match": "blk\\.(?P<n>\\d+)\\.ffn_gate_exps\\.weight",
                "dest": "layers.{n}.mlp.switch_mlp.gate_proj.weight",
                "expect_shape": ["n_experts", "inter", "hidden_size"],
            },
            {
                "name": "shared_gate",
                "match": "blk\\.(?P<n>\\d+)\\.ffn_gate_inp_shexp",
                "dest": "layers.{n}.mlp.shared_expert_gate.weight",
                "expect_shape": ["hidden_size"],
                "steps": [{"op": "unsqueeze", "args": {"axis": 0}}],
            },
        ],
        "coverage": {},
        "unmatched_tensors": "error",
        "output": {
            "model_type": "synthetic_moe",
            "architectures": ["SyntheticMoeForCausalLM"],
            "nest_config_under": "text_config",
            "text_config": {
                "hidden_size": ["hidden_size"],
                "num_experts": ["n_experts"],
                "rope_parameters": {
                    "rope_type": "default",
                    "mrope_section": ["gguf:rope.dimension_sections"],
                },
            },
            "required_fields": [
                "text_config.hidden_size",
                "text_config.rope_parameters.mrope_section",
            ],
            "tokenizer_files": ["tokenizer.json", "tokenizer_config.json"],
        },
    }


@pytest.fixture()
def moe_gguf(tmp_path):
    """One-layer fixture GGUF: F32 router, 3-D F32 experts, 1-D shared gate."""
    rng = np.random.default_rng(23)
    expert = (rng.standard_normal((N_EXPERTS, EXPERT_TOKENS, HIDDEN)) * 0.05).astype(np.float32)
    router = (rng.standard_normal((N_EXPERTS, HIDDEN)) * 0.1).astype(np.float32)
    shared_gate = (rng.standard_normal((HIDDEN,)) * 0.1).astype(np.float32)
    metadata = {
        "general.architecture": "synmoe",
        "synmoe.block_count": 1,
        "synmoe.embedding_length": HIDDEN,
        "synmoe.vocab_size": VOCAB,
        "synmoe.expert_count": N_EXPERTS,
        "synmoe.feed_forward_length": EXPERT_TOKENS,
        # small metadata array: decoded, not a placeholder
        "synmoe.rope.dimension_sections": [5, 6, 7, 8],
    }
    path = write_gguf(
        str(tmp_path / "moe.gguf"),
        arch="synmoe",
        metadata=metadata,
        f32_tensors=[
            ("token_embd.weight", np.zeros((VOCAB, HIDDEN), np.float32)),
            ("blk.0.ffn_gate_exps.weight", expert),
            ("blk.0.ffn_gate_inp.weight", router),
            ("blk.0.ffn_gate_inp_shexp", shared_gate),
        ],
    )
    return path, expert, router, shared_gate


def _run_conversion(tmp_path, moe_gguf, chunk_elements):
    path, expert, router, shared_gate = moe_gguf
    tok = write_minimal_tokenizer(tmp_path / "tok")
    out = tmp_path / "out"
    cfg = arch_config_from_dict(_moe_config_raw(), path="synthetic")
    source = GGUFSource(path)
    plan = plan_conversion(cfg, source)

    # bounded-memory spy: the 3-D expert tensor must stream via row reads,
    # never a whole-tensor read_matrix materialization
    full_reads: list[str] = []
    row_reads: dict[str, int] = {}
    orig_matrix = GGUFSource.read_matrix
    orig_rows = GGUFSource.read_rows

    def spy_matrix(self, name):
        full_reads.append(name)
        return orig_matrix(self, name)

    def spy_rows(self, name, lo=0, hi=None):
        row_reads[name] = row_reads.get(name, 0) + 1
        return orig_rows(self, name, lo, hi)

    GGUFSource.read_matrix = spy_matrix
    GGUFSource.read_rows = spy_rows
    try:
        runner = ConversionRunner(
            plan,
            source,
            str(out),
            quant=QuantSettings(bits=4, group_size=64),
            tokenizer_source=tok,
            chunk_elements=chunk_elements,
        )
        stats = runner.run()
    finally:
        GGUFSource.read_matrix = orig_matrix
        GGUFSource.read_rows = orig_rows

    return out, stats, full_reads, row_reads, expert, router, shared_gate


def test_moe_synthetic_pipeline(tmp_path, moe_gguf):
    out, stats, full_reads, row_reads, expert, router, shared_gate = _run_conversion(
        tmp_path, moe_gguf, chunk_elements=1  # forces 8 single-row chunks
    )

    # ---- conversion plan/coverage sanity
    assert stats.n_tensors == 4
    assert "blk.0.ffn_gate_exps.weight" not in full_reads  # chunked, bounded
    assert row_reads.get("blk.0.ffn_gate_exps.weight", 0) == N_EXPERTS * EXPERT_TOKENS

    # ---- reload from disk and check every layout contract
    from safetensors.numpy import load_file

    index = json.loads((out / "model.safetensors.index.json").read_text())
    shards = {
        fname: load_file(str(out / fname))
        for fname in sorted(set(index["weight_map"].values()))
    }
    tensors = {k: v for s in shards.values() for k, v in s.items()}

    # 3-D packed expert tensor: leading (n_experts, tokens) preserved,
    # last dim packed for 4-bit (64 * 4 / 32 = 8 uint32), one fp16 group
    packed = tensors["layers.0.mlp.switch_mlp.gate_proj.weight"]
    scales = tensors["layers.0.mlp.switch_mlp.gate_proj.scales"]
    biases = tensors["layers.0.mlp.switch_mlp.gate_proj.biases"]
    assert packed.dtype == np.uint32 and packed.shape == (N_EXPERTS, EXPERT_TOKENS, 8)
    assert scales.dtype == np.float16 and scales.shape == (N_EXPERTS, EXPERT_TOKENS, 1)
    assert biases.shape == (N_EXPERTS, EXPERT_TOKENS, 1)

    # dequantized values match the source expert tensor (4-bit tolerance)
    got = np.asarray(dequantize_weights(packed, scales, biases, bits=4, group_size=64))
    assert np.abs(got - expert).max() < 0.05

    # per-rule 8-bit override on the router
    r_packed = tensors["layers.0.mlp.gate.weight"]
    r_scales = tensors["layers.0.mlp.gate.scales"]
    assert r_packed.shape == (N_EXPERTS, 16)  # 64 * 8 / 32
    assert r_scales.shape == (N_EXPERTS, 1)
    r_got = np.asarray(dequantize_weights(r_packed, r_scales,
                                          tensors["layers.0.mlp.gate.biases"],
                                          bits=8, group_size=64))
    assert np.abs(r_got - router).max() < 0.01

    # shared gate unsqueezed to a (1, hidden) Linear weight, then quantized
    # at the global 4-bit setting: packed last dim 64 * 4 / 32 = 8
    sg = tensors["layers.0.mlp.shared_expert_gate.weight"]
    assert sg.shape == (1, 8)
    assert tensors["layers.0.mlp.shared_expert_gate.scales"].shape == (1, 1)

    # ---- output config: per-module overrides (mlx-lm key format: module
    # path without the trailing ".weight") + metadata array passthrough
    cfg = json.loads((out / "config.json").read_text())
    q = cfg["quantization"]
    assert q["bits"] == 4 and q["group_size"] == 64
    assert q["layers.0.mlp.gate"] == {"bits": 8, "group_size": 64}
    assert "layers.0.mlp.gate.weight" not in q
    assert "layers.0.mlp.switch_mlp.gate_proj.weight" not in q  # global 4-bit
    assert cfg["text_config"]["rope_parameters"]["mrope_section"] == [5, 6, 7, 8]

    # ---- verifier passes on the freshly written output
    source = GGUFSource(moe_gguf[0])
    plan = plan_conversion(arch_config_from_dict(_moe_config_raw(), path="s"), source)
    report = verify_conversion(plan, source, str(out))
    assert report.ok, report.failures


def test_moe_verifier_rejects_missing_override(tmp_path, moe_gguf):
    """A declared per-rule bits override MUST appear in config.json."""
    out, *_ , _expert, _router, _shared = _run_conversion(
        tmp_path, moe_gguf, chunk_elements=1 << 30  # no chunking
    )
    cfg = json.loads((out / "config.json").read_text())
    del cfg["quantization"]["layers.0.mlp.gate"]
    (out / "config.json").write_text(json.dumps(cfg))

    source = GGUFSource(moe_gguf[0])
    plan = plan_conversion(arch_config_from_dict(_moe_config_raw(), path="s"), source)
    report = verify_conversion(plan, source, str(out))
    assert not report.ok
    assert any("no per-tensor override" in f for f in report.failures)


def test_moe_unchunked_matches_chunked(tmp_path, moe_gguf):
    """Chunk boundaries must not change the packed result (row-order kept)."""
    out_a, *_ = _run_conversion(tmp_path / "a", moe_gguf, chunk_elements=1)
    path_b = tmp_path / "b"
    path_b.mkdir()
    out_b, *_ = _run_conversion(path_b, moe_gguf, chunk_elements=1 << 30)
    from safetensors.numpy import load_file

    a = load_file(str(out_a / "model-00001-of-00001.safetensors"))
    b = load_file(str(out_b / "model-00001-of-00001.safetensors"))
    assert set(a) == set(b)
    for k in a:
        assert np.array_equal(a[k], b[k]), k
