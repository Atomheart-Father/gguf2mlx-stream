"""Synthetic end-to-end pipeline test.

Builds a tiny GGUF with the *real* qwen3_5 tensor-name inventory (hybrid
GDN/full attention, MTP block, Q6_K + Q4_K quantized globals), then runs the
complete pipeline through the public CLI:

    read -> plan -> transform -> quantize -> shard write -> reload

and verifies numerics against analytically derived expectations.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import (
    QK_K,
    build_q4_block,
    build_q6_block,
    q4_expected,
    q6_expected,
    write_gguf,
    write_minimal_tokenizer,
)
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType
from safetensors.numpy import load_file

from gguf2mlx_stream.cli import main as cli_main
from gguf2mlx_stream.quantize import dequantize_weights, quantize_weights

ROOT = Path(__file__).resolve().parent.parent
QWEN35_YAML = str(ROOT / "configs" / "qwen3_5.yaml")

# ---------------------------------------------------------------------------
# fixture geometry (must be consistent with configs/qwen3_5.yaml semantics)
# ---------------------------------------------------------------------------

HIDDEN = 256
VOCAB = 512
N_LAYERS = 2  # layer 0: linear attention, layer 1: full attention
MTP = N_LAYERS  # blk.2.* must be dropped (nextn_predict_layers=1)
INTER = 512
N_HEADS, HEAD_DIM = 8, 32
N_KV = 2
NK, DK = 16, 16
NV, DV = 32, 8
KEY_DIM = NK * DK                   # 256
VALUE_DIM = NV * DV                 # 256
CONV_DIM = 2 * KEY_DIM + VALUE_DIM  # 768
CONV_K = 2
ROPE_DIM = 8  # partial_rotary_factor = 8/32 = 0.25

# current llama.cpp qwen35 metadata conventions (arch-prefixed, ssm.*)
META = {
    "embedding_length": HIDDEN,
    "block_count": N_LAYERS + 1,  # includes the NextN/MTP block
    "nextn_predict_layers": 1,
    "attention.head_count": N_HEADS,
    "attention.head_count_kv": N_KV,
    "attention.key_length": HEAD_DIM,
    "attention.layer_norm_rms_epsilon": 1e-5,
    "feed_forward_length": INTER,
    "context_length": 4096,
    "full_attention_interval": 2,
    "rope.freq_base": 1000000.0,
    "rope.dimension_count": ROPE_DIM,
    "qwen35.ssm.group_count": NK,
    "qwen35.ssm.state_size": DK,
    "qwen35.ssm.time_step_rank": NV,
    "qwen35.ssm.inner_size": VALUE_DIM,
    "qwen35.ssm.conv_kernel": CONV_K,
}


def zip_ref(x, axis, block):
    """GGUF v-head storage: concat(hf[0::2], hf[1::2]) over head blocks."""
    moved = np.moveaxis(x, axis, 0)
    n = moved.shape[0] // block
    v = moved.reshape(n, block, *moved.shape[1:])
    out = np.concatenate([v[0::2], v[1::2]], axis=0).reshape(moved.shape)
    return np.moveaxis(out, 0, axis)


def build_fixture_gguf(tmp_path):
    rng = np.random.default_rng(42)

    def mat(r, c):
        return (rng.standard_normal((r, c)) * 0.02).astype(np.float32)

    def vec(n):
        return (rng.standard_normal((n,)) * 0.02).astype(np.float32)

    # --- quantized globals -------------------------------------------------
    d6 = 0.004  # keep fixture values in a realistic weight range (~±0.2)
    sc6 = np.array([1, 2, 3, -1, 0, 2, -2, 3, 2, -1, 2, -2, 3, 2, -1, 2], np.int8)
    q6 = (np.arange(QK_K) * 7 + 3) % 64
    tok_raw = b"".join(build_q6_block((q6 + r) % 64, sc6, d6) for r in range(VOCAB))
    tok_expected = np.stack([q6_expected((q6 + r) % 64, sc6, d6) for r in range(VOCAB)])

    d4, dmin4 = 0.004, 0.002
    sc4 = [1, 2, 3, 4, 2, 3, 1, 2]
    m4 = [0, 1, 2, 3, 1, 2, 0, 1]
    qs_base = (np.arange(QK_K) * 11 + 5) % 16
    out_rows, out_expected = [], []
    for r in range(VOCAB):
        qs = (qs_base + 3 * r) % 16
        out_rows.append(build_q4_block(qs, sc4, m4, d4, dmin4))
        out_expected.append(q4_expected(qs, sc4, m4, d4, dmin4))
    out_raw = b"".join(out_rows)
    out_expected = np.stack(out_expected)

    # --- f32 tensors --------------------------------------------------------
    f32 = [("output_norm.weight", np.abs(vec(HIDDEN)) + 1.0)]
    expectations = {}
    for i in range(N_LAYERS):
        p = f"blk.{i}."
        f32 += [
            (p + "attn_norm.weight", np.abs(vec(HIDDEN)) + 1.0),
            (p + "post_attention_norm.weight", np.abs(vec(HIDDEN)) + 1.0),
            (p + "ffn_gate.weight", mat(INTER, HIDDEN)),
            (p + "ffn_up.weight", mat(INTER, HIDDEN)),
            (p + "ffn_down.weight", mat(HIDDEN, INTER)),
        ]
        if i % 2 == 0:  # GDN linear attention layer
            natural = mat(CONV_DIM, HIDDEN)
            f32.append((p + "attn_qkv.weight",
                        np.concatenate([natural[: 2 * KEY_DIM], zip_ref(natural[2 * KEY_DIM:], 0, DV)])))
            expectations[f"language_model.model.layers.{i}.linear_attn.in_proj_qkv.weight"] = natural

            natural = mat(VALUE_DIM, HIDDEN)
            f32.append((p + "attn_gate.weight", zip_ref(natural, 0, DV)))
            expectations[f"language_model.model.layers.{i}.linear_attn.in_proj_z.weight"] = natural

            for src, dst in (("ssm_alpha", "in_proj_a"), ("ssm_beta", "in_proj_b")):
                natural = mat(NV, HIDDEN)
                f32.append((p + src + ".weight", zip_ref(natural, 0, 1)))
                expectations[f"language_model.model.layers.{i}.linear_attn.{dst}.weight"] = natural

            natural = mat(HIDDEN, VALUE_DIM)
            f32.append((p + "ssm_out.weight", zip_ref(natural, 1, DV)))
            expectations[f"language_model.model.layers.{i}.linear_attn.out_proj.weight"] = natural

            conv_natural = np.abs(mat(CONV_DIM, CONV_K)) * 0.5
            f32.append((p + "ssm_conv1d.weight",
                        np.concatenate([conv_natural[: 2 * KEY_DIM], zip_ref(conv_natural[2 * KEY_DIM:], 0, DV)])))
            expectations[f"language_model.model.layers.{i}.linear_attn.conv1d.weight"] = np.ascontiguousarray(
                conv_natural[..., None]
            )

            dt_natural = vec(NV)
            f32.append((p + "ssm_dt.bias", zip_ref(dt_natural, 0, 1)))
            expectations[f"language_model.model.layers.{i}.linear_attn.dt_bias"] = dt_natural

            a_log = -np.abs(vec(NV)) - 0.1
            f32.append((p + "ssm_a", zip_ref(-np.exp(a_log), 0, 1)))
            expectations[f"language_model.model.layers.{i}.linear_attn.A_log"] = a_log

            f32.append((p + "ssm_norm.weight", np.abs(vec(DV)) + 1.0))
        else:  # full attention layer (q rows carry fused [q; gate])
            f32 += [
                (p + "attn_q.weight", mat(2 * N_HEADS * HEAD_DIM, HIDDEN)),
                (p + "attn_k.weight", mat(N_KV * HEAD_DIM, HIDDEN)),
                (p + "attn_v.weight", mat(N_KV * HEAD_DIM, HIDDEN)),
                (p + "attn_output.weight", mat(HIDDEN, HIDDEN)),
                (p + "attn_q_norm.weight", np.abs(vec(HEAD_DIM)) + 1.0),
                (p + "attn_k_norm.weight", np.abs(vec(HEAD_DIM)) + 1.0),
            ]

    # MTP block: must be dropped by the explicit drop rule
    f32 += [
        (f"blk.{MTP}.attn_norm.weight", np.abs(vec(HIDDEN)) + 1.0),
        (f"blk.{MTP}.ffn_up.weight", mat(INTER, HIDDEN)),
        (f"blk.{MTP}.output.weight", mat(HIDDEN, VOCAB)),
    ]

    path = write_gguf(
        tmp_path / "tiny.gguf",
        arch="qwen35",
        metadata=META,
        f32_tensors=f32,
        quant_tensors=[
            # ne = (inner, rows): token_embd (VOCAB, HIDDEN) -> ne (256, 512)
            ("token_embd.weight", tok_raw, (HIDDEN, VOCAB), GGMLQuantizationType.Q6_K),
            ("output.weight", out_raw, (HIDDEN, VOCAB), GGMLQuantizationType.Q4_K),
        ],
    )
    return path, expectations, tok_expected, out_expected


def test_full_pipeline(tmp_path):
    gguf_path, expectations, tok_expected, out_expected = build_fixture_gguf(tmp_path)
    out_dir = tmp_path / "out"

    # sanity: stored ne uses the llama.cpp (inner, rows) convention
    tok = next(t for t in GGUFReader(str(gguf_path)).tensors if t.name == "token_embd.weight")
    assert tuple(tok.shape) == (HIDDEN, VOCAB)

    rc = cli_main([
        "convert", str(gguf_path),
        "--arch-config", QWEN35_YAML,
        "--output", str(out_dir),
        "--tokenizer-source", write_minimal_tokenizer(tmp_path / "tokenizer"),
        "--bits", "4", "--group-size", "64",
        "--chunk-mb", "1",  # tiny chunks -> exercise the streaming path
        "--quiet",
    ])
    assert rc == 0, "conversion failed"

    # ---- reload and inspect the output directory ----
    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["model_type"] == "qwen3_5"
    assert cfg["architectures"] == ["Qwen3_5ForCausalLM"]
    assert cfg["quantization"] == {"bits": 4, "group_size": 64, "mode": "affine"}
    tc = cfg["text_config"]
    assert tc["hidden_size"] == HIDDEN
    assert tc["num_hidden_layers"] == N_LAYERS
    assert tc["vocab_size"] == VOCAB  # resolved from token_embd shape (tshape spec)
    assert tc["linear_num_value_heads"] == NV
    assert tc["linear_num_key_heads"] == NK
    assert tc["linear_key_head_dim"] == DK
    assert tc["linear_value_head_dim"] == DV
    assert tc["full_attention_interval"] == 2
    assert tc["rms_norm_eps"] == pytest.approx(1e-5)  # f32 round-trip through GGUF
    assert tc["max_position_embeddings"] == 4096
    assert tc["attention_bias"] is False
    assert tc["attention_dropout"] == 0.0
    assert tc["attn_output_gate"] is True
    assert tc["hidden_act"] == "silu"
    assert tc["rope_parameters"]["rope_theta"] == 1000000.0
    assert tc["rope_parameters"]["partial_rotary_factor"] == ROPE_DIM / HEAD_DIM
    assert cfg["tie_word_embeddings"] is False  # output.weight present
    def dig(d, dotted):
        node = d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    assert all(dig(cfg, f) is not None for f in [
        "tie_word_embeddings", "text_config.hidden_size", "text_config.vocab_size"])

    all_keys = set(wm)
    assert not any(f".layers.{MTP}." in k or "mtp" in k.lower() for k in all_keys)
    assert "language_model.model.embed_tokens.weight" in all_keys
    assert "language_model.lm_head.weight" in all_keys
    assert "language_model.model.layers.0.linear_attn.A_log" in all_keys
    assert "language_model.model.layers.1.self_attn.q_proj.weight" in all_keys

    tensors = {}
    for fname in sorted(set(wm.values())):
        tensors.update(load_file(str(out_dir / fname)))

    # ---- exact f32 tensors ----
    assert tensors["language_model.model.norm.weight"].dtype == np.float32
    a_log = tensors["language_model.model.layers.0.linear_attn.A_log"]
    np.testing.assert_allclose(
        a_log, expectations["language_model.model.layers.0.linear_attn.A_log"], atol=1e-6
    )
    dt = tensors["language_model.model.layers.0.linear_attn.dt_bias"]
    np.testing.assert_allclose(
        dt, expectations["language_model.model.layers.0.linear_attn.dt_bias"], atol=1e-6
    )
    conv = tensors["language_model.model.layers.0.linear_attn.conv1d.weight"]
    assert conv.shape == (CONV_DIM, CONV_K, 1)
    np.testing.assert_allclose(
        conv[..., 0],
        expectations["language_model.model.layers.0.linear_attn.conv1d.weight"][..., 0],
        atol=1e-6,
    )

    # ---- quantized outputs: requantization idempotence checks ----
    def qdequant(key):
        base = key[: -len(".weight")]
        return np.asarray(
            dequantize_weights(tensors[key], tensors[base + ".scales"],
                               tensors[base + ".biases"], bits=4, group_size=64)
        )

    def ref_dequant(expected):
        p, s, b = quantize_weights(expected, 4, 64)
        return np.asarray(dequantize_weights(p, s, b, bits=4, group_size=64))

    d = qdequant("language_model.model.embed_tokens.weight")
    assert d.shape == (VOCAB, HIDDEN)
    np.testing.assert_allclose(d, ref_dequant(tok_expected), atol=1e-6)

    d = qdequant("language_model.lm_head.weight")
    assert d.shape == (VOCAB, HIDDEN)
    np.testing.assert_allclose(d, ref_dequant(out_expected), atol=1e-6)

    d = qdequant("language_model.model.layers.0.linear_attn.in_proj_qkv.weight")
    assert d.shape == (CONV_DIM, HIDDEN)
    # v-rows must come out in the natural (unpermuted) head order
    np.testing.assert_allclose(
        d, ref_dequant(expectations["language_model.model.layers.0.linear_attn.in_proj_qkv.weight"]),
        atol=1e-6,
    )

    d = qdequant("language_model.model.layers.0.linear_attn.out_proj.weight")
    np.testing.assert_allclose(
        d, ref_dequant(expectations["language_model.model.layers.0.linear_attn.out_proj.weight"]),
        atol=1e-6,
    )

    # ---- official verifier pass ----
    rc = cli_main([
        "verify", str(gguf_path), str(out_dir),
        "--arch-config", QWEN35_YAML, "--bits", "4",
    ])
    assert rc == 0, "verify failed"


def test_convert_bits_auto_resolves_dominant_family(tmp_path):
    """--bits auto (default) resolves the global target from the byte-weighted
    source quant histogram; the evidence lands in the output config.json."""
    gguf_path, _, _, _ = build_fixture_gguf(tmp_path)
    out_dir = tmp_path / "out-auto"

    rc = cli_main([
        "convert", str(gguf_path),
        "--arch-config", QWEN35_YAML,
        "--output", str(out_dir),
        "--tokenizer-source", write_minimal_tokenizer(tmp_path / "tokenizer-auto"),
        "--chunk-mb", "1",
        "--quiet",
    ])
    assert rc == 0, "auto-bits conversion failed"

    cfg = json.loads((out_dir / "config.json").read_text())
    # fixture histogram: token_embd Q6_K (107520 B) outweighs output Q4_K
    # (73728 B) -> dominant family Q6_K -> global target 6 bits
    assert cfg["quantization"]["bits"] == 6
    sel = cfg["quantization_selection"]
    assert sel["requested"] == "auto"
    assert sel["target_bits"] == 6
    assert sel["dominant_source_type"] == "Q6_K"
    assert sel["source_quant_histogram_bytes"] == {"Q6_K": 107520, "Q4_K": 73728}
    assert "90.1%" not in sel["reason"] and "auto" in sel["reason"]

    # an explicit --bits must always win and be recorded as explicit
    out_dir2 = tmp_path / "out-explicit"
    rc = cli_main([
        "convert", str(gguf_path),
        "--arch-config", QWEN35_YAML,
        "--output", str(out_dir2),
        "--tokenizer-source", write_minimal_tokenizer(tmp_path / "tokenizer-explicit"),
        "--bits", "4", "--chunk-mb", "1",
        "--quiet",
    ])
    assert rc == 0
    cfg2 = json.loads((out_dir2 / "config.json").read_text())
    assert cfg2["quantization"]["bits"] == 4
    assert cfg2["quantization_selection"]["requested"] == "4"
    assert cfg2["quantization_selection"]["target_bits"] == 4


def test_convert_bits_auto_3bit_warns_but_converts(tmp_path, monkeypatch, capsys):
    """Final same-bit auto policy: an auto-derived 3-bit target converts
    normally and emits a fidelity warning (stderr + output config.json);
    explicit --bits is never re-warned."""
    gguf_path, _, _, _ = build_fixture_gguf(tmp_path)

    # force the dominant-family mapping to 3 bits (the fixture's dominant
    # family is Q6_K -> 6); this simulates an IQ3/Q3-dominant source without
    # needing Q3 block builders
    import gguf2mlx_stream.quant_select as qs
    monkeypatch.setattr(qs, "family_bits", lambda name: 3 if name.startswith(("Q", "IQ")) else None)

    out_dir = tmp_path / "out-auto-3bit"
    rc = cli_main([
        "convert", str(gguf_path),
        "--arch-config", QWEN35_YAML,
        "--output", str(out_dir),
        "--tokenizer-source", write_minimal_tokenizer(tmp_path / "tokenizer-warn"),
        "--chunk-mb", "1",
        "--quiet",
    ])
    assert rc == 0, "auto 3-bit conversion must not be blocked"

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "paired-oracle" in err
    assert "4-bit or higher is recommended" in err

    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["quantization"]["bits"] == 3
    sel = cfg["quantization_selection"]
    assert sel["requested"] == "auto"
    assert sel["target_bits"] == 3
    assert "4-bit or higher is recommended" in sel["fidelity_warning"]

    # explicit --bits 3 is the user's own choice: no warning, same result
    out_dir2 = tmp_path / "out-explicit-3bit"
    rc2 = cli_main([
        "convert", str(gguf_path),
        "--arch-config", QWEN35_YAML,
        "--output", str(out_dir2),
        "--tokenizer-source", write_minimal_tokenizer(tmp_path / "tokenizer-warn2"),
        "--bits", "3", "--chunk-mb", "1",
        "--quiet",
    ])
    assert rc2 == 0
    assert "WARNING" not in capsys.readouterr().err
    cfg2 = json.loads((out_dir2 / "config.json").read_text())
    assert cfg2["quantization"]["bits"] == 3
    assert "fidelity_warning" not in cfg2["quantization_selection"]
