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
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType
from safetensors.numpy import load_file

from gguf2mlx_stream.cli import main as cli_main
from gguf2mlx_stream.quantize import dequantize_weights, quantize_weights

from conftest import (
    QK_K,
    build_q4_block,
    build_q6_block,
    q4_expected,
    q6_expected,
    write_gguf,
)

ROOT = Path(__file__).resolve().parent.parent
QWEN35_YAML = str(ROOT / "configs" / "qwen3_5.yaml")

# ---------------------------------------------------------------------------
# fixture geometry (must be consistent with configs/qwen3_5.yaml semantics)
# ---------------------------------------------------------------------------

HIDDEN = 256
VOCAB = 512
N_LAYERS = 2  # layer 0: linear attention, layer 1: full attention
MTP = N_LAYERS  # blk.2.* must be dropped
INTER = 512
N_HEADS, HEAD_DIM = 8, 32
N_KV = 2
NK, DK = 16, 16
NV, DV = 32, 8
KEY_DIM = NK * DK                   # 256
VALUE_DIM = NV * DV                 # 256
CONV_DIM = 2 * KEY_DIM + VALUE_DIM  # 768
CONV_K = 2

META = {
    "embedding_length": HIDDEN,
    "block_count": N_LAYERS,
    "vocab_size": VOCAB,
    "attention.head_count": N_HEADS,
    "attention.head_count_kv": N_KV,
    "attention.key_length": HEAD_DIM,
    "feed_forward_length": INTER,
    "linear_num_key_heads": NK,
    "linear_key_head_dim": DK,
    "linear_num_value_heads": NV,
    "linear_value_head_dim": DV,
    "linear_conv_kernel_dim": CONV_K,
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
        arch="qwen3_5_text",
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
    tok = [t for t in GGUFReader(str(gguf_path)).tensors if t.name == "token_embd.weight"][0]
    assert tuple(tok.shape) == (HIDDEN, VOCAB)

    rc = cli_main([
        "convert", str(gguf_path),
        "--arch-config", QWEN35_YAML,
        "--output", str(out_dir),
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
    assert cfg["text_config"]["hidden_size"] == HIDDEN
    assert cfg["text_config"]["num_hidden_layers"] == N_LAYERS
    assert "linear_num_value_heads" not in cfg["text_config"]  # needs a ref config

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
