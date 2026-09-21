"""Structural regression against the locally converted Nyx reference models.

Without the (since-deleted) source GGUFs, these tests anchor the declarative
system to the *proven outputs* of the golden converter:

1. a tiny "skeleton" GGUF carrying the exact real Nyx tensor-name inventory
   is planned with ``configs/qwen3_5.yaml``; the planned destination key set
   (including quantization companions) must equal the reference
   ``model.safetensors.index.json`` weight map exactly;
2. the reference output directories themselves are checked structurally
   (index consistency, quantization triple shapes, config, finiteness).

Reference locations can be overridden via env vars; tests skip cleanly when
the references are not present:

    GGUF2MLX_NYX_REFERENCE_6BIT  (default: ~/Models/MLX/Nyx-...-MLX-6bit)
    GGUF2MLX_NYX_REFERENCE_4BIT  (default: ~/Models/MLX/Nyx-...-MLX-4bit)
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import write_gguf
from safetensors import safe_open

from gguf2mlx_stream.config.schema import load_arch_config
from gguf2mlx_stream.planner import plan_conversion
from gguf2mlx_stream.source.gguf import GGUFSource

ROOT = Path(__file__).resolve().parent.parent
QWEN35_YAML = ROOT / "configs" / "qwen3_5.yaml"

_HOME = Path.home()
REF_6BIT = Path(os.environ.get(
    "GGUF2MLX_NYX_REFERENCE_6BIT",
    _HOME / "Models/MLX/Nyx-RP-9B-Instruct-2608-v1-MLX-6bit",
))
REF_4BIT = Path(os.environ.get(
    "GGUF2MLX_NYX_REFERENCE_4BIT",
    _HOME / "Models/MLX/Nyx-RP-9B-Instruct-2608-v1-MLX-4bit",
))

# tiny geometry that satisfies every expect_shape relation in qwen3_5.yaml
HIDDEN, VOCAB, INTER = 64, 128, 128
N_LAYERS = 32
N_HEADS, HEAD_DIM, N_KV = 4, 16, 2
NK, DK, NV, DV = 4, 8, 8, 8
KEY_DIM, VALUE_DIM = NK * DK, NV * DV          # 32, 64
CONV_DIM = 2 * KEY_DIM + VALUE_DIM             # 128
CONV_K = 4


def _reference_ref_config(ref_dir: Path) -> dict:
    cfg = json.loads((ref_dir / "config.json").read_text())
    return cfg["text_config"]


def build_nyx_skeleton_gguf(tmp_path: Path) -> Path:
    """Tiny f32 GGUF whose tensor names mirror the real Nyx inventory."""
    rng = np.random.default_rng(7)

    def mat(r, c):
        return (rng.standard_normal((r, c)) * 0.01).astype(np.float32)

    def vec(n):
        return (rng.standard_normal((n,)) * 0.01).astype(np.float32)

    tensors = [
        ("token_embd.weight", mat(VOCAB, HIDDEN)),
        ("output.weight", mat(VOCAB, HIDDEN)),
        ("output_norm.weight", np.abs(vec(HIDDEN)) + 1.0),
    ]
    for i in range(N_LAYERS):
        p = f"blk.{i}."
        tensors += [
            (p + "attn_norm.weight", np.abs(vec(HIDDEN)) + 1.0),
            (p + "post_attention_norm.weight", np.abs(vec(HIDDEN)) + 1.0),
            (p + "ffn_gate.weight", mat(INTER, HIDDEN)),
            (p + "ffn_up.weight", mat(INTER, HIDDEN)),
            (p + "ffn_down.weight", mat(HIDDEN, INTER)),
        ]
        if (i + 1) % 4 == 0:  # full attention layer
            tensors += [
                (p + "attn_q.weight", mat(2 * N_HEADS * HEAD_DIM, HIDDEN)),
                (p + "attn_k.weight", mat(N_KV * HEAD_DIM, HIDDEN)),
                (p + "attn_v.weight", mat(N_KV * HEAD_DIM, HIDDEN)),
                (p + "attn_output.weight", mat(HIDDEN, HIDDEN)),
                (p + "attn_q_norm.weight", np.abs(vec(HEAD_DIM)) + 1.0),
                (p + "attn_k_norm.weight", np.abs(vec(HEAD_DIM)) + 1.0),
            ]
        else:  # GDN linear attention layer
            tensors += [
                (p + "attn_qkv.weight", mat(CONV_DIM, HIDDEN)),
                (p + "attn_gate.weight", mat(VALUE_DIM, HIDDEN)),
                (p + "ssm_alpha.weight", mat(NV, HIDDEN)),
                (p + "ssm_beta.weight", mat(NV, HIDDEN)),
                (p + "ssm_out.weight", mat(HIDDEN, VALUE_DIM)),
                (p + "ssm_conv1d.weight", mat(CONV_DIM, CONV_K)),
                (p + "ssm_dt.bias", vec(NV)),
                (p + "ssm_a", -np.abs(vec(NV)) - 0.1),
                (p + "ssm_norm.weight", np.abs(vec(DV)) + 1.0),
            ]
    # MTP block: 15 tensors, all dropped
    mtp_names = [
        "attn_norm.weight", "post_attention_norm.weight", "attn_qkv.weight",
        "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight",
        "ssm_out.weight", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a",
        "ssm_norm.weight", "ffn_gate.weight", "ffn_up.weight",
        "ffn_down.weight", "output.weight",
    ]
    for name in mtp_names:
        tensors.append((f"blk.{N_LAYERS}.{name}", vec(4)))
    return Path(write_gguf(
        tmp_path / "nyx_skeleton.gguf",
        arch="qwen3_5_text",
        metadata={
            "embedding_length": HIDDEN,
            # block_count includes the NextN/MTP block (llama.cpp convention)
            "block_count": N_LAYERS + 1,
            "nextn_predict_layers": 1,
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
        },
        f32_tensors=tensors,
    ))


def _planned_key_set(plan) -> set[str]:
    keys = set()
    for job in plan.jobs:
        keys.add(job.dest)
        if job.quantize:
            base = job.dest.removesuffix(".weight")
            keys |= {base + ".scales", base + ".biases"}
    return keys


@pytest.fixture(scope="module")
def skeleton_plan(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("skeleton")
    gguf = build_nyx_skeleton_gguf(tmp)
    cfg = load_arch_config(str(QWEN35_YAML))
    source = GGUFSource(str(gguf))
    ref = _reference_ref_config(REF_6BIT)
    return plan_conversion(cfg, source, ref_config=ref), source


@pytest.mark.skipif(not REF_6BIT.is_dir(), reason=f"reference not present: {REF_6BIT}")
def test_skeleton_key_set_matches_reference_6bit(skeleton_plan):
    plan, _ = skeleton_plan
    index = json.loads((REF_6BIT / "model.safetensors.index.json").read_text())
    ref_keys = set(index["weight_map"])
    planned = _planned_key_set(plan)
    missing = sorted(ref_keys - planned)
    extra = sorted(planned - ref_keys)
    assert not missing, f"planner misses reference keys: {missing[:8]}"
    assert not extra, f"planner invents keys absent from reference: {extra[:8]}"
    assert len(planned) == 927


@pytest.mark.skipif(not REF_6BIT.is_dir(), reason=f"reference not present: {REF_6BIT}")
def test_skeleton_layer_kinds_match_reference(skeleton_plan):
    _plan, _ = skeleton_plan
    index = json.loads((REF_6BIT / "model.safetensors.index.json").read_text())
    wm = set(index["weight_map"])
    prefix = "language_model.model.layers."
    for i in range(N_LAYERS):
        if (i + 1) % 4 == 0:
            assert f"{prefix}{i}.self_attn.q_proj.weight" in wm
            assert f"{prefix}{i}.linear_attn.in_proj_qkv.weight" not in wm
        else:
            assert f"{prefix}{i}.linear_attn.in_proj_qkv.weight" in wm
            assert f"{prefix}{i}.self_attn.q_proj.weight" not in wm


def _check_reference_structure(ref_dir: Path, report: list):
    cfg = json.loads((ref_dir / "config.json").read_text())
    index = json.loads((ref_dir / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]

    assert cfg["model_type"] == "qwen3_5"
    assert cfg["architectures"] == ["Qwen3_5ForCausalLM"]
    assert cfg["quantization"] == cfg["quantization_config"]
    tc = cfg["text_config"]
    assert tc["hidden_size"] == 4096 and tc["num_hidden_layers"] == 32
    assert tc["vocab_size"] == 248320 and tc["intermediate_size"] == 12288
    assert tc["linear_num_value_heads"] == 32 and tc["linear_value_head_dim"] == 128
    assert tc["full_attention_interval"] == 4

    # every shard file exists
    for fname in sorted(set(wm.values())):
        assert (ref_dir / fname).is_file(), fname

    total = 0
    n_quant = n_f32 = 0
    for fname in sorted(set(wm.values())):
        with safe_open(str(ref_dir / fname), framework="numpy") as f:
            # safe_open.keys() returns a plain list; the handle itself is not
            # iterable, so iterating .keys() is intentional here
            for key in f.keys():  # noqa: SIM118
                assert wm[key] == fname, f"index mismatch for {key}"
                arr = f.get_tensor(key)
                total += arr.nbytes
                base = key.removesuffix(".weight")
                if key.endswith((".scales", ".biases")):
                    assert arr.dtype == np.float16
                    n_f32 += 1
                elif key.endswith(".weight") and f"{base}.scales" in wm:
                    assert arr.dtype == np.uint32
                    n_quant += 1
                    with safe_open(str(ref_dir / wm[f"{base}.scales"]), framework="numpy") as f2:
                        scales = f2.get_tensor(f"{base}.scales")
                    assert scales.shape[0] == arr.shape[0]
                else:
                    n_f32 += 1
                if arr.dtype in (np.float16, np.float32):
                    assert np.isfinite(arr).all(), f"non-finite values in {key}"
    assert abs(total - index["metadata"]["total_size"]) < 1024
    report.append(f"{ref_dir.name}: {len(wm)} keys, {n_quant} quantized weights, "
                  f"{total / 2**30:.2f} GiB")

    # structural spot checks on special tensors
    def get(key):
        with safe_open(str(ref_dir / wm[key]), framework="numpy") as f:
            return f.get_tensor(key)

    conv = get("language_model.model.layers.0.linear_attn.conv1d.weight")
    assert conv.shape == (8192, 4, 1) and conv.dtype == np.float32
    a_log = get("language_model.model.layers.0.linear_attn.A_log")
    assert a_log.shape == (32,) and np.isfinite(a_log).all()
    dt = get("language_model.model.layers.0.linear_attn.dt_bias")
    assert dt.shape == (32,)
    packed, scales = get("language_model.model.embed_tokens.weight"), \
        get("language_model.model.embed_tokens.scales")
    assert scales.shape == (248320, 64)   # 4096 / 64 groups
    # per group: 64 elems * bits / 32 = 2*bits u32; 64 groups per row
    bits = cfg["quantization"]["bits"]
    assert packed.shape == (248320, 2 * bits * 64)


@pytest.mark.skipif(not REF_6BIT.is_dir(), reason=f"reference not present: {REF_6BIT}")
def test_reference_6bit_structure():
    report = []
    _check_reference_structure(REF_6BIT, report)
    print(report[0])


@pytest.mark.skipif(not REF_4BIT.is_dir(), reason=f"reference not present: {REF_4BIT}")
def test_reference_4bit_structure():
    report = []
    _check_reference_structure(REF_4BIT, report)
    print(report[0])
