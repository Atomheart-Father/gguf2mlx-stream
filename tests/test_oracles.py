"""Independent oracle tests.

The production pipeline (planner + runner) is driven over tiny synthetic
GGUFs, but every EXPECTED value is computed by ``tests/oracle_impl.py`` —
pure-numpy reimplementations that share no code with production operators.
If a production operator is wrong, these tests fail instead of silently
agreeing with a buggy verifier.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).parent))
import oracle_impl as oracle
from conftest import write_gguf, write_minimal_tokenizer

from gguf2mlx_stream.config.schema import load_arch_config
from gguf2mlx_stream.planner import plan_conversion
from gguf2mlx_stream.runner import ConversionRunner, QuantSettings
from gguf2mlx_stream.source.gguf import GGUFSource

ROOT = Path(__file__).resolve().parent.parent
QWEN35_YAML = ROOT / "configs" / "qwen3_5.yaml"
LLAMA_YAML = ROOT / "configs" / "llama.yaml"

# ---------------------------------------------------------------------------
# ratio matrix: the generic reorder must hold for any heads/kv-heads ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [1, 2, 3, 4])
def test_grouped_head_reorder_ratio_matrix(tmp_path, ratio):
    """Oracle-vs-pipeline: v-head rows/cols and block-1 vectors per ratio."""
    nk = 4                      # kv-head groups
    nv = nk * ratio             # value heads
    dk, dv = 4, 6               # per-head dims (block sizes)
    hidden, inter, vocab = 24, 32, 64
    key_dim, value_dim = nk * dk, nv * dv
    conv_dim = 2 * key_dim + value_dim
    conv_k = 2
    n_layers = 2                # one GDN layer (0) + one full-attn layer (1)
    n_heads, n_kv, head_dim = 4, 2, 8

    rng = np.random.default_rng(ratio * 11)

    def mat(r, c):
        return (rng.standard_normal((r, c)) * 0.05).astype(np.float32)

    def vec(n):
        return (rng.standard_normal((n,)) * 0.05).astype(np.float32)

    tensors = [
        ("token_embd.weight", mat(vocab, hidden)),
        ("output_norm.weight", np.abs(vec(hidden)) + 1.0),
    ]
    expectations = {}

    p = "blk.0."
    # qkv: gguf stores [q; k; permuted v]; dest must be natural order.
    # GDN q/k rows use the GDN key geometry (nk*dk each), not full-attention.
    q, k, v = mat(key_dim, hidden), mat(key_dim, hidden), mat(value_dim, hidden)
    tensors.append((p + "attn_qkv.weight",
                    np.concatenate([q, k, oracle.oracle_permute_heads(v, 0, dv, ratio)])))
    expectations["language_model.model.layers.0.linear_attn.in_proj_qkv.weight"] = \
        np.concatenate([q, k, v])

    z = mat(value_dim, hidden)
    tensors.append((p + "attn_gate.weight", oracle.oracle_permute_heads(z, 0, dv, ratio)))
    expectations["language_model.model.layers.0.linear_attn.in_proj_z.weight"] = z

    a, b = mat(nv, hidden), mat(nv, hidden)
    tensors.append((p + "ssm_alpha.weight", oracle.oracle_permute_heads(a, 0, 1, ratio)))
    tensors.append((p + "ssm_beta.weight", oracle.oracle_permute_heads(b, 0, 1, ratio)))
    expectations["language_model.model.layers.0.linear_attn.in_proj_a.weight"] = a
    expectations["language_model.model.layers.0.linear_attn.in_proj_b.weight"] = b

    o = mat(hidden, value_dim)
    tensors.append((p + "ssm_out.weight", oracle.oracle_permute_heads(o, 1, dv, ratio)))
    expectations["language_model.model.layers.0.linear_attn.out_proj.weight"] = o

    conv = np.abs(mat(conv_dim, conv_k)) * 0.5
    tensors.append((p + "ssm_conv1d.weight",
                    np.concatenate([conv[:2 * key_dim],
                                    oracle.oracle_permute_heads(conv[2 * key_dim:], 0, dv, ratio)])))
    expectations["language_model.model.layers.0.linear_attn.conv1d.weight"] = conv[..., None]

    dt = vec(nv)
    tensors.append((p + "ssm_dt.bias", oracle.oracle_permute_heads(dt, 0, 1, ratio)))
    expectations["language_model.model.layers.0.linear_attn.dt_bias"] = dt

    a_log = -np.abs(vec(nv)) - 0.1
    tensors.append((p + "ssm_a", oracle.oracle_permute_heads(-np.exp(a_log), 0, 1, ratio)))
    expectations["language_model.model.layers.0.linear_attn.A_log"] = a_log

    tensors.append((p + "ssm_norm.weight", np.abs(vec(dv)) + 1.0))
    tensors += [
        (p + "attn_norm.weight", np.abs(vec(hidden)) + 1.0),
        (p + "post_attention_norm.weight", np.abs(vec(hidden)) + 1.0),
        (p + "ffn_gate.weight", mat(inter, hidden)),
        (p + "ffn_up.weight", mat(inter, hidden)),
        (p + "ffn_down.weight", mat(hidden, inter)),
    ]

    p = "blk.1."
    tensors += [
        (p + "attn_q.weight", mat(2 * n_heads * head_dim, hidden)),  # fused [q; gate]
        (p + "attn_k.weight", mat(n_kv * head_dim, hidden)),
        (p + "attn_v.weight", mat(n_kv * head_dim, hidden)),
        (p + "attn_output.weight", mat(hidden, n_heads * head_dim)),
        (p + "attn_q_norm.weight", np.abs(vec(head_dim)) + 1.0),
        (p + "attn_k_norm.weight", np.abs(vec(head_dim)) + 1.0),
        (p + "attn_norm.weight", np.abs(vec(hidden)) + 1.0),
        (p + "post_attention_norm.weight", np.abs(vec(hidden)) + 1.0),
        (p + "ffn_gate.weight", mat(inter, hidden)),
        (p + "ffn_up.weight", mat(inter, hidden)),
        (p + "ffn_down.weight", mat(hidden, inter)),
    ]

    gguf_path = write_gguf(
        tmp_path / f"ratio{ratio}.gguf", arch="qwen35",
        metadata={
            "embedding_length": hidden, "block_count": n_layers,
            "nextn_predict_layers": 0,
            "attention.head_count": n_heads, "attention.head_count_kv": n_kv,
            "attention.key_length": head_dim,
            "attention.layer_norm_rms_epsilon": 1e-5,
            "feed_forward_length": inter, "context_length": 512,
            "full_attention_interval": 2,
            "rope.freq_base": 1e6, "rope.dimension_count": 2,
            "qwen35.ssm.group_count": nk, "qwen35.ssm.state_size": dk,
            "qwen35.ssm.time_step_rank": nv, "qwen35.ssm.inner_size": value_dim,
            "qwen35.ssm.conv_kernel": conv_k,
        },
        f32_tensors=tensors,
    )

    cfg = load_arch_config(str(QWEN35_YAML))
    source = GGUFSource(str(gguf_path))
    plan = plan_conversion(cfg, source)
    out = tmp_path / "out"
    runner = ConversionRunner(plan, source, str(out),
                              quant=QuantSettings(bits=None),
                              tokenizer_source=write_minimal_tokenizer(tmp_path / "tokenizer"),
                              log=lambda _: None)
    runner.run()
    tensors_out = {}
    for f in out.glob("*.safetensors"):
        tensors_out.update(load_file(str(f)))

    # oracle comparisons on every GDN-transformed tensor
    for name, expected in expectations.items():
        got = tensors_out[name]
        assert got.shape == expected.shape, name
        assert not oracle.oracle_has_nan_or_inf(got), name
        assert oracle.oracle_close(got, expected, tol=5e-3), \
            f"ratio {ratio}: oracle mismatch in {name}"

    # oracle-vs-production operator cross-check is enforced implicitly:
    # expectations used oracle.oracle_permute_heads to BUILD the gguf fixture,
    # so any disagreement between production and oracle breaks the round-trip.


def test_oracle_forward_inverse_roundtrip_all_ratios():
    """The oracle itself must be self-consistent for ratio 1..4."""
    for ratio in (1, 2, 3, 4):
        x = np.arange(4 * ratio * 3, dtype=np.float32).reshape(4 * ratio, 3)
        rt = oracle.oracle_unpermute_heads(
            oracle.oracle_permute_heads(x, 0, 1, ratio), 0, 1, ratio)
        assert np.array_equal(rt, x)


def test_a_log_oracle_matches_documented_formula():
    """A_log = log(-unperm(ssm_a)); check against hand-computed values."""
    # natural A values chosen for clean logs
    natural_a = np.array([2.0, 4.0, 0.5, 1.0], np.float32)  # 4 v-heads
    a_log_expected = np.log(natural_a).astype(np.float32)  # A_log = log(A) = log(-ssm_a)
    # gguf ssm_a = -exp(A_log) in gguf head order (ratio 2: [h0, h2, h1, h3])
    a_log = np.log(natural_a)
    ssm_a = -np.exp(a_log)
    ssm_a_gguf = np.array([ssm_a[0], ssm_a[2], ssm_a[1], ssm_a[3]], np.float32)
    got = oracle.oracle_a_log(ssm_a_gguf, value_heads=4, key_heads=2)
    np.testing.assert_allclose(got, a_log_expected, atol=1e-6)


# ---------------------------------------------------------------------------
# structural oracles
# ---------------------------------------------------------------------------


def test_mtp_removal_oracle_key_set(tmp_path):
    """Planned dest keys must equal an independently stated key set."""
    hidden, inter, vocab = 16, 24, 32
    n_layers = 4
    n_heads, n_kv, head_dim = 4, 2, 8
    nk, dk, nv, dv = 4, 4, 8, 4
    value_dim = 32
    conv_dim, conv_k = 64, 2

    rng = np.random.default_rng(3)

    def mat(r, c):
        return (rng.standard_normal((r, c)) * 0.05).astype(np.float32)

    def vec(n):
        return (rng.standard_normal((n,)) * 0.05).astype(np.float32)

    tensors = [("token_embd.weight", mat(vocab, hidden)),
               ("output_norm.weight", np.abs(vec(hidden)) + 1.0)]
    full_attn_layers = {1, 3}  # interval 2
    for i in range(n_layers + 1):  # +1: the MTP block, dropped
        p = f"blk.{i}."
        tensors += [
            (p + "attn_norm.weight", np.abs(vec(hidden)) + 1.0),
            (p + "post_attention_norm.weight", np.abs(vec(hidden)) + 1.0),
            (p + "ffn_gate.weight", mat(inter, hidden)),
            (p + "ffn_up.weight", mat(inter, hidden)),
            (p + "ffn_down.weight", mat(hidden, inter)),
        ]
        if i in full_attn_layers:
            tensors += [
                (p + "attn_q.weight", mat(2 * n_heads * head_dim, hidden)),
                (p + "attn_k.weight", mat(n_kv * head_dim, hidden)),
                (p + "attn_v.weight", mat(n_kv * head_dim, hidden)),
                (p + "attn_output.weight", mat(hidden, n_heads * head_dim)),
                (p + "attn_q_norm.weight", np.abs(vec(head_dim)) + 1.0),
                (p + "attn_k_norm.weight", np.abs(vec(head_dim)) + 1.0),
            ]
        elif i < n_layers:
            tensors += [
                (p + "attn_qkv.weight", mat(conv_dim, hidden)),
                (p + "attn_gate.weight", mat(value_dim, hidden)),
                (p + "ssm_alpha.weight", mat(nv, hidden)),
                (p + "ssm_beta.weight", mat(nv, hidden)),
                (p + "ssm_out.weight", mat(hidden, value_dim)),
                (p + "ssm_conv1d.weight", mat(conv_dim, conv_k)),
                (p + "ssm_dt.bias", vec(nv)),
                (p + "ssm_a", -np.abs(vec(nv)) - 0.1),
                (p + "ssm_norm.weight", np.abs(vec(dv)) + 1.0),
            ]

    gguf_path = write_gguf(
        tmp_path / "mtp.gguf", arch="qwen35",
        metadata={
            "embedding_length": hidden, "block_count": n_layers + 1,
            "nextn_predict_layers": 1,
            "attention.head_count": n_heads, "attention.head_count_kv": n_kv,
            "attention.key_length": head_dim,
            "attention.layer_norm_rms_epsilon": 1e-5,
            "feed_forward_length": inter, "context_length": 512,
            "full_attention_interval": 2,
            "rope.freq_base": 1e6, "rope.dimension_count": 2,
            "qwen35.ssm.group_count": nk, "qwen35.ssm.state_size": dk,
            "qwen35.ssm.time_step_rank": nv, "qwen35.ssm.inner_size": value_dim,
            "qwen35.ssm.conv_kernel": conv_k,
        },
        f32_tensors=tensors,
    )
    cfg = load_arch_config(str(QWEN35_YAML))
    plan = plan_conversion(cfg, GGUFSource(str(gguf_path)))

    def expand(tpl):
        return tpl.replace("{layer}", "LAYER")

    expected: set[str] = set()
    for k in oracle.oracle_expected_layer_keys(
            n_layers, full_attn_layers, quantized_keys=True,
            prefix="language_model."):
        if "LAYER" in k:
            for i in range(n_layers):
                expected.add(k.replace("LAYER", str(i)))
        else:
            expected.add(k)

    planned = set(plan.dest_map)
    for job in plan.jobs:
        if job.quantize:
            base = job.dest.removesuffix(".weight")
            planned |= {base + ".scales", base + ".biases"}

    missing = expected - planned
    extra = planned - expected
    assert not missing, f"planner misses oracle keys: {sorted(missing)[:6]}"
    assert not extra, f"planner invents keys: {sorted(extra)[:6]}"
    # the fixture carries no output.weight -> tied -> no lm_head keys expected
    assert not any("lm_head" in k for k in planned)
    # and the MTP block is explicitly dropped
    assert plan.dropped


def test_oracle_q_gate_fusion_semantics():
    """Fused [q; gate] rows pass through unchanged; oracle states it."""
    q = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    g = np.arange(3 * 4, dtype=np.float32).reshape(3, 4) + 100
    fused_expected, _ = oracle.oracle_q_gate_fusion(q, g)
    assert fused_expected.shape == (6, 4)
    np.testing.assert_array_equal(fused_expected[:3], q)
    np.testing.assert_array_equal(fused_expected[3:], g)


def test_oracle_conv1d_semantics():
    natural = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)
    got = oracle.oracle_conv1d(natural, key_dim=4, value_block=2,
                               value_heads=2, key_heads=1)
    assert got.shape == (8, 2, 1)
    np.testing.assert_array_equal(got[..., 0], natural)


# ---------------------------------------------------------------------------
# llama q/k out-axis unpermute (llama.cpp convert-time storage order)
# ---------------------------------------------------------------------------


def test_llama_qk_unpermute_oracle(tmp_path):
    """Oracle-vs-pipeline: llama attn_q/attn_k storage order must be undone."""
    n_heads, n_kv, head_dim = 4, 2, 8
    hidden, inter, vocab, n_layers = 24, 32, 64, 2

    rng = np.random.default_rng(7)

    def mat(r, c):
        return (rng.standard_normal((r, c)) * 0.05).astype(np.float32)

    def vec(n):
        return (rng.standard_normal((n,)) * 0.05).astype(np.float32)

    # the oracle must round-trip against itself first
    x = np.arange(n_heads * head_dim * 3, dtype=np.float32).reshape(n_heads * head_dim, 3)
    assert np.array_equal(oracle.oracle_llama_unpermute_qk(
        oracle.oracle_llama_permute_qk(x, n_heads, head_dim), n_heads, head_dim), x)

    tensors = [("token_embd.weight", mat(vocab, hidden)),
               ("output_norm.weight", np.abs(vec(hidden)) + 1.0)]
    expectations = {}
    for i in range(n_layers):
        p = f"blk.{i}."
        q_nat, k_nat = mat(n_heads * head_dim, hidden), mat(n_kv * head_dim, hidden)
        v_i, o_i = mat(n_kv * head_dim, hidden), mat(hidden, n_heads * head_dim)
        tensors += [
            (p + "attn_q.weight", oracle.oracle_llama_permute_qk(q_nat, n_heads, head_dim)),
            (p + "attn_k.weight", oracle.oracle_llama_permute_qk(k_nat, n_kv, head_dim)),
            (p + "attn_v.weight", v_i),
            (p + "attn_output.weight", o_i),
            (p + "attn_norm.weight", np.abs(vec(hidden)) + 1.0),
            (p + "ffn_norm.weight", np.abs(vec(hidden)) + 1.0),
            (p + "ffn_gate.weight", mat(inter, hidden)),
            (p + "ffn_up.weight", mat(inter, hidden)),
            (p + "ffn_down.weight", mat(hidden, inter)),
        ]
        expectations[f"model.layers.{i}.self_attn.q_proj.weight"] = q_nat
        expectations[f"model.layers.{i}.self_attn.k_proj.weight"] = k_nat

    gguf_path = write_gguf(
        tmp_path / "llama.gguf", arch="llama",
        metadata={
            "embedding_length": hidden, "block_count": n_layers,
            "attention.head_count": n_heads, "attention.head_count_kv": n_kv,
            "attention.key_length": head_dim,
            "attention.layer_norm_rms_epsilon": 1e-5,
            "feed_forward_length": inter, "context_length": 512,
            "rope.freq_base": 1e6,
        },
        f32_tensors=tensors,
    )

    cfg = load_arch_config(str(LLAMA_YAML))
    source = GGUFSource(str(gguf_path))
    plan = plan_conversion(cfg, source)
    out = tmp_path / "out"
    runner = ConversionRunner(plan, source, str(out),
                              quant=QuantSettings(bits=None),
                              tokenizer_source=write_minimal_tokenizer(tmp_path / "tokenizer"),
                              log=lambda _: None)
    runner.run()
    tensors_out = {}
    for f in out.glob("*.safetensors"):
        tensors_out.update(load_file(str(f)))

    for name, expected in expectations.items():
        got = tensors_out[name]
        assert got.shape == expected.shape, name
        assert not oracle.oracle_has_nan_or_inf(got), name
        assert oracle.oracle_close(got, expected, tol=5e-3), name

    # v/o pass through untouched (no permutation; f16 write rounding tolerated)
    for i in range(n_layers):
        src = {n: a for n, a in tensors}
        for gname, dname in [
            (f"blk.{i}.attn_v.weight", f"model.layers.{i}.self_attn.v_proj.weight"),
            (f"blk.{i}.attn_output.weight", f"model.layers.{i}.self_attn.o_proj.weight"),
        ]:
            assert oracle.oracle_close(tensors_out[dname], src[gname], tol=1e-3), dname