"""Independent oracle implementations for verifier-independence tests.

These functions are deliberately written from the GGUF layout specification
(NOT by calling gguf2mlx_stream operators). They exist so that a bug in a
production operator cannot hide behind a verifier that shares the same code.

Only numpy is used here. No gguf2mlx_stream imports.
"""

from __future__ import annotations


import numpy as np


# ---------------------------------------------------------------------------
# grouped v-head storage permutation
# ---------------------------------------------------------------------------


def oracle_unpermute_heads(x: np.ndarray, axis: int, block: int, ratio: int) -> np.ndarray:
    """Inverse of llama.cpp group-interleaved v-head storage.

    Storage order: concat(natural[j::ratio] for j in range(ratio)).
    Implemented with explicit index loops (intentionally different from the
    production operator's vectorized gather).
    """
    x = np.asarray(x)
    moved = np.moveaxis(x, axis, 0)
    n_blocks = moved.shape[0] // block
    assert n_blocks % ratio == 0, "head blocks must divide by ratio"
    n_groups = n_blocks // ratio  # kv-head groups
    blocks = [moved[i * block:(i + 1) * block] for i in range(n_blocks)]
    out_blocks = [None] * n_blocks
    for h in range(n_blocks):
        i, j = divmod(h, ratio)  # natural head h = i*ratio + j
        out_blocks[h] = blocks[j * n_groups + i]
    out = np.concatenate(out_blocks, axis=0)
    return np.moveaxis(out, 0, axis)


def oracle_permute_heads(x: np.ndarray, axis: int, block: int, ratio: int) -> np.ndarray:
    """Forward: natural -> llama.cpp group-interleaved storage order."""
    x = np.asarray(x)
    moved = np.moveaxis(x, axis, 0)
    n_blocks = moved.shape[0] // block
    n_groups = n_blocks // ratio
    blocks = [moved[i * block:(i + 1) * block] for i in range(n_blocks)]
    out_blocks = []
    for j in range(ratio):
        for i in range(n_groups):
            out_blocks.append(blocks[i * ratio + j])
    out = np.concatenate(out_blocks, axis=0)
    return np.moveaxis(out, 0, axis)


def oracle_a_log(ssm_a: np.ndarray, value_heads: int, key_heads: int) -> np.ndarray:
    """A_log = log(-unpermute(ssm_a)) for block-1 v-head vectors."""
    ratio = value_heads // key_heads
    natural = oracle_unpermute_heads(np.asarray(ssm_a, np.float32), 0, 1, ratio)
    return np.log(-natural).astype(np.float32)


# ---------------------------------------------------------------------------
# llama.cpp q/k out-axis storage permutation
# ---------------------------------------------------------------------------


def oracle_llama_permute_qk(natural: np.ndarray, groups: int, head_dim: int) -> np.ndarray:
    """Forward: natural -> llama.cpp convert-time storage order for q/k.

    Within each head block, natural row ``d + (head_dim//2)*c`` is stored at
    row ``2*d + c``. Explicit per-row index loops (no reshape/transpose).
    """
    x = np.asarray(natural, np.float32)
    half = head_dim // 2
    assert x.shape[0] == groups * head_dim
    out = np.empty_like(x)
    for g in range(groups):
        for d in range(half):
            for c in range(2):
                out[g * head_dim + 2 * d + c] = x[g * head_dim + d + half * c]
    return out


def oracle_llama_unpermute_qk(stored: np.ndarray, groups: int, head_dim: int) -> np.ndarray:
    """Inverse of :func:`oracle_llama_permute_qk` (storage -> natural)."""
    x = np.asarray(stored, np.float32)
    half = head_dim // 2
    assert x.shape[0] == groups * head_dim
    out = np.empty_like(x)
    for g in range(groups):
        for d in range(half):
            for c in range(2):
                out[g * head_dim + d + half * c] = x[g * head_dim + 2 * d + c]
    return out


# ---------------------------------------------------------------------------
# fused-tensor expectations
# ---------------------------------------------------------------------------


def oracle_in_proj_qkv(hidden: int, natural_q: np.ndarray, natural_k: np.ndarray,
                       natural_v: np.ndarray, key_heads: int, value_heads: int,
                       value_block: int) -> np.ndarray:
    """GGUF attn_qkv = [q; k; permuted v] rows; expected dest = [q; k; natural v]."""
    gguf_rows = np.concatenate(
        [natural_q, natural_k, oracle_permute_heads(natural_v, 0, value_block,
                                                    value_heads // key_heads)],
        axis=0)
    assert gguf_rows.shape[0] == natural_q.shape[0] + natural_k.shape[0] + natural_v.shape[0]
    del hidden
    return np.concatenate([natural_q, natural_k, natural_v], axis=0)


def oracle_q_gate_fusion(natural_q: np.ndarray, gate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full-attention q rows carry fused [q; gate]; dest keeps the fusion."""
    return np.concatenate([natural_q, gate], axis=0), np.concatenate([natural_q, gate], axis=0)


def oracle_conv1d(natural_conv: np.ndarray, key_dim: int, value_block: int,
                  value_heads: int, key_heads: int) -> np.ndarray:
    """Expected dest: natural-order rows with a trailing size-1 axis."""
    out = np.concatenate([
        natural_conv[:key_dim],
        oracle_unpermute_heads(natural_conv[key_dim:], 0, value_block,
                               value_heads // key_heads),
    ], axis=0)
    return out[..., None]


def oracle_expected_layer_keys(n_layers: int, full_attn_layers: set[int],
                               quantized_keys: bool, prefix: str = "") -> set[str]:
    """Independent statement of the destination key set for a hybrid model."""
    def key(base: str) -> str:
        return prefix + base

    def quant_triple(base: str) -> set[str]:
        if quantized_keys:
            stem = base[: -len(".weight")] if base.endswith(".weight") else base
            return {key(base), key(stem + ".scales"), key(stem + ".biases")}
        return {key(base)}

    keys: set[str] = set()
    keys |= quant_triple("model.embed_tokens.weight")
    keys |= {key("model.norm.weight")}
    for i in range(n_layers):
        p = f"model.layers.{i}."
        keys |= {key(p + "input_layernorm.weight"),
                 key(p + "post_attention_layernorm.weight")}
        if i in full_attn_layers:
            keys |= quant_triple(p + "self_attn.q_proj.weight")
            keys |= quant_triple(p + "self_attn.k_proj.weight")
            keys |= quant_triple(p + "self_attn.v_proj.weight")
            keys |= quant_triple(p + "self_attn.o_proj.weight")
            keys |= {key(p + "self_attn.q_norm.weight"),
                     key(p + "self_attn.k_norm.weight")}
        else:
            keys |= quant_triple(p + "linear_attn.in_proj_qkv.weight")
            keys |= quant_triple(p + "linear_attn.in_proj_z.weight")
            keys |= quant_triple(p + "linear_attn.in_proj_a.weight")
            keys |= quant_triple(p + "linear_attn.in_proj_b.weight")
            keys |= quant_triple(p + "linear_attn.out_proj.weight")
            keys |= {key(p + "linear_attn.conv1d.weight"),
                     key(p + "linear_attn.dt_bias"),
                     key(p + "linear_attn.A_log"),
                     key(p + "linear_attn.norm.weight")}
        keys |= quant_triple(p + "mlp.gate_proj.weight")
        keys |= quant_triple(p + "mlp.up_proj.weight")
        keys |= quant_triple(p + "mlp.down_proj.weight")
    return keys


def oracle_has_nan_or_inf(x: np.ndarray) -> bool:
    return bool(np.isnan(np.asarray(x, np.float32)).any()
                or np.isinf(np.asarray(x, np.float32)).any())


def oracle_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-5) -> bool:
    return bool(np.allclose(np.asarray(a, np.float32), np.asarray(b, np.float32),
                            atol=tol, rtol=0))
