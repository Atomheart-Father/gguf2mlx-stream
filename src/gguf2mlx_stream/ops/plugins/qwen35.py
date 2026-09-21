"""Qwen3.5 (hybrid GDN + full attention) plugin operators.

The single architecture-specific transform this family needs is the inverse
of llama.cpp's GDN v-head storage permutation. llama.cpp stores v-head
tensors in group-interleaved order::

    gguf = concat(hf[0::ratio], hf[1::ratio], ..., hf[(ratio-1)::ratio])

where ``ratio = linear_num_value_heads / linear_num_key_heads`` is the value
heads per key-value group. MLX (like HF) uses the natural head order, so
conversion applies the inverse permutation. This plugin is a thin semantic
wrapper: it resolves the head geometry from the architecture dims and
delegates the math to the generic ``reorder_grouped_heads`` primitive
(which is what actually moves data).

Affected tensors (see ``configs/qwen3_5.yaml``): ``in_proj_qkv`` (v rows),
``in_proj_z`` (rows), ``in_proj_a``/``in_proj_b`` (rows, block=1),
``out_proj`` (cols), ``conv1d`` (v rows), ``dt_bias`` (block=1).
"""

from __future__ import annotations

import numpy as np

from ..base import OpContext
from ..generic import op_reorder_grouped_heads
from ..registry import register_op


@register_op(
    "qwen35_v_head_unpermute",
    kind="permute",
    summary=(
        "Undo the llama.cpp GDN v-head permutation on one axis. Head geometry "
        "comes from the architecture dims (``linear_num_value_heads`` / "
        "``linear_num_key_heads`` / ``linear_value_head_dim``); the data "
        "movement is the generic ``reorder_grouped_heads`` operator."
    ),
    params=(
        ("axis", "axis holding the v-heads (0 = rows, 1 = cols); default 0"),
        ("heads", "dim name or int: number of v-heads; default linear_num_value_heads"),
        ("k_heads", "dim name or int: kv-head groups; default linear_num_key_heads"),
        ("block", "dim name or int: per-head block size; default linear_value_head_dim"),
    ),
)
def qwen35_v_head_unpermute(inputs, order, args, ctx: OpContext) -> np.ndarray:
    x = inputs[order[0]]
    heads = _resolve(args.get("heads", "linear_num_value_heads"), ctx)
    block = _resolve(args.get("block", "linear_value_head_dim"), ctx)
    axis = int(args.get("axis", 0))
    k_heads = args.get("k_heads", "linear_num_key_heads")
    k_heads = _resolve(k_heads, ctx) if k_heads is not None else heads
    if k_heads <= 0 or heads % k_heads != 0:
        raise ValueError(
            f"qwen35_v_head_unpermute: value heads {heads} must be divisible "
            f"by kv-head groups {k_heads}"
        )
    ratio = heads // k_heads
    if block == 1:
        # one scalar per head (e.g. dt_bias, in_proj_a/b row vectors)
        if x.shape[axis] != heads:
            raise ValueError(
                f"qwen35_v_head_unpermute: axis length {x.shape[axis]} != heads {heads}"
            )
    elif x.shape[axis] != heads * block:
        raise ValueError(
            f"qwen35_v_head_unpermute: axis length {x.shape[axis]} != "
            f"heads*block {heads * block}"
        )
    return op_reorder_grouped_heads(
        {"x": x}, ["x"], {"axis": axis, "block": block, "ratio": ratio}, ctx
    )


def _resolve(value, ctx: OpContext) -> int:
    if isinstance(value, int):
        return value
    return ctx.dim(str(value))
