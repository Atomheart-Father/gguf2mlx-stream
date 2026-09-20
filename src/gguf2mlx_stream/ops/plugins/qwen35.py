"""Qwen3.5 (hybrid GDN + full attention) plugin operators.

The single architecture-specific transform this family needs is the inverse
of llama.cpp's GDN v-head storage permutation. llama.cpp stores v-head
tensors as::

    gguf = concat(hf[0::2], hf[1::2])

over the value-head axis (per-head blocks). MLX (like HF) uses the natural
head order, so conversion must apply the inverse permutation — the plugin
resolves the head count / block size from the architecture's declared dims
and delegates to the generic ``unzip_blocks`` primitive.

Affected tensors (see ``configs/qwen3_5.yaml``): ``in_proj_qkv`` (v rows),
``in_proj_z`` (rows), ``in_proj_a``/``in_proj_b`` (rows, block=1),
``out_proj`` (cols), ``conv1d`` (v rows), ``dt_bias`` (block=1).
"""

from __future__ import annotations

import numpy as np

from ..base import OpContext
from ..generic import op_unzip_blocks
from ..registry import register_op


@register_op(
    "qwen35_v_head_unpermute",
    kind="permute",
    summary=(
        "Undo the llama.cpp GDN v-head permutation on one axis. Heads/block "
        "come from the architecture dims (``linear_num_value_heads`` / "
        "``linear_value_head_dim``) unless overridden."
    ),
    params=(
        ("axis", "axis holding the v-heads (0 = rows, 1 = cols); default 0"),
        ("heads", "dim name or int: number of v-heads; default linear_num_value_heads"),
        ("block", "dim name or int: per-head block size; default linear_value_head_dim"),
    ),
)
def qwen35_v_head_unpermute(inputs, order, args, ctx: OpContext) -> np.ndarray:
    x = inputs[order[0]]
    heads = _resolve(args.get("heads", "linear_num_value_heads"), ctx)
    block = _resolve(args.get("block", "linear_value_head_dim"), ctx)
    axis = int(args.get("axis", 0))
    if block == 1:
        # one scalar per head (e.g. dt_bias, in_proj_a/b row vectors)
        if x.shape[axis] != heads:
            raise ValueError(
                f"qwen35_v_head_unpermute: axis length {x.shape[axis]} != heads {heads}"
            )
    return op_unzip_blocks({"x": x}, ["x"], {"axis": axis, "block": block}, ctx)


def _resolve(value, ctx: OpContext) -> int:
    if isinstance(value, int):
        return value
    return ctx.dim(str(value))
