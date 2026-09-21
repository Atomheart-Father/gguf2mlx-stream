"""Generic tensor operators.

Every operator is a pure function over numpy arrays: inputs are never
mutated, outputs are freshly allocated (or contiguous copies). The uniform
operator signature is::

    fn(inputs: dict[str, np.ndarray], order: list[str], args: dict,
       ctx: OpContext) -> np.ndarray

``order`` lists the input slot names in declared order (single-input ops use
``inputs[order[0]]``). ``args`` values are plain ints/strings — the planner
resolves any dimension references before execution, so operators contain no
config-language logic.

Semantics operate on HF-convention arrays: 2-D weights are ``(out, in)``,
"rows" = axis 0.
"""

from __future__ import annotations

import numpy as np

from .base import OpContext
from .registry import register_op

_DTYPES = {"float32": np.float32, "float16": np.float16}


def _x(inputs: dict, order: list) -> np.ndarray:
    if len(order) != 1:
        raise ValueError(f"expected exactly one input, got {order}")
    return inputs[order[0]]


def _f32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


@register_op(
    "copy",
    kind="elementwise",
    summary="Identity copy to contiguous float32.",
)
def op_copy(inputs, order, args, ctx: OpContext) -> np.ndarray:
    return _f32(_x(inputs, order))


@register_op(
    "cast",
    kind="elementwise",
    summary="Cast to a supported dtype (float32 or float16).",
    params=(("dtype", "target dtype: 'float32' or 'float16'"),),
)
def op_cast(inputs, order, args, ctx: OpContext) -> np.ndarray:
    dtype = _DTYPES.get(str(args.get("dtype", "float32")))
    if dtype is None:
        raise ValueError(f"cast: unsupported dtype {args.get('dtype')!r}")
    return _x(inputs, order).astype(dtype, copy=True)


@register_op(
    "reshape",
    kind="shape",
    summary="Reshape to ``shape``; one -1 entry is inferred.",
    params=(("shape", "target shape as a list of ints; -1 infers"),),
)
def op_reshape(inputs, order, args, ctx: OpContext) -> np.ndarray:
    shape = tuple(int(s) for s in args["shape"])
    return _x(inputs, order).reshape(shape)


@register_op(
    "unsqueeze",
    kind="shape",
    summary="Add a size-1 axis.",
    params=(("axis", "position of the new axis (numpy convention, negatives allowed)"),),
)
def op_unsqueeze(inputs, order, args, ctx: OpContext) -> np.ndarray:
    return np.expand_dims(_x(inputs, order), int(args["axis"]))


@register_op(
    "squeeze",
    kind="shape",
    summary="Remove size-1 axes (all of them, or one given axis).",
    params=(("axis", "optional axis to squeeze; omit to squeeze all size-1 axes"),),
)
def op_squeeze(inputs, order, args, ctx: OpContext) -> np.ndarray:
    axis = args.get("axis")
    x = _x(inputs, order)
    return np.squeeze(x, axis=int(axis)) if axis is not None else np.squeeze(x)


@register_op(
    "transpose",
    kind="shape",
    summary="Transpose axes (default: full reverse, e.g. (out,in) -> (in,out)).",
    params=(("axes", "optional permutation of axes"),),
)
def op_transpose(inputs, order, args, ctx: OpContext) -> np.ndarray:
    axes = args.get("axes")
    x = _x(inputs, order)
    return np.ascontiguousarray(x.transpose(tuple(int(a) for a in axes)) if axes else x.T)


@register_op(
    "permute",
    kind="shape",
    summary="Alias of ``transpose`` with a required explicit axes permutation.",
    params=(("axes", "permutation of axes"),),
)
def op_permute(inputs, order, args, ctx: OpContext) -> np.ndarray:
    if "axes" not in args:
        raise ValueError("permute: 'axes' is required")
    return np.ascontiguousarray(_x(inputs, order).transpose(tuple(int(a) for a in args["axes"])))


@register_op(
    "slice",
    kind="shape",
    summary="Slice one axis: x[..., start:stop:step, ...].",
    params=(
        ("axis", "axis to slice"),
        ("start", "inclusive start (default 0)"),
        ("stop", "exclusive stop (default: end)"),
        ("step", "optional step (default 1)"),
    ),
)
def op_slice(inputs, order, args, ctx: OpContext) -> np.ndarray:
    axis = int(args["axis"])
    start = int(args.get("start", 0))
    stop = args.get("stop")
    step = int(args.get("step", 1))
    x = _x(inputs, order)
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(start, None if stop is None else int(stop), step)
    return np.ascontiguousarray(x[tuple(sl)])


@register_op(
    "concat",
    kind="combine",
    summary="Concatenate 2+ named inputs along ``axis`` (in listed order).",
    params=(("axis", "axis along which to concatenate"),),
    inputs_doc="two or more named inputs, order matters",
)
def op_concat(inputs, order, args, ctx: OpContext) -> np.ndarray:
    if len(order) < 2:
        raise ValueError("concat: requires at least two inputs")
    return _f32(np.concatenate([_f32(inputs[n]) for n in order], axis=int(args.get("axis", 0))))


def _block_view(x: np.ndarray, axis: int, block: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """View x with ``axis`` split into (n_blocks, block), blocks moved to front."""
    n = x.shape[axis]
    if block <= 0 or n % block != 0:
        raise ValueError(f"axis length {n} not divisible by block {block}")
    moved = np.moveaxis(x, axis, 0)
    rest = moved.shape[1:]
    return moved.reshape(n // block, block, *rest), rest


def _unblock(v: np.ndarray, rest: tuple[int, ...], axis: int, out_shape: tuple[int, ...]) -> np.ndarray:
    n_blocks, block = v.shape[0], v.shape[1]
    moved = v.reshape(n_blocks * block, *rest)
    out = np.moveaxis(moved, 0, axis)
    if out.shape != out_shape:
        raise AssertionError(f"block op shape mismatch {out.shape} != {out_shape}")
    return out


@register_op(
    "zip_blocks",
    kind="permute",
    summary=(
        "Forward head-block interleave: output blocks are concat(b[0::2], b[1::2]) "
        "along ``axis`` (the llama.cpp GDN v-head layout)."
    ),
    params=(
        ("axis", "axis holding the blocks"),
        ("block", "block size along that axis"),
    ),
)
def op_zip_blocks(inputs, order, args, ctx: OpContext) -> np.ndarray:
    axis, block = int(args["axis"]), int(args["block"])
    x = _x(inputs, order)
    v, rest = _block_view(x, axis, block)
    n = v.shape[0]
    if n % 2 != 0:
        raise ValueError(f"odd block count {n} on axis {axis}")
    out = np.empty_like(v)
    out[: n // 2] = v[0::2]
    out[n // 2 :] = v[1::2]
    return _unblock(out, rest, axis, x.shape)


@register_op(
    "unzip_blocks",
    kind="permute",
    summary=(
        "Inverse of ``zip_blocks``: recovers natural order from "
        "concat(b[0::2], b[1::2]) block interleave."
    ),
    params=(
        ("axis", "axis holding the blocks"),
        ("block", "block size along that axis"),
    ),
)
def op_unzip_blocks(inputs, order, args, ctx: OpContext) -> np.ndarray:
    axis, block = int(args["axis"]), int(args["block"])
    x = _x(inputs, order)
    v, rest = _block_view(x, axis, block)
    n = v.shape[0]
    if n % 2 != 0:
        raise ValueError(f"odd block count {n} on axis {axis}")
    out = np.empty_like(v)
    out[0::2] = v[: n // 2]
    out[1::2] = v[n // 2 :]
    return _unblock(out, rest, axis, x.shape)


@register_op(
    "reorder_grouped_heads",
    kind="permute",
    summary=(
        "Undo group-interleaved head storage for ``heads/heads_kv = ratio`` groups: "
        "storage order is concat(natural[j::ratio] for j in range(ratio)); restores "
        "natural order. ratio=1 is identity, ratio=2 equals ``unzip_blocks``."
    ),
    params=(
        ("axis", "axis holding the head blocks"),
        ("block", "block size along that axis (per-head rows/cols; 1 for vectors)"),
        ("ratio", "value heads per key-value group (>= 1); default 2"),
    ),
)
def op_reorder_grouped_heads(inputs, order, args, ctx: OpContext) -> np.ndarray:
    axis, block = int(args["axis"]), int(args["block"])
    ratio = int(args.get("ratio", 2))
    if ratio < 1:
        raise ValueError(f"reorder_grouped_heads: ratio must be >= 1, got {ratio}")
    x = _x(inputs, order)
    v, rest = _block_view(x, axis, block)
    n = v.shape[0]  # total head blocks (value heads)
    if n % ratio != 0:
        raise ValueError(
            f"reorder_grouped_heads: {n} head blocks not divisible by ratio {ratio}"
        )
    if ratio == 1:
        return np.ascontiguousarray(x)
    k = n // ratio  # key-value head groups
    idx = np.arange(n)
    gguf_index = (idx % ratio) * k + idx // ratio
    out = np.ascontiguousarray(v.take(gguf_index, axis=0))
    return _unblock(out, rest, axis, x.shape)


def _unary(name: str, fn):
    def run(inputs, order, args, ctx: OpContext) -> np.ndarray:
        return fn(np.asarray(_x(inputs, order), dtype=np.float32))

    register_op(name, kind="elementwise", summary=f"Elementwise {name} (float32).")(run)


_unary("neg", np.negative)
_unary("log", np.log)
_unary("exp", np.exp)


def _binary(name: str, fn):
    def run(inputs, order, args, ctx: OpContext) -> np.ndarray:
        if len(order) == 2:
            a, b = _f32(inputs[order[0]]), _f32(inputs[order[1]])
        elif len(order) == 1 and "value" in args:
            a, b = _f32(inputs[order[0]]), np.float32(args["value"])
        else:
            raise ValueError(f"{name}: needs inputs [a, b] or one input plus args.value")
        return fn(a, b)

    register_op(
        name,
        kind="combine",
        summary=f"Elementwise {name} of two inputs (or one input and args.value).",
        inputs_doc="[a, b] or [a] with args.value",
    )(run)


_binary("add", np.add)
_binary("sub", np.subtract)
_binary("mul", np.multiply)
_binary("div", np.divide)
