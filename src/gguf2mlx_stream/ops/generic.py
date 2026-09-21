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

Every operator also registers a plan-time *shape inference* function with
the same conventions but operating on shape tuples::

    infer_shape(shapes: dict[str, tuple[int, ...]], order: list[str],
                args: dict, ctx: OpContext) -> tuple[int, ...]

Inference functions raise ``ValueError`` on invalid parameters or
incompatible shapes; the planner turns that into a ``PlanError`` before any
tensor data is read. Runtime functions remain the source of truth for
values; inference only mirrors shape semantics.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .base import OpContext
from .registry import register_op

_DTYPE_NAMES = {"float32": np.float32, "float16": np.float16}


def _x(inputs: dict, order: list) -> np.ndarray:
    if len(order) != 1:
        raise ValueError(f"expected exactly one input, got {order}")
    return inputs[order[0]]


def _f32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# shape-inference helpers
# ---------------------------------------------------------------------------

def _one_shape(shapes: Mapping[str, tuple], order: list) -> tuple[int, ...]:
    if len(order) != 1:
        raise ValueError(f"expected exactly one input, got {order}")
    return tuple(int(d) for d in shapes[order[0]])


def _norm_axis(axis: int, ndim: int, where: str) -> int:
    a = int(axis)
    if not (-ndim <= a < ndim):
        raise ValueError(f"{where}: axis {axis} out of range for rank {ndim}")
    return a


def _check_axes_permutation(axes, ndim: int) -> tuple[int, ...]:
    ax = tuple(int(a) for a in axes)
    if sorted(ax) != list(range(ndim)):
        raise ValueError(
            f"axes {list(ax)} are not a permutation of range({ndim})"
        )
    return ax


def _identity_infer(shapes, order, args, ctx: OpContext) -> tuple[int, ...]:
    return _one_shape(shapes, order)


@register_op(
    "copy",
    kind="elementwise",
    summary="Identity copy to contiguous float32.",
    infer_shape=_identity_infer,
)
def op_copy(inputs, order, args, ctx: OpContext) -> np.ndarray:
    return _f32(_x(inputs, order))


@register_op(
    "cast",
    kind="elementwise",
    summary="Cast to a supported dtype (float32 or float16).",
    params=(("dtype", "target dtype: 'float32' or 'float16'"),),
    infer_shape=_identity_infer,
)
def op_cast(inputs, order, args, ctx: OpContext) -> np.ndarray:
    dtype = _DTYPE_NAMES.get(str(args.get("dtype", "float32")))
    if dtype is None:
        raise ValueError(f"cast: unsupported dtype {args.get('dtype')!r}")
    return _x(inputs, order).astype(dtype, copy=True)


def _infer_reshape(shapes, order, args, ctx: OpContext) -> tuple[int, ...]:
    if "shape" not in args:
        raise ValueError("reshape: 'shape' is required")
    shape = tuple(int(s) for s in args["shape"])
    x = _one_shape(shapes, order)
    if any(d < -1 for d in shape):
        raise ValueError(f"reshape: invalid target shape {list(shape)}")
    if shape.count(-1) > 1:
        raise ValueError("reshape: at most one -1 entry is allowed")
    known = math.prod(d for d in shape if d != -1)
    total = math.prod(x)
    if known == 0 or total % known != 0:
        raise ValueError(
            f"reshape: cannot reshape {x} (size {total}) into {list(shape)}"
        )
    return tuple(total // known if d == -1 else d for d in shape)


@register_op(
    "reshape",
    kind="shape",
    summary="Reshape to ``shape``; one -1 entry is inferred.",
    params=(("shape", "target shape as a list of ints; -1 infers"),),
    infer_shape=_infer_reshape,
)
def op_reshape(inputs, order, args, ctx: OpContext) -> np.ndarray:
    shape = tuple(int(s) for s in args["shape"])
    return _x(inputs, order).reshape(shape)


def _infer_unsqueeze(shapes, order, args, ctx: OpContext) -> tuple[int, ...]:
    if "axis" not in args:
        raise ValueError("unsqueeze: 'axis' is required")
    x = _one_shape(shapes, order)
    a = int(args["axis"])
    # np.expand_dims convention: [-ndim - 1, ndim]
    if not (-len(x) - 1 <= a <= len(x)):
        raise ValueError(f"unsqueeze: axis {a} out of range for rank {len(x)}")
    a = a if a >= 0 else a + len(x) + 1
    return x[:a] + (1,) + x[a:]


@register_op(
    "unsqueeze",
    kind="shape",
    summary="Add a size-1 axis.",
    params=(("axis", "position of the new axis (numpy convention, negatives allowed)"),),
    infer_shape=_infer_unsqueeze,
)
def op_unsqueeze(inputs, order, args, ctx: OpContext) -> np.ndarray:
    return np.expand_dims(_x(inputs, order), int(args["axis"]))


def _infer_squeeze(shapes, order, args, ctx: OpContext) -> tuple[int, ...]:
    x = _one_shape(shapes, order)
    axis = args.get("axis")
    if axis is None:
        return tuple(d for d in x if d != 1)
    a = _norm_axis(axis, len(x), "squeeze")
    if x[a] != 1:
        raise ValueError(f"squeeze: cannot squeeze axis {axis} of size {x[a]}")
    return x[:a] + x[a + 1:]


@register_op(
    "squeeze",
    kind="shape",
    summary="Remove size-1 axes (all of them, or one given axis).",
    params=(("axis", "optional axis to squeeze; omit to squeeze all size-1 axes"),),
    infer_shape=_infer_squeeze,
)
def op_squeeze(inputs, order, args, ctx: OpContext) -> np.ndarray:
    axis = args.get("axis")
    x = _x(inputs, order)
    return np.squeeze(x, axis=int(axis)) if axis is not None else np.squeeze(x)


def _infer_transpose(shapes, order, args, ctx: OpContext) -> tuple[int, ...]:
    x = _one_shape(shapes, order)
    axes = args.get("axes")
    if axes is None:
        return tuple(reversed(x))
    ax = _check_axes_permutation(axes, len(x))
    return tuple(x[a] for a in ax)


@register_op(
    "transpose",
    kind="shape",
    summary="Transpose axes (default: full reverse, e.g. (out,in) -> (in,out)).",
    params=(("axes", "optional permutation of axes"),),
    infer_shape=_infer_transpose,
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
    infer_shape=_infer_transpose,
)
def op_permute(inputs, order, args, ctx: OpContext) -> np.ndarray:
    if "axes" not in args:
        raise ValueError("permute: 'axes' is required")
    return np.ascontiguousarray(_x(inputs, order).transpose(tuple(int(a) for a in args["axes"])))


def _infer_slice(shapes, order, args, ctx: OpContext) -> tuple[int, ...]:
    if "axis" not in args:
        raise ValueError("slice: 'axis' is required")
    x = _one_shape(shapes, order)
    a = _norm_axis(args["axis"], len(x), "slice")
    start = int(args.get("start", 0))
    stop = args.get("stop")
    step = int(args.get("step", 1))
    if step == 0:
        raise ValueError("slice: step must not be 0")
    lo, hi, st = slice(start, None if stop is None else int(stop), step).indices(x[a])
    n = len(range(lo, hi, st))
    return x[:a] + (n,) + x[a + 1:]


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
    infer_shape=_infer_slice,
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


def _infer_concat(shapes, order, args, ctx: OpContext) -> tuple[int, ...]:
    if len(order) < 2:
        raise ValueError("concat: requires at least two inputs")
    if "axis" not in args:
        raise ValueError("concat: 'axis' is required")
    axis = int(args["axis"])
    first = tuple(int(d) for d in shapes[order[0]])
    a = _norm_axis(axis, len(first), "concat")
    total = 0
    for name in order:
        s = tuple(int(d) for d in shapes[name])
        if len(s) != len(first):
            raise ValueError(
                f"concat: input {name!r} has rank {len(s)}, expected {len(first)}"
            )
        for i in range(len(first)):
            if i != a and s[i] != first[i]:
                raise ValueError(
                    f"concat: input {name!r} shape {list(s)} incompatible with "
                    f"{list(first)} along non-concat axis {a}"
                )
        total += s[a]
    return first[:a] + (total,) + first[a + 1:]


@register_op(
    "concat",
    kind="combine",
    summary="Concatenate 2+ named inputs along ``axis`` (in listed order).",
    params=(("axis", "axis along which to concatenate"),),
    inputs_doc="two or more named inputs, order matters",
    infer_shape=_infer_concat,
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


def _infer_block_identity(check_even: bool, ratio: int | None):
    """Shared inference for the block-permutation ops (identity shapes)."""
    def infer(shapes, order, args, ctx: OpContext) -> tuple[int, ...]:
        x = _one_shape(shapes, order)
        if "axis" not in args or "block" not in args:
            raise ValueError("block ops require 'axis' and 'block' args")
        axis = int(args["axis"])
        block = int(args["block"])
        a = _norm_axis(axis, len(x), "block op")
        if block <= 0 or x[a] % block != 0:
            raise ValueError(f"axis length {x[a]} not divisible by block {block}")
        n = x[a] // block
        if check_even:
            if n % 2 != 0:
                raise ValueError(f"odd block count {n} on axis {axis}")
        elif ratio is not None:
            r = int(args.get("ratio", ratio))
            if r < 1:
                raise ValueError(f"ratio must be >= 1, got {r}")
            if n % r != 0:
                raise ValueError(f"{n} head blocks not divisible by ratio {r}")
        return x
    return infer


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
    infer_shape=_infer_block_identity(check_even=True, ratio=None),
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
    infer_shape=_infer_block_identity(check_even=True, ratio=None),
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
    infer_shape=_infer_block_identity(check_even=False, ratio=2),
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

    register_op(
        name, kind="elementwise", summary=f"Elementwise {name} (float32).",
        infer_shape=_identity_infer,
    )(run)


_unary("neg", np.negative)
_unary("log", np.log)
_unary("exp", np.exp)


def _infer_binary(shapes, order, args, ctx: OpContext) -> tuple[int, ...]:
    if len(order) == 2:
        return tuple(int(d) for d in np.broadcast_shapes(
            tuple(int(d) for d in shapes[order[0]]),
            tuple(int(d) for d in shapes[order[1]]),
        ))
    if len(order) == 1 and "value" in args:
        return _one_shape(shapes, order)
    raise ValueError("binary op: needs inputs [a, b] or one input plus args.value")


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
        infer_shape=_infer_binary,
    )(run)


_binary("add", np.add)
_binary("sub", np.subtract)
_binary("mul", np.multiply)
_binary("div", np.divide)
