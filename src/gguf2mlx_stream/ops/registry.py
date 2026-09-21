"""Operator registry.

Operators are named, deterministic, pure tensor functions. The registry is a
plain module-level dict populated exclusively through :func:`register_op`
calls in library code (builtin generic operators and built-in plugin
modules). Architecture YAML configs may only reference operator *names* that
exist in this registry — YAML can never import or execute arbitrary code.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from ..errors import ConfigError

# Kinds describe how the planner/runner may treat an operator:
#   elementwise: f(x) applied independently per element; row-axis chunk-safe.
#   shape:       shape manipulation (reshape/transpose/...), not chunk-safe in general.
#   combine:     consumes 2+ named inputs (concat/add/...).
#   permute:     block/head permutation along one axis; chunk-safe if the
#                chunk boundaries are multiples of the block span.
OP_KINDS = ("elementwise", "shape", "combine", "permute")


@dataclasses.dataclass(frozen=True)
class OpSpec:
    name: str
    fn: Callable
    kind: str
    summary: str
    params: tuple[tuple[str, str], ...] = ()  # (param name, description)
    inputs_doc: str = "x"  # description of expected named inputs
    # Optional plan-time shape inference:
    #   infer_shape(shapes, order, args, ctx) -> output shape tuple
    # where ``shapes`` maps input slot names to shape tuples. Operators that
    # provide it can be fully validated by the planner before any tensor is
    # read; the planner refuses to plan operators without it.
    infer_shape: Callable | None = None

    @property
    def streaming(self) -> bool:
        """Whether the op is safe to run inside arbitrary row-chunked jobs.

        Only elementwise ops are: shape ops and permutations need the whole
        tensor (or chunks aligned to their block span, which the planner does
        not assume).
        """
        return self.kind == "elementwise"


_REGISTRY: dict[str, OpSpec] = {}


def register_op(
    name: str,
    *,
    kind: str,
    summary: str,
    params: tuple[tuple[str, str], ...] = (),
    inputs_doc: str = "x",
    infer_shape: Callable | None = None,
) -> Callable:
    """Class/function decorator registering an operator under ``name``."""

    def _wrap(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"operator already registered: {name}")
        if kind not in OP_KINDS:
            raise ValueError(f"operator {name}: unknown kind {kind!r}")
        _REGISTRY[name] = OpSpec(
            name=name, fn=fn, kind=kind, summary=summary, params=tuple(params),
            inputs_doc=inputs_doc, infer_shape=infer_shape,
        )
        return fn

    return _wrap


def get_op(name: str) -> OpSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ConfigError(
            f"unknown operator {name!r}; see `gguf2mlx-stream list-ops` "
            "for registered operators"
        ) from None


def all_ops() -> dict[str, OpSpec]:
    return dict(_REGISTRY)
