"""Runtime context handed to operators.

Operators never touch global state: everything they may need (resolved
architecture dimensions, the source tensor description being processed) is
passed explicitly through :class:`OpContext`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OpContext:
    dims: Mapping[str, int]
    tensor_name: str | None = None
    dest_name: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def dim(self, name: str) -> int:
        try:
            return int(self.dims[name])
        except KeyError:
            raise KeyError(
                f"dimension {name!r} is not defined for this architecture; "
                f"known dims: {sorted(self.dims)}"
            ) from None
