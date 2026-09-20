"""Operator package: registry plus builtin generic operators and plugins.

Importing this package registers every built-in operator, including
architecture plugin operators (which live in
``gguf2mlx_stream.ops.plugins`` and are ordinary registered operators —
never dynamically imported from config files).
"""

from .registry import OpSpec, all_ops, get_op, register_op

# Builtin generic operators.
from . import generic as _generic  # noqa: F401  (registers ops)

# Builtin plugin operators.
from . import plugins as _plugins  # noqa: F401  (registers plugins)

__all__ = ["OpSpec", "all_ops", "get_op", "register_op"]
