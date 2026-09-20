"""Builtin plugin operators, one module per architecture family.

Plugins are imported (and therefore registered) at package import time.
They must stay small, deterministic, tested, and referenceable from YAML
configs by their registered name only.
"""

from . import qwen35 as _qwen35  # noqa: F401
