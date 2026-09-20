"""Error hierarchy for gguf2mlx-stream.

All errors raised intentionally by the library derive from
:class:`Gguf2MlxError`, so callers can distinguish tool failures from bugs.
"""


class Gguf2MlxError(Exception):
    """Base class for all gguf2mlx-stream errors."""


class SourceError(Gguf2MlxError):
    """Raised when a GGUF source cannot be read or is malformed."""


class ConfigError(Gguf2MlxError):
    """Raised when an architecture config fails schema validation."""


class PlanError(Gguf2MlxError):
    """Raised when a config cannot be compiled into a valid conversion plan."""


class ConversionError(Gguf2MlxError):
    """Raised when streaming conversion fails at runtime."""


class VerifyError(Gguf2MlxError):
    """Raised when verification of an output directory fails."""
