"""gguf2mlx-stream: declarative, bounded-memory GGUF -> MLX-LM streaming transcoder.

Core pipeline:

    GGUF tensor (quantized bytes, mmap'd)
        -> bounded dequantization (tensor or row-chunk)
        -> declarative transform plan (generic + plugin operators)
        -> MLX affine quantization
        -> sharded safetensors + standard MLX-LM config
        -> release

Architecture knowledge lives in declarative YAML configs under ``configs/``;
the engine itself is model-agnostic.
"""

from .errors import (
    ConfigError,
    ConversionError,
    Gguf2MlxError,
    PlanError,
    SourceError,
    VerifyError,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ConfigError",
    "ConversionError",
    "Gguf2MlxError",
    "PlanError",
    "SourceError",
    "VerifyError",
]
