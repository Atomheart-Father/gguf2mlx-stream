"""Shared numeric constants for quantization parameters.

Kept dependency-free so both the config schema (import-time validation)
and the quantization stage agree on what conversion-time settings accept.
"""

# Affine quantization bit widths accepted by MLX's ``quantize``/``dequantize``.
SUPPORTED_BITS = (2, 3, 4, 6, 8)
# Group sizes accepted by MLX (validated at conversion start, not deep inside
# the MLX kernel, so a bad config fails with a clear error).
SUPPORTED_GROUP_SIZES = (32, 64, 128)
