"""MLX affine quantization helpers.

Quantization is the terminal stage of each job, controlled by the rule
(``quantize: true``, with optional per-rule ``bits``/``group_size`` overrides)
plus conversion-level settings (bits, group size, mode). Weights are
dequantized to float32 by the pipeline, converted to float16 (the mlx
quantizer's input convention, matching mlx-lm), and quantized. Only
``packed`` (uint32), ``scales`` (float16) and ``biases`` (float16) arrays are
ever held in memory.

Quantization semantics are N-D: the **last axis is quantized in groups** and
every leading axis is preserved verbatim. A 2-D ``(out, in)`` matrix yields
packed/scales/biases of shape ``(out, ...)``; a 3-D expert tensor
``(n_experts, out, in)`` yields ``(n_experts, out, ...)``. Chunked execution
stays correct because quantization treats each leading-axis slice
independently: concatenating per-chunk results along axis 0 equals
quantizing the whole array.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from .constants import SUPPORTED_BITS, SUPPORTED_GROUP_SIZES
from .errors import ConversionError

SUPPORTED_MODES = ("affine",)


def quantize_weights(
    arr_f32: np.ndarray, bits: int, group_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize a float32 array along its last axis.

    Returns ``(packed uint32, scales f16, biases f16)`` whose leading
    dimensions match ``arr_f32.shape[:-1]``.
    """
    if bits not in SUPPORTED_BITS:
        raise ConversionError(f"unsupported quantization bits: {bits}")
    if group_size not in SUPPORTED_GROUP_SIZES:
        raise ConversionError(f"unsupported quantization group size: {group_size}")
    if arr_f32.ndim < 2:
        raise ConversionError(
            f"quantization expects a tensor of rank >= 2 (last axis quantized "
            f"in groups, leading axes preserved), got shape {arr_f32.shape}"
        )
    if arr_f32.shape[-1] % group_size != 0:
        raise ConversionError(
            f"inner dim {arr_f32.shape[-1]} not divisible by group size {group_size}"
        )
    w = mx.array(np.ascontiguousarray(arr_f32, dtype=np.float32)).astype(mx.float16)
    packed, scales, biases = mx.quantize(w, group_size, bits)
    out = (np.asarray(packed), np.asarray(scales), np.asarray(biases))
    del w, packed, scales, biases
    return out


def dequantize_weights(
    packed: np.ndarray, scales: np.ndarray, biases: np.ndarray, bits: int, group_size: int
) -> np.ndarray:
    """Inverse of :func:`quantize_weights` (used by the verifier)."""
    w = mx.dequantize(
        mx.array(packed), mx.array(scales), mx.array(biases), group_size, bits
    )
    return np.asarray(w)
