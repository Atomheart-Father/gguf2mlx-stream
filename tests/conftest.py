"""Shared fixtures: synthetic GGUF builders.

Quantized tensors are written as *raw bytes constructed from the GGML
k-quant block specification*, so the dequantization tests are not circular:
expected values are derived analytically from the chosen scale/quant codes,
not from calling the library under test.

Block layouts (GGML k-quants, QK_K = 256):

Q6_K (210 B/block): ql[128], qh[64], scales int8[16], d f16
    element f (0..255):  a = f // 128, r = f % 128
        low4  = (ql[a*64 + r % 64] >> (4 * (r // 64))) & 0xF
        high2 = (qh[a*32 + r % 32] >> (2 * (r // 32))) & 0x3
        q6 = low4 | (high2 << 4)          # 0..63
    value = d * scales[f // 16] * (q6 - 32)

Q4_K (144 B/block): d f16, dmin f16, scales uint8[12], qs[128]
    sub-block s (0..7, 32 elements each), element p (0..31):
        sc[0..3] =  scales[0:4]  & 0x3F
        m[0..3]  =  scales[4:8]  & 0x3F
        sc[4..7] = (scales[8:12] & 0x0F) | ((scales[0:4] >> 2) & 0x30)
        m[4..7]  = (scales[8:12] >> 4)    | ((scales[4:8] >> 2) & 0x30)
        q = (qs[(s // 2) * 32 + p] >> (4 * (s % 2))) & 0xF
    value = d * sc[s] * q - dmin * m[s]
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pytest
from gguf import GGUFWriter
from gguf.constants import GGMLQuantizationType

QK_K = 256


def f16_bytes(x: float) -> bytes:
    return np.float16(x).tobytes()


def build_q6_block(q6: np.ndarray, scales: np.ndarray, d: float) -> bytes:
    """Build one 210-byte Q6_K block from per-element 6-bit codes (0..63)."""
    q6 = np.asarray(q6, dtype=np.uint8)
    assert q6.shape == (QK_K,)
    ql = np.zeros(128, dtype=np.uint8)
    qh = np.zeros(64, dtype=np.uint8)
    for f in range(QK_K):
        a, r = f // 128, f % 128
        b, c = r // 64, r % 64
        ql[a * 64 + c] |= (int(q6[f]) & 0xF) << (4 * b)
        qh[a * 32 + (r % 32)] |= ((int(q6[f]) >> 4) & 0x3) << (2 * (r // 32))
    assert np.asarray(scales, dtype=np.int8).shape == (16,)
    return ql.tobytes() + qh.tobytes() + np.asarray(scales, np.int8).tobytes() + f16_bytes(d)


def q6_expected(q6: np.ndarray, scales: np.ndarray, d: float) -> np.ndarray:
    q6 = np.asarray(q6, np.int32)
    per_elem_scale = np.repeat(np.asarray(scales, np.float32), 16)
    return np.float16(d).astype(np.float32) * per_elem_scale * (q6 - 32).astype(np.float32)


def build_q4_block(qs: np.ndarray, sc: list[int], m: list[int], d: float, dmin: float) -> bytes:
    """Build one 144-byte Q4_K block from per-element 4-bit codes and scales."""
    qs = np.asarray(qs, dtype=np.uint8)
    assert qs.shape == (QK_K,)
    sub = qs.reshape(8, 32) & 0xF
    qbytes = np.zeros(128, dtype=np.uint8)
    for s in range(8):
        for p in range(32):
            qbytes[(s // 2) * 32 + p] |= int(sub[s, p]) << (4 * (s % 2))
    packed = np.zeros(12, dtype=np.uint8)
    for i in range(4):
        packed[i] = (sc[i] & 0x3F) | ((sc[4 + i] >> 4) & 0x3) << 6
        packed[4 + i] = (m[i] & 0x3F) | ((m[4 + i] >> 4) & 0x3) << 6
        packed[8 + i] = (sc[4 + i] & 0xF) | (m[4 + i] & 0xF) << 4
    return f16_bytes(d) + f16_bytes(dmin) + packed.tobytes() + qbytes.tobytes()


def q4_expected(qs: np.ndarray, sc: list[int], m: list[int], d: float, dmin: float) -> np.ndarray:
    sub = np.asarray(qs, np.int32).reshape(8, 32)
    out = np.empty(QK_K, np.float32)
    for s in range(8):
        out[s * 32:(s + 1) * 32] = (
            np.float16(d).astype(np.float32) * sc[s] * sub[s]
            - np.float16(dmin).astype(np.float32) * m[s]
        )
    return out


def _add_metadata(w: GGUFWriter, metadata: Mapping[str, object]) -> None:
    for key, value in metadata.items():
        if isinstance(value, str):
            w.add_string(key, value)
        elif isinstance(value, bool):
            w.add_bool(key, value)
        elif isinstance(value, int):
            w.add_uint32(key, value)
        elif isinstance(value, float):
            w.add_float32(key, value)
        elif isinstance(value, (list, tuple)):
            w.add_array(key, list(value))
        else:
            raise TypeError(f"unsupported metadata {key}={value!r}")


def write_gguf(
    path,
    arch: str,
    metadata: Mapping[str, object],
    f32_tensors: list[tuple[str, np.ndarray]] = (),
    quant_tensors: list[tuple[str, bytes, tuple[int, ...], GGMLQuantizationType]] = (),
) -> str:
    """Write a synthetic GGUF file.

    f32_tensors: (name, array) with HF-convention shapes; stored with
        GGUF ne = reversed(shape) (llama.cpp convention). Any rank works,
        including 3-D expert tensors.
    quant_tensors: (name, raw_bytes, ne, qtype) with ne in llama.cpp order
        (inner first); bytes are laid out row-major over ne[1:].
    """
    w = GGUFWriter(str(path), arch=arch)
    w.add_architecture()
    _add_metadata(w, metadata)
    for name, arr in f32_tensors:
        # HF-shaped array passed directly: gguf-py reverses the numpy shape
        # when writing ne, so a (out, in) array yields ne = (in, out) and the
        # C-order bytes match GGUF's row-major-over-ne layout.
        w.add_tensor(name, np.ascontiguousarray(np.asarray(arr, np.float32)))
    for name, raw, ne, qtype in quant_tensors:
        # The writer reverses ti.shape into file ne, so pass the reversed ne
        # as raw_shape; an int8 view skips gguf-py's byte-shape
        # reinterpretation and tofile writes the flat bytes unchanged.
        w.add_tensor(name, np.frombuffer(raw, np.uint8).view(np.int8),
                     raw_shape=tuple(reversed(ne)), raw_dtype=qtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


@pytest.fixture(scope="session")
def venv_python() -> str:
    import sys

    return sys.executable


def write_minimal_tokenizer(directory) -> str:
    """Create a directory satisfying the tokenizer output contract.

    The conversion-time contract check validates file presence, non-emptiness
    and JSON parseability — not that a full tokenizer works (that is the job
    of the mlx_lm.load()-based integration stages).
    """
    import json
    import os

    os.makedirs(directory, exist_ok=True)
    tok = {
        "version": "1.0",
        "model": {"vocab": {"<unk>": 0, "a": 1, "b": 2}, "merges": []},
    }
    with open(os.path.join(directory, "tokenizer.json"), "w") as f:
        json.dump(tok, f)
    with open(os.path.join(directory, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast"}, f)
    return str(directory)
