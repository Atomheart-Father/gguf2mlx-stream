"""GGUF source layer tests: real Q4_K/Q6_K block bytes with known values."""

import numpy as np
import pytest
from conftest import (
    QK_K,
    build_q4_block,
    build_q6_block,
    q4_expected,
    q6_expected,
    write_gguf,
)
from gguf.constants import GGMLQuantizationType

from gguf2mlx_stream.errors import SourceError
from gguf2mlx_stream.source.gguf import GGUFSource


def _make_q6_matrix(n_rows: int) -> tuple[bytes, np.ndarray, tuple[int, ...], float]:
    """Two rows of Q6_K with distinct, analytic values."""
    d = 0.25
    scales = np.array([1, 2, 3, -1, 0, 5, -2, 7, 8, -3, 10, -4, 13, 14, -5, 16], dtype=np.int8)
    q6 = (np.arange(QK_K) * 7 + 3) % 64
    rows = []
    for r in range(n_rows):
        block = build_q6_block((q6 + r) % 64, scales, d)
        rows.append(np.frombuffer(block, np.uint8))
    raw = np.concatenate(rows).tobytes()
    expected = np.stack([q6_expected((q6 + r) % 64, scales, d) for r in range(n_rows)])
    return raw, expected, (QK_K, n_rows), d


def _make_q4_matrix(n_rows: int):
    d, dmin = 0.125, 0.0625
    sc = [1, 2, 3, 4, 5, 6, 7, 8]
    m = [0, 1, 2, 3, 4, 5, 6, 7]
    qs_base = (np.arange(QK_K) * 11 + 5) % 16
    rows, expected_rows = [], []
    for r in range(n_rows):
        qs = (qs_base + 3 * r) % 16
        rows.append(np.frombuffer(build_q4_block(qs, sc, m, d, dmin), np.uint8))
        expected_rows.append(q4_expected(qs, sc, m, d, dmin))
    raw = np.concatenate(rows).tobytes()
    return raw, np.stack(expected_rows), (QK_K, n_rows), (d, dmin, sc, m)


def test_q6_k_full_and_rows(tmp_path):
    raw, expected, ne, _ = _make_q6_matrix(4)
    path = write_gguf(
        tmp_path / "t.gguf",
        arch="testarch",
        metadata={"embedding_length": QK_K},
        quant_tensors=[("w6", raw, ne, GGMLQuantizationType.Q6_K)],
    )
    src = GGUFSource(path)
    t = src.info("w6")
    assert t.qtype == GGMLQuantizationType.Q6_K
    assert t.hf_shape == (4, 256)  # (out=rows, in=elements)
    mat = src.read_matrix("w6")
    assert mat.shape == (4, 256)
    np.testing.assert_allclose(mat, expected, rtol=0, atol=1e-4)
    # bounded row reads match the full matrix
    for lo, hi in ((0, 1), (1, 3), (3, 4), (0, 4)):
        rows = src.read_rows("w6", lo, hi)
        np.testing.assert_allclose(rows, expected[lo:hi], rtol=0, atol=1e-4)


def test_q4_k_full_and_rows(tmp_path):
    raw, expected, ne, _ = _make_q4_matrix(3)
    path = write_gguf(
        tmp_path / "t.gguf",
        arch="testarch",
        metadata={},
        quant_tensors=[("w4", raw, ne, GGMLQuantizationType.Q4_K)],
    )
    src = GGUFSource(path)
    mat = src.read_matrix("w4")
    np.testing.assert_allclose(mat, expected, rtol=0, atol=1e-4)
    rows = src.read_rows("w4", 1, 3)
    np.testing.assert_allclose(rows, expected[1:3], rtol=0, atol=1e-4)


def test_f32_vector_and_metadata(tmp_path):
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mat = np.arange(6, dtype=np.float32).reshape(2, 3)  # HF (out=2, in=3)
    path = write_gguf(
        tmp_path / "t.gguf",
        arch="testarch",
        metadata={"embedding_length": 3, "general.name": "tiny", "testarch.block_count": 7},
        f32_tensors=[("v", vec), ("m", mat)],
    )
    src = GGUFSource(path)
    np.testing.assert_allclose(src.read_vector("v"), vec)
    m = src.read_matrix("m")
    assert m.shape == (2, 3)
    np.testing.assert_allclose(m, mat)
    assert src.metadata_value("embedding_length") == 3
    assert src.metadata_value("block_count") == 7  # bare name finds {arch}.-prefixed key
    assert src.metadata_value("missing") is None
    assert src.arch == "testarch"


def test_missing_tensor_raises(tmp_path):
    path = write_gguf(tmp_path / "t.gguf", arch="a", metadata={})
    src = GGUFSource(path)
    with pytest.raises(SourceError):
        src.info("nope")
    with pytest.raises(SourceError):
        GGUFSource(tmp_path / "missing.gguf")


def test_bad_row_range(tmp_path):
    raw, _, ne, _ = _make_q6_matrix(2)
    path = write_gguf(
        tmp_path / "t.gguf", arch="a", metadata={},
        quant_tensors=[("w", raw, ne, GGMLQuantizationType.Q6_K)],
    )
    src = GGUFSource(path)
    with pytest.raises(SourceError):
        src.read_rows("w", 0, 5)
