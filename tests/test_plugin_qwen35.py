"""Qwen3.5 v-head plugin tests: round-trip, block=1 vectors, dim resolution."""

import numpy as np
import pytest

from gguf2mlx_stream.ops import get_op
from gguf2mlx_stream.ops.base import OpContext

SPEC = get_op("qwen35_v_head_unpermute")

# 4 v-heads in 2 kv-groups -> ratio 2 (the Qwen3.5-9B geometry)
DIMS = {
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_value_head_dim": 2,
}


def run(x, args=None, dims=DIMS):
    ctx = OpContext(dims=dims)
    return SPEC.fn({"x": np.asarray(x, np.float32)}, ["x"], dict(args or {}), ctx)


def zip_ref(x, axis, block, ratio=2):
    """GGUF storage: concat(natural[j::ratio] for j in range(ratio)) over blocks."""
    moved = np.moveaxis(x, axis, 0)
    n = moved.shape[0] // block
    v = moved.reshape(n, block, *moved.shape[1:])
    out = np.concatenate([v[j::ratio] for j in range(ratio)], axis=0).reshape(moved.shape)
    return np.moveaxis(out, 0, axis)


def test_unpermute_rows_2d():
    # 8 rows = 4 v-heads x 2 per-head rows; gguf = zip of natural order
    natural = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
    gguf = zip_ref(natural, 0, 2)
    out = run(gguf, {"axis": 0})
    np.testing.assert_array_equal(out, natural)


def test_unpermute_cols_2d():
    natural = np.arange(5 * 8, dtype=np.float32).reshape(5, 8)
    gguf = zip_ref(natural, 1, 2)
    out = run(gguf, {"axis": 1})
    np.testing.assert_array_equal(out, natural)


def test_unpermute_vector_block1():
    # one scalar per head (32 heads in the real model, 4 here)
    natural = np.array([10.0, 20.0, 30.0, 40.0], np.float32)
    gguf = zip_ref(natural, 0, 1)
    assert gguf.tolist() == [10.0, 30.0, 20.0, 40.0]
    out = run(gguf, {"axis": 0, "block": 1})
    np.testing.assert_array_equal(out, natural)


def test_default_dims_come_from_context():
    natural = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)
    gguf = zip_ref(natural, 0, 2)  # block defaults to linear_value_head_dim=2
    np.testing.assert_array_equal(run(gguf, {"axis": 0}), natural)


def test_explicit_int_overrides_ratio1():
    # heads=6, kv groups=6 -> ratio 1 -> identity
    natural = np.arange(6, dtype=np.float32)
    out = run(natural, {"axis": 0, "heads": 6, "k_heads": 6, "block": 1})
    np.testing.assert_array_equal(out, natural)


def test_ratio3_from_dims():
    dims = {"linear_num_value_heads": 6, "linear_num_key_heads": 2, "linear_value_head_dim": 1}
    natural = np.arange(6, dtype=np.float32)
    gguf = zip_ref(natural, 0, 1, ratio=3)
    out = run(gguf, {"axis": 0, "block": 1}, dims=dims)
    np.testing.assert_array_equal(out, natural)


def test_shape_mismatch_raises():
    bad = np.zeros((7, 3), np.float32)
    with pytest.raises(ValueError):
        run(bad, {"axis": 0})
    with pytest.raises(ValueError):
        run(np.zeros(5, np.float32), {"axis": 0, "block": 1})


def test_indivisible_heads_raise():
    with pytest.raises(ValueError):
        run(np.zeros((8, 2), np.float32), {"axis": 0},
            dims={"linear_num_value_heads": 5, "linear_num_key_heads": 2,
                  "linear_value_head_dim": 2})


def test_unknown_dim_raises():
    with pytest.raises(KeyError):
        run(np.zeros((4, 4), np.float32), {"axis": 0}, dims={})
