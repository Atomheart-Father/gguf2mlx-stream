"""Generic operator unit tests (small synthetic tensors)."""

import numpy as np
import pytest

from gguf2mlx_stream.errors import ConfigError
from gguf2mlx_stream.ops import all_ops, get_op
from gguf2mlx_stream.ops.base import OpContext

CTX = OpContext(dims={})


def run(name, x, args=None):
    spec = get_op(name)
    return spec.fn({"x": np.asarray(x)}, ["x"], dict(args or {}), CTX)


def run2(name, a, b, args=None):
    spec = get_op(name)
    return spec.fn({"a": np.asarray(a), "b": np.asarray(b)}, ["a", "b"], dict(args or {}), CTX)


def test_registry_has_expected_ops():
    expected = {
        "copy", "cast", "reshape", "unsqueeze", "squeeze", "transpose", "permute",
        "slice", "concat", "zip_blocks", "unzip_blocks",
        "neg", "log", "exp", "add", "sub", "mul", "div",
        "qwen35_v_head_unpermute",
    }
    assert expected <= set(all_ops())


def test_unknown_op_raises():
    with pytest.raises(ConfigError):
        get_op("does_not_exist")


def test_copy_is_contiguous_f32():
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    out = run("copy", x)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out, x)


def test_cast():
    x = np.ones((2, 2), dtype=np.float32)
    assert run("cast", x, {"dtype": "float16"}).dtype == np.float16
    with pytest.raises(ValueError):
        run("cast", x, {"dtype": "bfloat16"})


def test_reshape_unsqueeze_squeeze():
    x = np.arange(24, dtype=np.float32).reshape(4, 6)
    assert run("reshape", x, {"shape": [2, -1]}).shape == (2, 12)
    assert run("unsqueeze", x, {"axis": -1}).shape == (4, 6, 1)
    y = run("unsqueeze", x, {"axis": 0})
    assert run("squeeze", y).shape == (4, 6)
    with pytest.raises(ValueError):  # cannot squeeze an axis with size > 1
        run("squeeze", x, {"axis": 0})
    with pytest.raises(ValueError):
        run("reshape", x, {"shape": [5, -1]})


def test_transpose_permute_slice():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(run("transpose", x), x.T)
    np.testing.assert_array_equal(run("permute", x, {"axes": [1, 0]}), x.T)
    with pytest.raises(ValueError):
        run("permute", x, {})
    s = run("slice", x, {"axis": 0, "start": 1, "stop": 2})
    np.testing.assert_array_equal(s, x[1:2])


def test_concat_order_and_axis():
    a = np.zeros((2, 2), np.float32)
    b = np.ones((2, 2), np.float32)
    spec = get_op("concat")
    out = spec.fn({"a": a, "b": b}, ["a", "b"], {"axis": 0}, CTX)
    np.testing.assert_array_equal(out, np.concatenate([a, b], axis=0))
    with pytest.raises(ValueError):
        spec.fn({"a": a}, ["a"], {"axis": 0}, CTX)


def test_elementwise_unary():
    x = np.array([1.0, 4.0, 9.0], np.float32)
    np.testing.assert_allclose(run("neg", x), -x)
    np.testing.assert_allclose(run("log", x), np.log(x))
    np.testing.assert_allclose(run("exp", x), np.exp(x))


def test_binary_ops():
    a = np.array([2.0, 6.0], np.float32)
    b = np.array([1.0, 3.0], np.float32)
    np.testing.assert_allclose(run2("add", a, b), a + b)
    np.testing.assert_allclose(run2("sub", a, b), a - b)
    np.testing.assert_allclose(run2("mul", a, b), a * b)
    np.testing.assert_allclose(run2("div", a, b), a / b)
    np.testing.assert_allclose(run("add", a, {"value": 1}), a + 1)
    with pytest.raises(ValueError):
        run("div", a, {})


def _ref_zip(x, axis, block):
    """Reference: out = concat(b[0::2], b[1::2]) along axis in blocks."""
    moved = np.moveaxis(x, axis, 0)
    n = moved.shape[0] // block
    v = moved.reshape(n, block, *moved.shape[1:])
    out = np.concatenate([v[0::2], v[1::2]], axis=0).reshape(moved.shape)
    return np.moveaxis(out, 0, axis)


def _ref_unzip(y, axis, block):
    moved = np.moveaxis(y, axis, 0)
    n = moved.shape[0] // block
    v = moved.reshape(n, block, *moved.shape[1:])
    out = np.empty_like(v)
    out[0::2] = v[: n // 2]
    out[1::2] = v[n // 2 :]
    return np.moveaxis(out.reshape(moved.shape), 0, axis)


@pytest.mark.parametrize("axis,block", [(0, 1), (0, 2), (1, 3), (-1, 3)])
def test_zip_unzip_roundtrip(axis, block):
    shape = [4, 12, 6]
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    z = run("zip_blocks", x, {"axis": axis, "block": block})
    np.testing.assert_array_equal(z, _ref_zip(x, axis, block))
    u = run("unzip_blocks", z, {"axis": axis, "block": block})
    np.testing.assert_array_equal(u, x)
    # unzip is the inverse of zip on the unpermuted input too
    u2 = run("unzip_blocks", x, {"axis": axis, "block": block})
    np.testing.assert_array_equal(run("zip_blocks", u2, {"axis": axis, "block": block}), x)


def test_block_validation():
    x = np.zeros((8, 4), np.float32)
    with pytest.raises(ValueError):
        run("unzip_blocks", x, {"axis": 0, "block": 3})
    with pytest.raises(ValueError):
        run("zip_blocks", x, {"axis": 0, "block": 8})  # odd block count


def test_ops_are_pure():
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    orig = x.copy()
    run("unzip_blocks", x, {"axis": 0, "block": 2})
    run("transpose", x)
    run("neg", x)
    np.testing.assert_array_equal(x, orig)
