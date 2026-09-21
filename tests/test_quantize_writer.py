"""Quantization + shard writer unit tests."""

import json

import numpy as np
import pytest

from gguf2mlx_stream.errors import ConversionError
from gguf2mlx_stream.quantize import dequantize_weights, quantize_weights
from gguf2mlx_stream.writer import ShardedSafetensorsWriter, build_output_config


def test_quantize_roundtrip_4bit():
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((64, 128)) * 0.05).astype(np.float32)
    packed, scales, biases = quantize_weights(w, bits=4, group_size=64)
    assert packed.dtype == np.uint32
    assert scales.dtype == np.float16 and biases.dtype == np.float16
    assert packed.shape == (64, 16)  # 2 groups/row, 8 uint32 per 64-elem group
    assert scales.shape == (64, 2)
    d = dequantize_weights(packed, scales, biases, bits=4, group_size=64)
    err = np.abs(np.asarray(d) - w).max()
    assert err < 0.05, err  # 4-bit group step over a ±0.2 range


def test_quantize_roundtrip_6bit():
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((32, 64)) * 0.05).astype(np.float32)
    packed, scales, biases = quantize_weights(w, bits=6, group_size=64)
    assert packed.shape == (32, 12)  # 1 group/row: 64 * 6 bits = 12 uint32
    d = np.asarray(dequantize_weights(packed, scales, biases, bits=6, group_size=64))
    assert np.abs(d - w).max() < 0.005


def test_quantize_rejects_bad_grouping():
    w = np.zeros((8, 48), np.float32)
    with pytest.raises(ConversionError):
        quantize_weights(w, bits=4, group_size=64)
    with pytest.raises(ConversionError):
        quantize_weights(np.zeros((8,), np.float32), bits=4, group_size=64)


def test_shard_writer_splits_and_indexes(tmp_path):
    w = ShardedSafetensorsWriter(str(tmp_path), max_shard_bytes=1024)
    keys = []
    for i in range(6):
        key = f"t{i}.weight"
        w.add(key, np.full((8, 8), i, np.float32))
        keys.append(key)
    index = w.finalize()
    assert w.n_shards >= 2
    files = sorted(set(index["weight_map"].values()))
    assert files == [f"model-{i:05d}-of-{w.n_shards:05d}.safetensors" for i in range(1, w.n_shards + 1)]
    on_disk = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".safetensors")
    assert on_disk == files  # pre-rename files must not linger
    assert index["metadata"]["total_size"] == sum(
        np.full((8, 8), i, np.float32).nbytes for i in range(6)
    )
    # every key maps to an existing file and round-trips
    from safetensors.numpy import load_file

    for key, fname in index["weight_map"].items():
        data = load_file(str(tmp_path / fname))
        assert key in data


def test_build_output_config():
    # flat families merge text fields at the top level
    cfg = build_output_config(
        model_type="m",
        architectures=["A"],
        top_level={"tie_word_embeddings": False},
        text_config={"hidden_size": 4},
        quantization={"bits": 4, "group_size": 64, "mode": "affine"},
    )
    assert cfg["model_type"] == "m"
    assert cfg["hidden_size"] == 4
    assert "text_config" not in cfg
    assert cfg["quantization"] == cfg["quantization_config"]
    # nested families place fields under nest_under
    cfg_nest = build_output_config(
        model_type="m", architectures=["A"], top_level={},
        text_config={"hidden_size": 4}, quantization=None, nest_under="text_config",
    )
    assert cfg_nest["text_config"] == {"hidden_size": 4}
    cfg2 = build_output_config("m", ["A"], {}, {}, None)
    assert "quantization" not in json.dumps(cfg2)
