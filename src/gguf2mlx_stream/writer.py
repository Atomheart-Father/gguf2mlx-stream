"""MLX-LM output writer: sharded safetensors + index + config.json + tokenizer.

Produces a standard MLX-LM model directory (``mlx_lm.load()`` compatible):

    config.json
    model-00001-of-0000N.safetensors ...
    model.safetensors.index.json
    tokenizer files (copied from a source directory)

Shards are streamed: tensors are buffered into the current shard and flushed
whenever ``max_shard_bytes`` would be exceeded. A single tensor is never
split across shards.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Mapping

import numpy as np
from safetensors.numpy import save_file

from .errors import ConversionError


class ShardedSafetensorsWriter:
    """Buffers tensors into shards and finalizes an MLX-LM directory."""

    def __init__(self, out_dir: str, max_shard_bytes: int):
        self.out_dir = out_dir
        self.max_shard_bytes = max_shard_bytes
        self._shard_idx = 0
        self._shard: dict[str, np.ndarray] = {}
        self._shard_bytes = 0
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0

    def add(self, key: str, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr)
        self._shard[key] = arr
        self._shard_bytes += int(arr.nbytes)
        self.total_bytes += int(arr.nbytes)
        if self._shard_bytes >= self.max_shard_bytes:
            self.flush()

    def add_quantized(self, dest: str, packed: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> None:
        if dest.endswith(".weight"):
            s, b = dest[: -len(".weight")] + ".scales", dest[: -len(".weight")] + ".biases"
        else:
            s, b = dest + ".scales", dest + ".biases"
        self.add(dest, packed)
        self.add(s, scales)
        self.add(b, biases)

    def flush(self) -> str | None:
        if not self._shard:
            return None
        self._shard_idx += 1
        fname = f"model-{self._shard_idx:05d}.safetensors"
        save_file(
            self._shard,
            os.path.join(self.out_dir, fname),
            metadata={"format": "pt"},
        )
        for k in self._shard:
            self.weight_map[k] = fname
        written = self._shard_bytes
        self._shard = {}
        self._shard_bytes = 0
        return fname if written else None

    @property
    def n_shards(self) -> int:
        return self._shard_idx

    def finalize(self) -> dict[str, Any]:
        """Flush, rename shards to ``-of-N`` form, and write the index."""
        self.flush()
        n = self._shard_idx
        if n == 0:
            raise ConversionError("no tensors were written")
        final_map: dict[str, str] = {}
        for key, fname in self.weight_map.items():
            idx = int(fname.split("-")[1].split(".")[0])
            final_map[key] = f"model-{idx:05d}-of-{n:05d}.safetensors"
        for i in range(1, n + 1):
            src = os.path.join(self.out_dir, f"model-{i:05d}.safetensors")
            dst = os.path.join(self.out_dir, f"model-{i:05d}-of-{n:05d}.safetensors")
            os.replace(src, dst)
        index = {"metadata": {"total_size": self.total_bytes}, "weight_map": final_map}
        with open(os.path.join(self.out_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
        return index


def build_output_config(
    model_type: str,
    architectures: list[str],
    top_level: Mapping[str, Any],
    text_config: Mapping[str, Any],
    quantization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "architectures": list(architectures),
        "model_type": model_type,
    }
    cfg.update(dict(top_level))
    cfg["text_config"] = dict(text_config)
    if quantization:
        cfg["quantization"] = dict(quantization)
        cfg["quantization_config"] = dict(quantization)
    return cfg


def copy_tokenizer_files(source_dir: str, out_dir: str, filenames: list[str]) -> list[str]:
    copied = []
    for fn in filenames:
        src = os.path.join(source_dir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, fn))
            copied.append(fn)
    return copied
