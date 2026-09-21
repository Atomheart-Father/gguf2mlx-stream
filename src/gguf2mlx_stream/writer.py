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
    nest_under: str | None = None,
) -> dict[str, Any]:
    """Build config.json for an MLX-LM model directory.

    Nested families (e.g. qwen3_5 with ``nest_under="text_config"``) place
    model fields under the given key; flat families (llama, qwen3,
    gemma3_text) merge them at the top level, matching what their mlx-lm
    ``ModelArgs.from_dict`` implementations read.
    """
    cfg: dict[str, Any] = {
        "architectures": list(architectures),
        "model_type": model_type,
    }
    cfg.update(dict(top_level))
    if nest_under:
        cfg[nest_under] = dict(text_config)
    else:
        cfg.update(dict(text_config))
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


# ---------------------------------------------------------------------------
# tokenizer output contract
# ---------------------------------------------------------------------------
#
# A successful conversion MUST produce a directory whose tokenizer can be
# loaded by ``mlx_lm.load()``. The minimal file set that guarantees this is:
#
#   * one vocab-capable tokenizer file — ``tokenizer.json`` (fast tokenizer,
#     fully self-describing) or ``tokenizer.model`` (sentencepiece);
#   * ``tokenizer_config.json`` — carries the tokenizer class, special
#     tokens and (for our target families) the chat template; without it
#     the directory is not a standard MLX-LM model directory.
#
# Conversions fail transactionally when this set cannot be produced: the
# staging directory is discarded and an existing output is never replaced.

TOKENIZER_VOCAB_FILES = ("tokenizer.json", "tokenizer.model")
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


def _nonempty_file(directory: str, name: str) -> bool:
    path = os.path.join(directory, name)
    return os.path.isfile(path) and os.path.getsize(path) > 0


def describe_tokenizer_status(directory: str) -> str:
    vocab = [f for f in TOKENIZER_VOCAB_FILES if _nonempty_file(directory, f)]
    has_cfg = _nonempty_file(directory, TOKENIZER_CONFIG_FILE)
    return f"vocab={vocab or 'none'} {TOKENIZER_CONFIG_FILE}={'yes' if has_cfg else 'no'}"


def check_tokenizer_available(source_dir: str) -> None:
    """Fail fast *before* conversion when the source cannot supply a tokenizer."""
    if not any(_nonempty_file(source_dir, f) for f in TOKENIZER_VOCAB_FILES):
        raise ConversionError(
            f"tokenizer source {source_dir!r} has no usable tokenizer file: one of "
            f"{list(TOKENIZER_VOCAB_FILES)} is required (pass --tokenizer-source "
            "pointing at a directory with HF tokenizer files, e.g. the model's "
            "tokenizer/ snapshot)"
        )
    if not _nonempty_file(source_dir, TOKENIZER_CONFIG_FILE):
        raise ConversionError(
            f"tokenizer source {source_dir!r} is missing {TOKENIZER_CONFIG_FILE}, "
            "which is required for a standard MLX-LM model directory"
        )
    _validate_tokenizer_json(source_dir, "tokenizer source")


def check_tokenizer_output(out_dir: str) -> None:
    """Fail before commit when the staged output lacks a loadable tokenizer."""
    if not any(_nonempty_file(out_dir, f) for f in TOKENIZER_VOCAB_FILES):
        raise ConversionError(
            f"output directory {out_dir!r} would lack a loadable tokenizer: none of "
            f"{list(TOKENIZER_VOCAB_FILES)} present "
            f"({describe_tokenizer_status(out_dir)})"
        )
    if not _nonempty_file(out_dir, TOKENIZER_CONFIG_FILE):
        raise ConversionError(
            f"output directory {out_dir!r} would lack {TOKENIZER_CONFIG_FILE}"
        )
    _validate_tokenizer_json(out_dir, "output")


def _validate_tokenizer_json(directory: str, what: str) -> None:
    path = os.path.join(directory, "tokenizer.json")
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
    except (OSError, ValueError) as exc:
        raise ConversionError(f"{what} has an invalid tokenizer.json: {exc}") from exc
