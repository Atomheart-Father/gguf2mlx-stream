"""GGUF source layer.

Reads a GGUF file via mmap and exposes metadata plus per-tensor,
bounded-memory dequantized access. This layer knows nothing about model
architectures; it only understands the GGUF container and GGML quantization
formats (delegation to the ``gguf`` package, MIT licensed, llama.cpp project).

Memory contract: tensors are never fully materialized eagerly. Reads are
either whole-tensor (bounded by one tensor's dequantized size) or row-range
slices; the underlying file is mmap'd so pages are evictable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, dequantize
from gguf.constants import GGML_QUANT_SIZES

from ..errors import SourceError


@dataclass(frozen=True)
class TensorInfo:
    """Static description of one tensor inside a GGUF file.

    ``hf_shape`` follows the HF/PyTorch convention for 2-D weights,
    ``(out_features, in_features)``, which is the reverse of the GGUF ``ne``
    storage order. For 1-D tensors the two are identical.
    """

    name: str
    ne: tuple[int, ...]
    qtype: GGMLQuantizationType
    n_elements: int
    n_bytes: int
    data_offset: int

    @property
    def hf_shape(self) -> tuple[int, ...]:
        return tuple(reversed(self.ne))

    @property
    def is_quantized(self) -> bool:
        return self.qtype not in (
            GGMLQuantizationType.F32,
            GGMLQuantizationType.F16,
            GGMLQuantizationType.F64,
            GGMLQuantizationType.BF16,
        )

    @property
    def row_bytes(self) -> int:
        """Bytes consumed by one outermost row (``ne[0]`` elements)."""
        block_size, type_size = GGML_QUANT_SIZES[self.qtype]
        if self.ne[0] % block_size != 0:
            raise SourceError(
                f"tensor {self.name!r}: ne[0]={self.ne[0]} not divisible by "
                f"block size {block_size} for {self.qtype.name}"
            )
        return self.ne[0] // block_size * type_size

    @property
    def n_rows(self) -> int:
        return int(np.prod(self.ne[1:])) if len(self.ne) > 1 else 1


def _scalar(field) -> Any:
    return field.contents()


class GGUFSource:
    """mmap-backed GGUF reader with dequantizing accessors."""

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        if not os.path.isfile(self.path):
            raise SourceError(f"GGUF file not found: {self.path}")
        try:
            self._reader = GGUFReader(self.path)
        except Exception as exc:  # malformed container
            raise SourceError(f"failed to open GGUF {self.path}: {exc}") from exc
        self._data = self._reader.data  # np.memmap over the whole file
        self.tensors: dict[str, TensorInfo] = {}
        for t in self._reader.tensors:
            if t.name in self.tensors:
                raise SourceError(f"duplicate tensor name in GGUF: {t.name}")
            self.tensors[t.name] = TensorInfo(
                name=t.name,
                ne=tuple(int(x) for x in t.shape),
                qtype=t.tensor_type,
                n_elements=int(t.n_elements),
                n_bytes=int(t.n_bytes),
                data_offset=int(t.data_offset),
            )
        self._metadata: dict[str, Any] | None = None

    # ---------- metadata ----------

    @property
    def arch(self) -> str:
        value = self.metadata.get("general.architecture")
        if not value:
            raise SourceError(f"{self.path}: missing general.architecture")
        return str(value)

    @property
    def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            meta: dict[str, Any] = {}
            for key, field in self._reader.fields.items():
                try:
                    if field.types[0].name == "ARRAY":
                        meta[key] = f"<array:{field.types[1].name}>"
                    else:
                        meta[key] = _scalar(field)
                except Exception:
                    meta[key] = "<unreadable>"
            self._metadata = meta
        return self._metadata

    def metadata_value(self, key: str) -> Any | None:
        """Return a scalar metadata value, trying ``key`` then ``{arch}.key``."""
        meta = self.metadata
        if key in meta:
            return meta[key]
        arch_key = f"{self.arch}.{key}"
        if arch_key in meta:
            return meta[arch_key]
        return None

    # ---------- tensor access ----------

    def info(self, name: str) -> TensorInfo:
        try:
            return self.tensors[name]
        except KeyError:
            raise SourceError(f"tensor not found in GGUF: {name!r}") from None

    def _raw_rows(self, name: str, lo: int, hi: int) -> np.ndarray:
        """Raw quantized bytes for outer rows [lo, hi) of tensor ``name``."""
        t = self.info(name)
        if not (0 <= lo < hi <= t.n_rows):
            raise SourceError(
                f"tensor {name!r}: row range [{lo}, {hi}) invalid for {t.n_rows} rows"
            )
        start = t.data_offset + lo * t.row_bytes
        stop = t.data_offset + hi * t.row_bytes
        return self._data[start:stop]

    def read_rows(self, name: str, lo: int = 0, hi: int | None = None) -> np.ndarray:
        """Dequantized float32 rows [lo, hi) along the *outer* GGUF axis.

        For a 2-D tensor with hf_shape (out, in) this returns shape
        (hi - lo, in): GGUF stores each outer row's ``ne[0]`` inner elements
        contiguously, so the slice reshapes directly to (rows, in).
        """
        t = self.info(name)
        hi = t.n_rows if hi is None else hi
        raw = np.asarray(self._raw_rows(name, lo, hi))
        flat = dequantize(raw, t.qtype)
        rows = hi - lo
        if t.n_rows == 1:
            return np.ascontiguousarray(flat, dtype=np.float32)
        inner = t.ne[0]
        return np.ascontiguousarray(flat.reshape(rows, inner), dtype=np.float32)

    def read_matrix(self, name: str) -> np.ndarray:
        """Full tensor as float32 with HF-convention shape (for 2-D: (out, in))."""
        t = self.info(name)
        flat = dequantize(np.asarray(self._raw_rows(name, 0, t.n_rows)), t.qtype)
        return np.ascontiguousarray(flat, dtype=np.float32).reshape(t.hf_shape)

    def read_vector(self, name: str) -> np.ndarray:
        """Full 1-D tensor as float32."""
        t = self.info(name)
        if len(t.ne) != 1:
            raise SourceError(f"tensor {name!r} is not 1-D: ne={t.ne}")
        return self.read_rows(name)

    def iter_infos(self) -> Iterator[TensorInfo]:
        return iter(self.tensors.values())

    # ---------- summary ----------

    def summary(self) -> dict[str, Any]:
        total_bytes = sum(t.n_bytes for t in self.tensors.values())
        by_type: dict[str, int] = {}
        for t in self.tensors.values():
            by_type[t.qtype.name] = by_type.get(t.qtype.name, 0) + 1
        return {
            "path": self.path,
            "architecture": self.metadata.get("general.architecture"),
            "n_tensors": len(self.tensors),
            "total_tensor_bytes": total_bytes,
            "tensors_by_type": dict(sorted(by_type.items())),
        }

    def dump_metadata_json(self) -> str:
        return json.dumps(self.metadata, indent=2, default=str)
