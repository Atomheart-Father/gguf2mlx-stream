"""Bounded-memory conversion executor.

Executes a :class:`~gguf2mlx_stream.planner.ConversionPlan` job by job:

    read quantized GGUF tensor (mmap) or row-chunk
        -> dequantize (bounded: one tensor or one chunk at a time)
        -> operator steps (pure numpy, float32)
        -> MLX affine quantize (or dtype cast)
        -> shard buffer (flushed at max_shard_bytes)
        -> release

No step ever materializes more than one tensor (or one chunk of a huge
tensor) of dequantized weights. There is no full-model FP16 staging.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import resource
import shutil
import time
import uuid
from typing import Any, Callable, Mapping

import numpy as np
import mlx.core as mx

from .errors import ConversionError
from .planner import ConversionPlan, PlannedJob
from .quantize import quantize_weights
from .ops import all_ops
from .ops.base import OpContext
from .source.gguf import GGUFSource
from .writer import (
    ShardedSafetensorsWriter,
    build_output_config,
    check_tokenizer_available,
    check_tokenizer_output,
    copy_tokenizer_files,
)


@dataclasses.dataclass
class QuantSettings:
    bits: int | None = 4
    group_size: int = 64
    mode: str = "affine"

    @property
    def enabled(self) -> bool:
        return self.bits is not None


@dataclasses.dataclass
class RunStats:
    output_bytes: int = 0
    n_tensors: int = 0
    n_shards: int = 0
    elapsed_s: float = 0.0
    peak_rss_gib: float = 0.0


def peak_rss_gib() -> float:
    # macOS reports ru_maxrss in bytes; Linux in KiB.
    import sys

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 2**30 if sys.platform == "darwin" else raw / 2**20


def _clear() -> None:
    gc.collect()
    try:
        mx.clear_cache()
    except AttributeError:
        pass


class ConversionRunner:
    def __init__(
        self,
        plan: ConversionPlan,
        source: GGUFSource,
        out_dir: str,
        quant: QuantSettings | None = None,
        ref_config: Mapping[str, Any] | None = None,
        tokenizer_source: str | None = None,
        chunk_elements: int = 1 << 27,  # ~128M float32 elements (~512 MiB) per chunk
        check_finite: bool = True,
        log: Callable[[str], None] = print,
        max_shard_bytes: int | None = None,
        bits_record: Mapping[str, Any] | None = None,
    ):
        self.plan = plan
        self.max_shard_bytes = (
            int(max_shard_bytes) if max_shard_bytes else plan.config.output.max_shard_bytes
        )
        self.source = source
        self.out_dir = out_dir
        self.quant = quant or QuantSettings()
        self.ref_config = ref_config
        self.tokenizer_source = tokenizer_source
        self.chunk_elements = chunk_elements
        self.check_finite = check_finite
        self.log = log
        self.bits_record = dict(bits_record) if bits_record else None
        self.stats: RunStats | None = None

    # ---------- job execution ----------

    def _job_inputs(self, job: PlannedJob) -> dict[str, np.ndarray]:
        slots: dict[str, np.ndarray] = {}
        for sname, tensor_name, rows in job.slots:
            if rows is not None:
                arr = self.source.read_rows(tensor_name, rows[0], rows[1])
            else:
                arr = self.source.read_matrix(tensor_name)
                in_mem = next((s for n, _, s in job.slot_specs if n == sname), None)
                if in_mem is not None:
                    sl = [slice(None)] * arr.ndim
                    hi = in_mem.hi if in_mem.hi is not None else arr.shape[in_mem.axis]
                    sl[in_mem.axis] = slice(int(in_mem.lo), int(hi))
                    arr = np.ascontiguousarray(arr[tuple(sl)])
            slots[sname] = arr
        return slots

    def _run_steps(
        self, job: PlannedJob, slots: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        available = dict(slots)
        prev: str | None = None
        ctx = OpContext(dims=self.plan.dims, tensor_name=job.source.name, dest_name=job.dest)
        if not job.steps:
            if len(slots) != 1:
                raise ConversionError(f"{job.dest}: multi-slot job requires operator steps")
            return slots["x"]
        for i, step in enumerate(job.steps):
            spec = all_ops()[step.op]
            if step.inputs:
                missing = [n for n in step.inputs if n not in available]
                if missing:
                    raise ConversionError(f"{job.dest}: step {i} missing inputs {missing}")
                order = list(step.inputs)
                inputs = {n: available[n] for n in order}
            else:
                key = step.input or prev or "x"
                if key == "_":
                    key = prev
                if key is None or key not in available:
                    raise ConversionError(f"{job.dest}: step {i} input {step.input!r} not available")
                order = [key]
                inputs = {key: available[key]}
            try:
                out = spec.fn(inputs, order, dict(step.args), ctx)
            except Exception as exc:
                raise ConversionError(
                    f"{job.dest}: operator {step.op!r} failed: {exc}"
                ) from exc
            key = step.output or f"_step{i}"
            available[key] = out
            prev = key
        return available[prev]

    # ---------- main loop ----------

    def run(self, overwrite: bool = False) -> RunStats:
        """Convert all planned jobs into a *transactional* output directory.

        Everything is written to a sibling staging directory
        (``<out>.tmp-<uuid>``) and atomically renamed into place only after
        the index, config.json, and tokenizer files are complete. A failed
        conversion never leaves a partial model at the output path, and an
        existing output is never replaced without ``overwrite=True``.

        The tokenizer output contract is enforced before any tensor is read
        (tokenizer source must be able to supply a loadable tokenizer) and
        again on the staged output right before the commit.
        """
        out_path = os.path.abspath(self.out_dir)
        parent = os.path.dirname(out_path)
        if os.path.exists(out_path) and not os.path.isdir(out_path):
            raise ConversionError(f"output path exists and is not a directory: {out_path}")
        os.makedirs(parent, exist_ok=True)
        if os.path.isdir(out_path) and os.listdir(out_path) and not overwrite:
            raise ConversionError(
                f"output directory {out_path!r} already exists and is not empty; "
                "pass --overwrite to replace it"
            )
        if self.tokenizer_source:
            check_tokenizer_available(self.tokenizer_source)
        stage = f"{out_path}.tmp-{uuid.uuid4().hex[:8]}"
        old = f"{out_path}.old-{uuid.uuid4().hex[:8]}"
        os.makedirs(stage)
        try:
            self._run_into(stage)
            if os.path.isdir(out_path) and os.listdir(out_path):
                if not overwrite:
                    raise ConversionError(
                        f"output directory {out_path!r} already exists; "
                        "pass --overwrite to replace it"
                    )
                os.rename(out_path, old)
                try:
                    os.rename(stage, out_path)
                except OSError:
                    os.rename(old, out_path)  # restore previous output
                    raise
                shutil.rmtree(old, ignore_errors=True)
            else:
                if os.path.isdir(out_path):
                    os.rmdir(out_path)  # replace an empty pre-created dir
                os.rename(stage, out_path)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        return self.stats

    def _run_into(self, out_dir: str) -> None:
        writer = ShardedSafetensorsWriter(out_dir, self.max_shard_bytes)
        t0 = time.time()
        stats = RunStats()

        for job in self.plan.jobs:
            self._convert_job(job, writer)
            stats.n_tensors += 1
            self.log(
                f"[job] {job.dest}  ({job.source.n_bytes / 2**20:.1f} MiB source, "
                f"peak RSS {peak_rss_gib():.2f} GiB)"
            )
            _clear()

        index = writer.finalize()
        stats.output_bytes = writer.total_bytes
        stats.n_shards = writer.n_shards

        self._emit_config(out_dir, writer, index)
        copied: list[str] = []
        if self.tokenizer_source:
            copied = copy_tokenizer_files(
                self.tokenizer_source,
                out_dir,
                list(self.plan.config.output.tokenizer_files),
            )
            self.log(f"[out] tokenizer files copied: {copied or 'none found'}")
        # tokenizer output contract: a successful conversion is always a
        # loadable MLX-LM directory; otherwise the transaction aborts here,
        # before the staged output can replace an existing model
        check_tokenizer_output(out_dir)

        stats.elapsed_s = time.time() - t0
        stats.peak_rss_gib = peak_rss_gib()
        self.log(
            f"[done] {stats.output_bytes / 2**30:.2f} GiB across {stats.n_shards} shard(s), "
            f"{stats.n_tensors} tensors, {stats.elapsed_s:.0f}s, peak RSS "
            f"{stats.peak_rss_gib:.2f} GiB"
        )
        self.stats = stats

    def _effective_quant(self, job: PlannedJob) -> tuple[int, int]:
        """Per-job quantization parameters: rule override else conversion-level."""
        bits = job.bits if job.bits is not None else self.quant.bits
        group = job.group_size if job.group_size is not None else self.quant.group_size
        return int(bits), int(group)

    def _convert_job(self, job: PlannedJob, writer: ShardedSafetensorsWriter) -> None:
        n_rows = job.source.n_rows
        inner = job.source.ne[0]
        use_quant = job.quantize and self.quant.enabled
        chunk_rows = max(1, self.chunk_elements // max(inner, 1))

        if use_quant and job.chunkable and n_rows > chunk_rows:
            self._convert_chunked(job, writer, chunk_rows)
            return

        slots = self._job_inputs(job)
        result = self._run_steps(job, slots)

        if self.check_finite and not np.isfinite(result).all():
            raise ConversionError(f"{job.dest}: non-finite values after transform")

        if use_quant:
            bits, group = self._effective_quant(job)
            packed, scales, biases = quantize_weights(result, bits, group)
            writer.add_quantized(job.dest, packed, scales, biases)
        else:
            dtype = "float16" if (job.quantize and not self.quant.enabled) else job.dtype
            writer.add(job.dest, result.astype(dtype))
        del slots, result

    def _convert_chunked(
        self, job: PlannedJob, writer: ShardedSafetensorsWriter, chunk_rows: int
    ) -> None:
        """Stream a huge quantization-only job in bounded row chunks.

        Chunks are read along the *flattened leading axes* (GGUF outer rows),
        quantized per chunk, and concatenated along axis 0. Quantization only
        touches the last axis, so the concatenated result reshapes exactly to
        the N-D packed layout for any rank (e.g. 3-D expert tensors).
        """
        bits, group = self._effective_quant(job)
        n_rows = job.source.n_rows
        name = job.source.name
        packed_parts, scale_parts, bias_parts = [], [], []
        for lo in range(0, n_rows, chunk_rows):
            hi = min(n_rows, lo + chunk_rows)
            arr = self.source.read_rows(name, lo, hi)
            arr = self._run_steps(job, {"x": arr})
            packed, scales, biases = quantize_weights(arr, bits, group)
            packed_parts.append(packed)
            scale_parts.append(scales)
            bias_parts.append(biases)
            del arr, packed, scales, biases
        packed = np.concatenate(packed_parts)
        scales = np.concatenate(scale_parts)
        biases = np.concatenate(bias_parts)
        packed_parts.clear()
        scale_parts.clear()
        bias_parts.clear()
        lead = tuple(int(d) for d in job.out_shape[:-1])
        if len(lead) > 1:
            # restore the N-D leading layout (row-order preserved by concat)
            packed = packed.reshape(lead + packed.shape[1:])
            scales = scales.reshape(lead + scales.shape[1:])
            biases = biases.reshape(lead + biases.shape[1:])
        writer.add_quantized(job.dest, packed, scales, biases)

    # ---------- metadata emission ----------

    def _emit_config(
        self, out_dir: str, writer: ShardedSafetensorsWriter, index: Mapping[str, Any]
    ) -> None:
        from .planner import DimResolver, _ARITH

        out = self.plan.config.output
        if not out.model_type:
            raise ConversionError("config output.model_type is not set")
        resolver = DimResolver(self.source, self.ref_config)

        def _resolve_deep(spec: Any, where: str) -> Any:
            # Only plain mappings nest deeper specs; an arithmetic spec
            # ({mul/not/...}) and fallback-chain lists go to the resolver as
            # a whole (literal list values come from ref: paths).
            if (
                isinstance(spec, dict)
                and not (len(spec) == 1 and next(iter(spec), None) in _ARITH)
            ):
                return {
                    k: _resolve_deep(v, f"{where}.{k}")
                    for k, v in spec.items()
                }
            return resolver.resolve(spec, where, self.plan.dims)

        text_config: dict[str, Any] = {}
        if self.ref_config:
            base = {
                k: v
                for k, v in self.ref_config.items()
                if k not in out.ref_drop_fields and k not in ("architectures", "model_type")
            }
            text_config.update(base)
        for key, spec in out.text_config.items():
            value = _resolve_deep(spec, f"output.text_config.{key}")
            if key in text_config and text_config[key] != value:
                self.log(
                    f"[out] WARNING: text_config.{key}: reference value "
                    f"{text_config[key]!r} overridden by resolved {value!r}"
                )
            text_config[key] = value

        top_level = {
            k: _resolve_deep(v, f"output.top_level.{k}")
            for k, v in out.top_level.items()
        }
        quant: dict[str, Any] | None = None
        if self.quant.enabled:
            quant = {
                "bits": self.quant.bits,
                "group_size": self.quant.group_size,
                "mode": self.quant.mode,
            }
            # per-rule overrides: recorded flat under the owning *module*
            # path (weight path minus the trailing ".weight"), which is the
            # key format mlx-lm's loader and mlx.nn.quantize class_predicate
            # compare against
            for job in self.plan.jobs:
                if not job.quantize:
                    continue
                bits, group = self._effective_quant(job)
                if (bits, group) != (self.quant.bits, self.quant.group_size):
                    module_key = (
                        job.dest[: -len(".weight")]
                        if job.dest.endswith(".weight")
                        else job.dest
                    )
                    quant[module_key] = {"bits": bits, "group_size": group}
        cfg = build_output_config(
            model_type=out.model_type,
            architectures=list(out.architectures),
            top_level=top_level,
            text_config=text_config,
            quantization=quant,
            nest_under=out.nest_config_under,
        )

        missing = [
            path
            for path in out.required_fields
            if _dig(cfg, path) is _MISSING
        ]
        if missing:
            raise ConversionError(
                "output config.json is missing required architecture field(s) "
                f"(broken or over-permissive config): {missing}"
            )
        if self.quant.enabled:
            # evidence block: how the target bits were chosen, the source
            # quant histogram backing that choice, and the final target
            # (top-level sibling of "quantization"; mlx-lm ignores unknown keys)
            cfg["quantization_selection"] = dict(self.bits_record or {})
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)


_MISSING = object()


def _dig(cfg: Mapping[str, Any], dotted: str) -> Any:
    """Walk a dotted path; returns _MISSING when any segment is absent.
    A present-but-null value is a legitimate explicit value."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node
