"""Verification layer.

Checks a converted MLX-LM directory against its source GGUF and the
architecture config:

* index / shard consistency and key-set parity with the plan;
* bidirectional index <-> shard key-set checks (unindexed tensors, stale
  index entries and unreferenced shard files are all rejected);
* quantization parameter validation against ``config.json`` (bits,
  group_size, mode); explicit CLI arguments that conflict with the output
  metadata are a hard failure;
* shape checks for every planned tensor;
* finiteness (NaN/inf) checks for every tensor in the output;
* numeric checks: by default *every* quantized output tensor is recomputed
  from the GGUF through the same operator pipeline and compared against the
  saved weights (dequantized first). Sampled checking (first tensor per
  rule + small tensors) is available as an explicit opt-in via
  ``sampled=True`` / ``--sampled``.
* structural quantization checks (packed/scales/biases triple shapes);
* optional ``mlx_lm.load()`` smoke test.

llama.cpp (``llama-completion``) is deliberately NOT required: it is an
optional external semantic check, not part of this verifier.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from safetensors import safe_open

from .errors import VerifyError
from .ops import all_ops
from .ops.base import OpContext
from .planner import ConversionPlan
from .quantize import dequantize_weights
from .source.gguf import GGUFSource

_DEFAULT_TOL = {2: 0.06, 3: 0.04, 4: 0.02, 6: 0.01, 8: 0.005}  # absolute floors


@dataclass
class VerifyReport:
    checked_numeric: int = 0
    checked_shapes: int = 0
    checked_finite: int = 0
    failures: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _load_output(out_dir: str) -> tuple[dict[str, str], dict[str, Any]]:
    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        # single-file layout
        single = os.path.join(out_dir, "model.safetensors")
        if not os.path.isfile(single):
            raise VerifyError(f"{out_dir}: no safetensors index or model.safetensors")
        with safe_open(single, framework="numpy") as f:
            keys = list(f.keys())
        return {k: "model.safetensors" for k in keys}, {}
    with open(index_path) as f:
        index = json.load(f)
    return index["weight_map"], index.get("metadata", {})


def _check_index_shard_parity(
    out_dir: str, weight_map: Mapping[str, str], report: VerifyReport
) -> None:
    """Bidirectional index <-> shard key-set check.

    Every key the index maps into a shard must exist there, every key in an
    indexed shard must be listed in the index, and every ``*.safetensors``
    file in the directory must be referenced by the index.
    """
    indexed: dict[str, set[str]] = {}
    for key, fname in weight_map.items():
        indexed.setdefault(fname, set()).add(key)
    for fname in sorted(indexed):
        path = os.path.join(out_dir, fname)
        if not os.path.isfile(path):
            report.failures.append(f"missing shard file {fname}")
            continue
        with safe_open(path, framework="numpy") as f:
            actual = set(f.keys())
        want = indexed[fname]
        for k in sorted(want - actual):
            report.failures.append(
                f"index maps {k!r} to {fname}, which does not contain it"
            )
        for k in sorted(actual - want):
            report.failures.append(f"{fname} contains unindexed key {k!r}")
    present = {f for f in os.listdir(out_dir) if f.endswith(".safetensors")}
    for fname in sorted(present - set(indexed)):
        report.failures.append(f"shard file {fname} is not referenced by the index")


def _read_output_quantization(out_dir: str) -> dict[str, Any]:
    """Read the quantization block from the output's config.json."""
    cfg_path = os.path.join(out_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise VerifyError(f"{out_dir}: missing config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    q = cfg.get("quantization") or cfg.get("quantization_config")
    if not q:
        raise VerifyError("output is not quantized; verify expects a quantized output")
    q = dict(q)
    for field_name in ("bits", "group_size", "mode"):
        if field_name not in q or q[field_name] is None:
            raise VerifyError(
                f"output config.json quantization is missing {field_name!r}; "
                "a complete gguf2mlx-stream output always records it"
            )
    return q


def _saved(out_dir: str, weight_map: Mapping[str, str], key: str) -> np.ndarray:
    with safe_open(os.path.join(out_dir, weight_map[key]), framework="numpy") as f:
        if key not in f.keys():
            raise VerifyError(f"key {key!r} missing from {weight_map[key]}")
        return f.get_tensor(key)


def _job_quant_params(
    quant_cfg: Mapping[str, Any], job, report: VerifyReport
) -> tuple[int, int]:
    """Expected ``(bits, group_size)`` for one planned job.

    The output's config.json is the source of truth: a flat per-module entry
    inside the ``quantization`` mapping overrides the global settings. Keys
    are module paths (weight path minus the trailing ".weight"), which is
    also the format mlx-lm's loader consumes.
    """
    module_key = (
        job.dest[: -len(".weight")] if job.dest.endswith(".weight") else job.dest
    )
    override = quant_cfg.get(module_key)
    if isinstance(override, dict):
        try:
            return int(override["bits"]), int(override["group_size"])
        except (KeyError, TypeError, ValueError):
            report.failures.append(
                f"{module_key}: malformed per-tensor quantization override {override!r}"
            )
    return int(quant_cfg["bits"]), int(quant_cfg["group_size"])


def _recompute(plan: ConversionPlan, source: GGUFSource, job) -> np.ndarray:
    slots: dict[str, np.ndarray] = {}
    for sname, tensor_name, rows in job.slots:
        if rows is not None:
            arr = source.read_rows(tensor_name, rows[0], rows[1])
        else:
            arr = source.read_matrix(tensor_name)
            in_mem = next((s for n, _, s in job.slot_specs if n == sname), None)
            if in_mem is not None:
                sl = [slice(None)] * arr.ndim
                hi = in_mem.hi if in_mem.hi is not None else arr.shape[in_mem.axis]
                sl[in_mem.axis] = slice(int(in_mem.lo), int(hi))
                arr = np.ascontiguousarray(arr[tuple(sl)])
        slots[sname] = arr
    available = dict(slots)
    prev = None
    ctx = OpContext(dims=plan.dims, tensor_name=job.source.name, dest_name=job.dest)
    if not job.steps:
        return slots["x"]
    for i, step in enumerate(job.steps):
        spec = all_ops()[step.op]
        if step.inputs:
            order = list(step.inputs)
            inputs = {n: available[n] for n in order}
        else:
            key = step.input or prev or "x"
            if key == "_":
                key = prev
            order = [key]
            inputs = {key: available[key]}
        out = spec.fn(inputs, order, dict(step.args), ctx)
        key = step.output or f"_step{i}"
        available[key] = out
        prev = key
    return available[prev]


def verify_conversion(
    plan: ConversionPlan,
    source: GGUFSource,
    out_dir: str,
    bits: int | None = None,
    group_size: int | None = None,
    mode: str | None = None,
    tolerance: float | None = None,
    sampled: bool = False,
    check_finite_all: bool = True,
) -> VerifyReport:
    """Run all checks; returns a report (never raises for check failures).

    Numeric coverage policy: by default every quantized output tensor is
    numerically recomputed and compared. ``sampled=True`` explicitly opts
    into the cheaper strategy (first tensor per rule + tensors whose source
    is smaller than 64 MiB). ``bits``/``group_size``/``mode`` default to the
    values recorded in the output's config.json; explicitly passing a
    different value is a hard failure.
    """
    report = VerifyReport()
    weight_map, meta = _load_output(out_dir)
    out_keys = set(weight_map)

    # bidirectional index <-> shard consistency (and unindexed shard files)
    _check_index_shard_parity(out_dir, weight_map, report)

    # key-set parity with the plan (quantized jobs also emit scales/biases)
    planned = set()
    for job in plan.jobs:
        planned.add(job.dest)
        if job.quantize:
            base = job.dest[: -len(".weight")] if job.dest.endswith(".weight") else job.dest
            planned.add(base + ".scales")
            planned.add(base + ".biases")
    extra = sorted(out_keys - planned)
    missing = sorted(planned - out_keys)
    if extra:
        report.failures.append(f"output contains keys not in plan: {extra[:5]}")
    if missing:
        report.failures.append(f"plan keys missing from output: {missing[:5]}")

    # quantization parameters: config.json is the source of truth; explicit
    # CLI arguments must agree with it
    q = _read_output_quantization(out_dir)
    cfg_bits, cfg_group, cfg_mode = int(q["bits"]), int(q["group_size"]), str(q["mode"])
    if bits is not None and bits != cfg_bits:
        raise VerifyError(
            f"--bits {bits} conflicts with output config.json quantization.bits {cfg_bits}"
        )
    if group_size is not None and group_size != cfg_group:
        raise VerifyError(
            f"--group-size {group_size} conflicts with output config.json "
            f"quantization.group_size {cfg_group}"
        )
    if mode is not None and mode != cfg_mode:
        raise VerifyError(
            f"--mode {mode!r} conflicts with output config.json quantization.mode {cfg_mode!r}"
        )

    seen_rules: set[str] = set()
    for job in plan.jobs:
        if job.dest not in out_keys:
            continue
        if job.quantize:
            base = job.dest[: -len(".weight")] if job.dest.endswith(".weight") else job.dest
            packed = _saved(out_dir, weight_map, job.dest)
            scales = _saved(out_dir, weight_map, base + ".scales")
            biases = _saved(out_dir, weight_map, base + ".biases")
            # expected quantization parameters for THIS tensor: the global
            # settings, unless the config records a per-key override (which
            # any rule-level bits/group_size override must have produced)
            job_bits, job_group = _job_quant_params(q, job, report)
            if job.rule.bits is not None:
                want_bits = job.rule.bits
                want_group = job.rule.group_size or cfg_group
                entry = q.get(base)
                if not isinstance(entry, dict):
                    report.failures.append(
                        f"{job.dest}: architecture config declares bits={want_bits} "
                        "but config.json quantization has no per-tensor override"
                    )
                elif (
                    int(entry.get("bits", -1)) != want_bits
                    or int(entry.get("group_size", -1)) != want_group
                ):
                    report.failures.append(
                        f"{job.dest}: config.json override {entry!r} does not match "
                        f"the architecture-config override (bits={want_bits}, "
                        f"group_size={want_group})"
                    )
            lead = tuple(int(d) for d in job.out_shape[:-1])
            inner = int(job.out_shape[-1])
            # structural checks are cheap: run for every quantized tensor.
            # N-D semantics: leading dims preserved, last dim packed/grouped.
            if packed.dtype != np.uint32 or packed.shape[:-1] != lead:
                report.failures.append(
                    f"{job.dest}: packed shape/dtype {packed.shape}/{packed.dtype} "
                    f"inconsistent with leading dims {lead}"
                )
            if packed.shape[-1] != inner * job_bits // 32:
                report.failures.append(
                    f"{job.dest}: packed last dim {packed.shape[-1]} inconsistent "
                    f"with inner dim {inner} at bits={job_bits}"
                )
            if scales.dtype != np.float16 or biases.dtype != np.float16:
                report.failures.append(f"{job.dest}: scales/biases must be float16")
            want_tail = (inner // job_group,)
            if scales.shape != lead + want_tail or biases.shape != lead + want_tail:
                report.failures.append(
                    f"{job.dest}: scales/biases shapes {scales.shape}/{biases.shape} "
                    f"inconsistent with leading dims {lead} and group_size {job_group}"
                )
            report.checked_shapes += 1
            # default: numerically check EVERY quantized tensor; sampling is
            # an explicit opt-in (first job per rule + small tensors)
            small = job.source.n_bytes < 64 * 2**20
            do_numeric = (not sampled) or small or (job.rule.display_name not in seen_rules)
            seen_rules.add(job.rule.display_name)
            if not do_numeric:
                continue
            expected = _recompute(plan, source, job)
            try:
                got = np.asarray(
                    dequantize_weights(packed, scales, biases, job_bits, job_group)
                )
            except Exception as exc:
                report.failures.append(
                    f"{job.dest}: recorded quantization parameters (bits={job_bits}, "
                    f"group_size={job_group}) do not match the saved weights: {exc}"
                )
                report.checked_numeric += 1
                continue
            # scale-aware default tolerance: one quantization step of the
            # tensor's value range, floored at a small absolute value
            tol = tolerance
            if tol is None:
                span = float(expected.max()) - float(expected.min())
                tol = max(_DEFAULT_TOL.get(job_bits, 0.1) * 0.2,
                          span / (2 ** job_bits - 1))
            diff = float(np.abs(got - expected).max())
            if got.shape != expected.shape:
                report.failures.append(
                    f"{job.dest}: shape {got.shape} != expected {expected.shape}"
                )
            elif diff > tol:
                report.failures.append(
                    f"{job.dest}: max|diff| {diff:.6g} > tol {tol:g}"
                )
            else:
                report.details.append(
                    f"OK {job.dest}: max|diff|={diff:.6g} (tol {tol:g})"
                )
            report.checked_numeric += 1
        else:
            expected = _recompute(plan, source, job)
            saved = _saved(out_dir, weight_map, job.dest)
            tol = tolerance if tolerance is not None else 1e-5
            report.checked_shapes += 1
            report.checked_numeric += 1
            if saved.shape != expected.shape:
                report.failures.append(
                    f"{job.dest}: shape {saved.shape} != expected {expected.shape}"
                )
            else:
                diff = float(np.abs(saved.astype(np.float64) - expected.astype(np.float64)).max())
                if diff > tol:
                    report.failures.append(
                        f"{job.dest}: max|diff| {diff:.6g} > tol {tol:g}"
                    )
                else:
                    report.details.append(
                        f"OK {job.dest}: max|diff|={diff:.6g} (tol {tol:g})"
                    )

    # finiteness over every tensor in the output
    if check_finite_all:
        for key in sorted(out_keys):
            arr = _saved(out_dir, weight_map, key)
            report.checked_finite += 1
            if arr.dtype in (np.float16, np.float32, np.float64) and not np.isfinite(arr).all():
                report.failures.append(f"{key}: contains non-finite values")

    return report


def load_test(out_dir: str, prompt: str = "Hello", max_tokens: int = 8) -> str:
    """Optional mlx-lm load + short generation smoke test."""
    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:  # pragma: no cover
        raise VerifyError(
            f"mlx-lm not available for load test: {exc} (pip install 'gguf2mlx-stream[loadtest]')"
        ) from None
    model, tokenizer = load(out_dir)
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )
    out = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0.0),
    )
    return out
