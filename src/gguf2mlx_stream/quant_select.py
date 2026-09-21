"""Target-bits auto selection from the source GGUF quantization histogram.

The MLX output is a single global bits setting plus the config's declared
per-rule overrides — it is NOT a per-tensor replica of the source quantization
mix. ``--bits auto`` therefore derives the global target from a *byte-weighted*
histogram of the source quant families over the tensors that the plan will
quantize:

    IQ2* -> 2, IQ3* -> 3, IQ4* -> 4, Q2* -> 2, Q3* -> 3, Q4* -> 4,
    Q6* -> 6, Q8* -> 8

Sources whose dominant family has no MLX affine equivalent (e.g. Q5*, IQ1*,
TQ*) fail with an explicit error asking the user to pass ``--bits``; auto never
silently guesses. An explicit ``--bits`` always wins and is recorded, not
re-derived.

Note on semantics: "source IQ3" and "MLX affine 3-bit" are the same *target
bit magnitude*, not bit-for-bit equivalent encodings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .constants import SUPPORTED_BITS
from .errors import PlanError
from .source.gguf import TensorInfo

# GGUF quant family name prefix -> MLX affine bits. Matching is by leading
# family marker (``IQ3_XXS`` -> IQ3 -> 3, ``Q4_K_M`` -> Q4 -> 4, ...). Q5 and
# IQ1 have no MLX affine equivalent and are deliberately unmappable.
_FAMILY_PATTERN = re.compile(r"^(IQ|Q)(\d)")


def family_bits(qtype_name: str) -> int | None:
    """Map a GGML quantization type name to MLX affine bits (None if unmappable)."""
    m = _FAMILY_PATTERN.match(qtype_name)
    if not m:
        return None
    bits = int(m.group(2))
    return bits if bits in SUPPORTED_BITS else None


def consumed_fraction(
    tensor: TensorInfo,
    rows: tuple[int, int] | None,
    in_mem: Any,
) -> float:
    """Fraction of a source tensor's elements consumed by one job slot."""
    frac = 1.0
    if rows is not None:
        frac *= (rows[1] - rows[0]) / max(tensor.n_rows, 1)
    if in_mem is not None:
        axis_len = tensor.hf_shape[in_mem.axis]
        hi = in_mem.hi if in_mem.hi is not None else axis_len
        frac *= (hi - in_mem.lo) / max(axis_len, 1)
    return min(max(frac, 0.0), 1.0)


def quant_histogram(
    jobs: Iterable[Any], resolve_info: Any
) -> tuple[dict[str, int], dict[str, float]]:
    """Byte-weighted histogram of source quant types over quantize-planned jobs.

    Jobs are duck-typed (need ``quantize``, ``slots``, ``slot_specs``) so tests
    can pass lightweight stand-ins; ``resolve_info(name) -> TensorInfo`` resolves
    slot tensor names (multi-slot jobs consume several source tensors). Each
    slot contributes the source bytes it actually consumes, so split/sliced
    tensors are not double counted.

    Returns ``(bytes_by_type, tensor_equivalents_by_type)``.
    """
    histogram: dict[str, int] = {}
    counts: dict[str, float] = {}
    for job in jobs:
        if not job.quantize:
            continue
        for (_, tensor_name, rows), (_, _, in_mem) in zip(job.slots, job.slot_specs):
            info = resolve_info(tensor_name)
            if not info.is_quantized:
                continue
            frac = consumed_fraction(info, rows, in_mem)
            if frac <= 0.0:
                continue
            histogram[info.qtype.name] = histogram.get(info.qtype.name, 0) + int(
                round(info.n_bytes * frac)
            )
            counts[info.qtype.name] = counts.get(info.qtype.name, 0.0) + frac
    return histogram, counts


@dataclass(frozen=True)
class BitsDecision:
    """Resolved target bits plus the evidence recorded alongside the output."""

    requested: str  # "auto" or the explicit value as given
    bits: int
    histogram_bytes: dict[str, int] = field(default_factory=dict)
    histogram_counts: dict[str, float] = field(default_factory=dict)
    dominant_type: str | None = None
    dominant_bits: int | None = None
    reason: str = ""

    def as_record(self) -> dict[str, Any]:
        """Serializable record for reports and the output config.json."""
        return {
            "requested": self.requested,
            "target_bits": self.bits,
            "dominant_source_type": self.dominant_type,
            "source_quant_histogram_bytes": dict(self.histogram_bytes),
            "source_quant_histogram_counts": {
                k: round(v, 3) for k, v in self.histogram_counts.items()
            },
            "reason": self.reason,
        }


def decide_target_bits(
    histogram: Mapping[str, int],
    requested: str | int,
    counts: Mapping[str, float] | None = None,
) -> BitsDecision:
    """Resolve ``requested`` ("auto" or explicit bits) against a byte histogram.

    ``histogram`` maps GGML quantization type names to consumed byte totals;
    ``counts`` optionally carries full-tensor-equivalent per-type counts for
    reporting. Explicit values always win; ``auto`` picks the byte-dominant
    family and requires it to be mappable.
    """
    if requested is None:
        return BitsDecision(
            requested="none",
            bits=0,
            reason="--no-quantize: weights are written as float16",
        )
    if isinstance(requested, int) or (isinstance(requested, str) and requested != "auto"):
        bits = int(requested)  # type: ignore[arg-type]
        if bits not in SUPPORTED_BITS:
            raise PlanError(
                f"unsupported --bits {bits}; choose one of {list(SUPPORTED_BITS)} or 'auto'"
            )
        return BitsDecision(
            requested=str(requested),
            bits=bits,
            histogram_bytes=dict(histogram),
            histogram_counts=dict(counts or {}),
            reason=f"explicit --bits {bits} selected by the user; "
            "source histogram recorded for reference only",
        )

    if not histogram:
        raise PlanError(
            "--bits auto needs at least one source tensor planned for "
            "quantization; pass --bits explicitly or --no-quantize"
        )
    total = sum(histogram.values())
    dominant_type, dominant_bytes = max(
        sorted(histogram.items()), key=lambda kv: (kv[1], kv[0])
    )
    dominant_bits = family_bits(dominant_type)
    share = dominant_bytes / total * 100.0
    if dominant_bits is None:
        mappable = {t: b for t, b in histogram.items() if family_bits(t) is not None}
        unmappable = {t: b for t, b in histogram.items() if family_bits(t) is None}
        raise PlanError(
            f"--bits auto: dominant source quant type {dominant_type!r} "
            f"({share:.1f}% of quantized source bytes) has no MLX affine "
            f"equivalent, and mixed quantization is not replicated per tensor. "
            f"Pass an explicit --bits from {list(SUPPORTED_BITS)}. "
            f"Source histogram (bytes): {dict(sorted(mappable.items(), key=lambda kv: -kv[1]))} "
            f"| unmappable: {dict(sorted(unmappable.items(), key=lambda kv: -kv[1]))}"
        )
    reason = (
        f"auto: dominant source family {dominant_type!r} covers {share:.1f}% of "
        f"quantized source bytes ({dominant_bytes / 2**20:.1f} MiB) and maps to "
        f"{dominant_bits} bits; the MLX output uses this global bit magnitude "
        f"plus declared per-rule overrides, not a per-tensor source replica"
    )
    return BitsDecision(
        requested="auto",
        bits=dominant_bits,
        histogram_bytes=dict(histogram),
        histogram_counts=dict(counts or {}),
        dominant_type=dominant_type,
        dominant_bits=dominant_bits,
        reason=reason,
    )


def select_target_bits(plan: Any, requested: str | int | None) -> BitsDecision:
    """Resolve the target bits for a compiled conversion plan.

    ``requested`` is the CLI ``--bits`` value (``"auto"`` by default) or
    ``None`` when ``--no-quantize`` was passed, in which case quantization is
    disabled and the decision records that fact.
    """
    if requested is None:
        return BitsDecision(
            requested="none",
            bits=0,
            reason="--no-quantize: weights are written as float16",
        )
    histogram, counts = quant_histogram(plan.jobs, plan.source.info)
    return decide_target_bits(histogram, requested, counts)


# Auto-derived targets with failing capability-gate evidence. Keyed by
# (architecture config id, dominant source family prefix). Until the gate
# evidence flips to PASS for a calibrated profile, plain ``--bits auto``
# refuses to produce these configurations; ``--allow-experimental`` or an
# explicit ``--bits`` overrides the guard.
EXPERIMENTAL_AUTO_TARGETS: dict[tuple[str, str], str] = {
    ("qwen3_5_moe", "IQ3"): (
        "auto-derived 3-bit for qwen35moe + IQ3 sources FAILED the "
        "ARC-Challenge capability gate (2026-09-21: clean accuracy 55% vs "
        "source 90%; Llama-1B 3-bit calibration also failed). 3-bit converts "
        "correctly but is not a recommended default. Pass --allow-experimental "
        "to convert anyway, or an explicit --bits to choose the target yourself."
    ),
}


def experimental_guard(
    plan: Any, decision: BitsDecision, allow_experimental: bool
) -> dict[str, str] | None:
    """Gate auto-derived targets with failing evidence.

    Returns an evidence marker to embed in the output record when the guard
    applies and ``--allow-experimental`` was passed; raises
    :class:`PlanError` when it applies and was not allowed; returns ``None``
    when it does not apply (explicit bits, other architectures/families).
    """
    if decision.requested != "auto":
        return None
    arch_id = plan.config.architecture.id
    for (arch, family), reason in EXPERIMENTAL_AUTO_TARGETS.items():
        if arch == arch_id and (decision.dominant_type or "").startswith(family):
            if allow_experimental:
                return {"experimental": True, "experimental_reason": reason}
            raise PlanError(
                "--bits auto would silently produce an experimental, "
                f"gate-failed configuration: {reason}"
            )
    return None
