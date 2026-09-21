"""Unit tests for --bits auto selection (byte-weighted quant family mapping)."""

from types import SimpleNamespace

import pytest
from gguf import GGMLQuantizationType as QT

from gguf2mlx_stream.config.schema import SliceSpec
from gguf2mlx_stream.constants import SUPPORTED_BITS
from gguf2mlx_stream.errors import PlanError
from gguf2mlx_stream.quant_select import (
    BitsDecision,
    auto_fidelity_warning,
    consumed_fraction,
    decide_target_bits,
    family_bits,
    quant_histogram,
)
from gguf2mlx_stream.source.gguf import TensorInfo

MiB = 2**20


def _info(name: str, qtype: QT, ne: tuple[int, ...], n_bytes: int) -> TensorInfo:
    n_elements = 1
    for d in ne:
        n_elements *= d
    return TensorInfo(
        name=name,
        ne=ne,
        qtype=qtype,
        n_elements=n_elements,
        n_bytes=n_bytes,
        data_offset=0,
    )


def _job(*slots, quantize: bool = True):
    """slots: (slot_name, tensor_name, rows | None, in_mem | None)."""
    return SimpleNamespace(
        quantize=quantize,
        slots=tuple((s, t, rows) for s, t, rows, _ in slots),
        slot_specs=tuple((s, t, in_mem) for s, t, _, in_mem in slots),
    )


# ---------------------------------------------------------------------------
# family mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "qtype,expected",
    [
        ("IQ2_XXS", 2),
        ("IQ2_S", 2),
        ("IQ3_XXS", 3),
        ("IQ3_S", 3),
        ("IQ3_M", 3),
        ("IQ4_XS", 4),
        ("IQ4_NL", 4),
        ("Q2_K", 2),
        ("Q3_K_M", 3),
        ("Q4_K", 4),
        ("Q4_0", 4),
        ("Q6_K", 6),
        ("Q8_0", 8),
        # no MLX affine equivalent
        ("Q5_K_M", None),
        ("Q5_1", None),
        ("IQ1_S", None),
        ("IQ1_M", None),
        ("TQ1_0", None),
        ("F16", None),
        ("BF16", None),
    ],
)
def test_family_bits(qtype, expected):
    assert family_bits(qtype) == expected


# ---------------------------------------------------------------------------
# histogram construction
# ---------------------------------------------------------------------------


def test_histogram_byte_weighted_and_partial_slots():
    infos = {
        "big_q4": _info("big_q4", QT.Q4_K, (64, 100), 40 * MiB),
        "small_iq3": _info("small_iq3", QT.IQ3_S, (64, 4), 3 * MiB),
        "split": _info("split", QT.Q6_K, (64, 10), 10 * MiB),
        "norm_f32": _info("norm_f32", QT.F32, (64,), 1 * MiB),
    }
    jobs = [
        _job(("x", "big_q4", None, None)),
        _job(("x", "small_iq3", None, None)),
        _job(("a", "split", (0, 4), None), ("b", "split", (4, 10), None)),
        _job(("x", "norm_f32", None, None)),  # fp32 norm stays fp32 in MLX too
        _job(("x", "big_q4", None, None), quantize=False),  # not quantized out
    ]
    hist, counts = quant_histogram(jobs, infos.get)
    assert hist == {"Q4_K": 40 * MiB, "IQ3_S": 3 * MiB, "Q6_K": 10 * MiB}
    assert counts == {"Q4_K": 1.0, "IQ3_S": 1.0, "Q6_K": 1.0}


def test_histogram_in_memory_axis_slice():
    infos = {"t": _info("t", QT.Q8_0, (64, 8), 8 * MiB)}
    jobs = [_job(("x", "t", None, SliceSpec(axis=0, lo=0, hi=2)))]
    hist, counts = quant_histogram(jobs, infos.get)
    assert hist == {"Q8_0": 2 * MiB}
    assert counts["Q8_0"] == pytest.approx(0.25)


def test_histogram_empty_when_no_quantized_jobs():
    jobs = [_job(("x", "t", None, None), quantize=False)]
    assert quant_histogram(jobs, lambda n: None) == ({}, {})


def test_consumed_fraction_rows_and_axis():
    t = _info("t", QT.Q4_K, (64, 10), 10 * MiB)  # hf_shape (10, 64), 10 rows
    assert consumed_fraction(t, None, None) == 1.0
    assert consumed_fraction(t, (0, 4), None) == pytest.approx(0.4)
    assert consumed_fraction(t, (0, 10), SliceSpec(axis=1, lo=0, hi=32)) == pytest.approx(0.5)
    assert consumed_fraction(t, (5, 10), SliceSpec(axis=1, lo=16, hi=48)) == pytest.approx(0.25)
    # clamped: out-of-range slices cannot exceed the tensor
    assert consumed_fraction(t, (0, 99), None) == 1.0
    assert consumed_fraction(t, None, SliceSpec(axis=0, lo=8, hi=99)) == 1.0


# ---------------------------------------------------------------------------
# decide_target_bits
# ---------------------------------------------------------------------------


def test_auto_iq3_dominant_selects_3():
    hist = {"IQ3_XXS": 700 * MiB, "Q4_K": 200 * MiB, "IQ2_S": 100 * MiB}
    d = decide_target_bits(hist, "auto")
    assert isinstance(d, BitsDecision)
    assert d.bits == 3
    assert d.dominant_type == "IQ3_XXS"
    assert d.requested == "auto"
    assert "70.0%" in d.reason
    assert d.histogram_bytes == hist


def test_auto_q4_dominant_selects_4():
    d = decide_target_bits({"Q4_K": 900 * MiB, "Q6_K": 100 * MiB}, "auto")
    assert d.bits == 4


def test_auto_q6_dominant_selects_6():
    d = decide_target_bits({"Q6_K": 950 * MiB, "Q8_0": 50 * MiB}, "auto")
    assert d.bits == 6


def test_auto_q8_dominant_selects_8():
    d = decide_target_bits({"Q8_0": 10 * MiB}, "auto")
    assert d.bits == 8


def test_auto_mixed_byte_weighting_beats_tensor_count():
    # one huge Q4 tensor outweighs many small IQ3 tensors
    hist = {"Q4_K": 600 * MiB, "IQ3_S": 400 * MiB}
    d = decide_target_bits(hist, "auto", counts={"Q4_K": 1.0, "IQ3_S": 20.0})
    assert d.bits == 4
    # and the byte weighting dominates in the other direction too
    d = decide_target_bits({"Q4_K": 300 * MiB, "IQ3_S": 700 * MiB}, "auto")
    assert d.bits == 3


def test_auto_unknown_dominant_type_fails_loudly():
    with pytest.raises(PlanError) as exc:
        decide_target_bits({"Q5_K_M": 500 * MiB, "IQ3_S": 400 * MiB}, "auto")
    msg = str(exc.value)
    assert "Q5_K_M" in msg
    assert "--bits" in msg
    assert "IQ3_S" in msg


def test_auto_completely_unknown_type_fails():
    with pytest.raises(PlanError):
        decide_target_bits({"MXFP4": 100 * MiB}, "auto")


def test_auto_empty_histogram_fails():
    with pytest.raises(PlanError):
        decide_target_bits({}, "auto")


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
def test_explicit_bits_always_wins(bits):
    d = decide_target_bits({"Q4_K": 100 * MiB}, bits)
    assert d.bits == bits
    assert d.requested == str(bits)
    assert "explicit" in d.reason
    # histogram is still recorded for the report trail
    assert d.histogram_bytes == {"Q4_K": 100 * MiB}


def test_explicit_bits_string_form():
    assert decide_target_bits({}, "6").bits == 6


def test_explicit_unsupported_bits_fails():
    for bad in (5, 1, 7, 0):
        with pytest.raises(PlanError):
            decide_target_bits({}, bad)


def test_no_quantize_marker():
    d = decide_target_bits({}, None)
    assert d.bits == 0
    assert d.requested == "none"
    assert "no-quantize" in d.reason


def test_bits_decision_record_shape():
    d = decide_target_bits({"IQ3_S": 3 * MiB}, "auto", counts={"IQ3_S": 2.0})
    rec = d.as_record()
    assert rec["target_bits"] == 3
    assert rec["dominant_source_type"] == "IQ3_S"
    assert rec["source_quant_histogram_bytes"] == {"IQ3_S": 3 * MiB}
    assert rec["source_quant_histogram_counts"] == {"IQ3_S": 2.0}
    assert "auto" in rec["reason"]


def test_supported_bits_constant_unchanged():
    assert SUPPORTED_BITS == (2, 3, 4, 6, 8)


# ---------------------------------------------------------------------------
# auto 3-bit fidelity warning (warning only — same-bit auto always converts)
# ---------------------------------------------------------------------------


def _decision(requested: str, bits: int, dominant: str | None):
    return BitsDecision(
        requested=requested, bits=bits, dominant_type=dominant,
        dominant_bits=bits if dominant else None, reason="test",
    )


def test_auto_3bit_warns_and_records():
    d = _decision("auto", 3, "IQ3_S")
    w = auto_fidelity_warning(d)
    assert w is not None
    # the warning states observed evidence and the recommendation, without
    # claiming that 3-bit fails for every model
    assert "paired-oracle" in w
    assert "MLX affine 3-bit" in w
    assert "GGUF->MLX transcoder" in w
    assert "4-bit or higher is recommended" in w
    assert "fails" not in w.lower()
    rec = d.as_record() | {"fidelity_warning": w}
    assert rec["fidelity_warning"] == w
    assert rec["target_bits"] == 3


def test_warning_covers_all_q3_family_names():
    for dominant in ("Q3_K_M", "IQ3_XXS", "IQ3_S", "Q3_K_S"):
        assert auto_fidelity_warning(_decision("auto", 3, dominant)) is not None


def test_warning_ignores_explicit_bits():
    assert auto_fidelity_warning(_decision("3", 3, "IQ3_S")) is None


def test_warning_ignores_other_bits():
    for bits in (2, 4, 6, 8):
        assert auto_fidelity_warning(_decision("auto", bits, "Q4_K")) is None


def test_warning_ignores_no_quantize():
    assert auto_fidelity_warning(_decision("none", 0, "IQ3_S")) is None
