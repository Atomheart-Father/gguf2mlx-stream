"""Unit tests for the capability-eval scorer (eval/score_eval.py).

These tests import the scorer module directly from eval/ so the exact code
that produces the committed reports is what runs under pytest/CI.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
_spec = importlib.util.spec_from_file_location("eval_score", _EVAL_DIR / "score_eval.py")
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


def _rec(raw: str, truncated: bool = False, max_tokens: int = 512) -> dict:
    return {
        "raw_output": raw,
        "truncated_suspect": truncated,
        "max_tokens": max_tokens,
        "temp": 0.0,
    }


# ---------------------------------------------------------------------------
# recorded runtime parameters
# ---------------------------------------------------------------------------


def test_params_read_from_records_not_defaults():
    src = {"a": _rec("x", max_tokens=4096), "b": _rec("y", max_tokens=4096)}
    mlx = {"a": _rec("x", max_tokens=4096), "b": _rec("y", max_tokens=4096)}
    assert sc.assert_param_consistency(src, mlx) == (4096, 0.0)


def test_param_mismatch_refuses_to_score():
    src = {"a": _rec("x", max_tokens=4096)}
    mlx = {"a": _rec("x", max_tokens=1536)}
    with pytest.raises(SystemExit, match="mismatch"):
        sc.assert_param_consistency(src, mlx)


def test_param_mismatch_within_one_side_refuses():
    src = {"a": _rec("x", max_tokens=512), "b": _rec("y", max_tokens=4096)}
    mlx = {"a": _rec("x", max_tokens=512), "b": _rec("y", max_tokens=512)}
    with pytest.raises(SystemExit, match="mismatch"):
        sc.assert_param_consistency(src, mlx)


# ---------------------------------------------------------------------------
# anomaly taxonomy (each type separate)
# ---------------------------------------------------------------------------


def test_truncation_flagged_from_record():
    # ascending numbers: long, not itself a loop -> only the truncation flag
    text = " ".join(str(i) for i in range(1, 300))
    _, _, blocked = sc.score_side(_rec(text, truncated=True), [])
    assert blocked == ["truncated"]


def test_truncated_and_looping_output_flags_both_types():
    _, _, blocked = sc.score_side(_rec("hello " * 400, truncated=True), [])
    assert "truncated" in blocked and "repetition_loop" in blocked


def test_repetition_loop_detected_tail_repeat():
    loop = "The sky is blue. " * 30
    assert sc.is_repetition_loop(loop)
    _, _, blocked = sc.score_side(_rec(loop), [])
    assert "repetition_loop" in blocked


def test_repetition_loop_detected_repeated_segment():
    text = ("Once upon a time there was a small village. " * 3
            + "and then everything was fine and dandy forever")
    assert not sc.is_repetition_loop(text)
    looped = text + " the same sentence over and over ends badly " * 6
    assert sc.is_repetition_loop(looped)


def test_repetition_loop_detected_repeated_lines():
    lines = ["line one of output", "line two of output",
             "repeat me please now"] + ["repeat me please now"] * 4
    assert sc.is_repetition_loop("\n".join(lines))


def test_normal_text_not_flagged_as_loop():
    text = ("Paris is the capital of France. It sits on the Seine river, "
            "and the city is known for the Eiffel Tower, museums, and cafes "
            "across twenty arrondissements with distinct characters.")
    assert not sc.is_repetition_loop(text)
    _, _, blocked = sc.score_side(_rec(text), [])
    assert blocked == []


def test_empty_and_garbled_separate_types():
    empty_final = _rec("<think>only thinking, no answer ever</think>")
    _, _, blocked = sc.score_side(empty_final, [])
    assert blocked == ["empty_output"]
    garbled = _rec("answer \ufffd\ufffd\ufffd text here with enough length "
                   "to pass the short-output guard rails of the scorer")
    flags = sc.detect_anomalies(garbled, sc.extract_final(garbled["raw_output"]))
    assert "garbled" in flags or "replacement_chars" in flags


# ---------------------------------------------------------------------------
# the gate: keyword inside a looping/truncated output is NOT a success
# ---------------------------------------------------------------------------

CHECKS_CONTAINS = [{"type": "contains", "patterns": ["巴黎"]}]


def test_keyword_inside_loop_is_not_success():
    loop = ("巴黎是法国的首都，也是法国的首都，巴黎是法国的首都，" * 30)
    correct, _, blocked = sc.score_side(_rec(loop), CHECKS_CONTAINS)
    assert "repetition_loop" in blocked
    assert correct is False


def test_truncated_keyword_output_is_not_success():
    raw = "巴黎是法国的首都，" + "补充说明补充说明" * 500
    correct, _, blocked = sc.score_side(_rec(raw, truncated=True), CHECKS_CONTAINS)
    assert "truncated" in blocked
    assert correct is False


def test_clean_keyword_output_is_success():
    correct, _, blocked = sc.score_side(
        _rec("巴黎是法国的首都。"), CHECKS_CONTAINS)
    assert blocked == []
    assert correct is True


# ---------------------------------------------------------------------------
# strict format constraints
# ---------------------------------------------------------------------------


def test_number_requires_exactly_one_number():
    check = [{"type": "number", "value": 42}]
    assert sc.eval_check(check[0], "42")
    assert not sc.eval_check(check[0], "The answer is 42, i.e. 42.")
    assert not sc.eval_check(check[0], "41, then 42")
    assert not sc.eval_check(check[0], "no number here")


def test_number_tolerates_units_and_signs():
    check = [{"type": "number", "value": 3.14}]
    assert sc.eval_check(check[0], "π ≈ 3.14")
    check = [{"type": "number", "value": -5}]
    assert sc.eval_check(check[0], "-5")


def test_exactly_three_colors_strict():
    check = [{"type": "all_colors"}]
    assert sc.eval_check(check[0], "red, green, blue")
    assert sc.eval_check(check[0], "红旗、绿树、蓝天")
    # a fourth color fails the "exactly three" constraint
    assert not sc.eval_check(check[0], "red, green, blue, and yellow")
    assert not sc.eval_check(check[0], "红色、绿色、蓝色、白色")
    # missing one of the three fails too
    assert not sc.eval_check(check[0], "red and green")


def test_one_word_strict():
    check = [{"type": "one_word", "patterns": ["Paris"]}]
    assert sc.eval_check(check[0], "Paris")
    assert sc.eval_check(check[0], "Paris.")
    assert not sc.eval_check(check[0], "Paris is a city")
    assert not sc.eval_check(check[0], "London")


def test_yes_no_strict():
    check = [{"type": "yes_no", "target": "no"}]
    assert sc.eval_check(check[0], "no")
    assert sc.eval_check(check[0], "No.")
    assert not sc.eval_check(check[0], "No, because the sun is a star")
    check = [{"type": "yes_no", "target": "yes"}]
    assert sc.eval_check(check[0], "yes")
    assert not sc.eval_check(check[0], "no")


def test_max_chars_strict():
    check = [{"type": "max_chars", "max_chars": 8, "patterns": ["北京"]}]
    assert sc.eval_check(check[0], "北京")
    assert not sc.eval_check(check[0], "北京是中国的首都，也是文化中心")


# ---------------------------------------------------------------------------
# report whitespace hygiene
# ---------------------------------------------------------------------------


def test_report_output_whitespace_stripped():
    dirty = "line one  \nline two\t\nline three"
    cleaned = sc._strip_trailing_ws(dirty)
    assert cleaned == "line one\nline two\nline three"
    assert not any(ln.endswith((" ", "\t")) for ln in cleaned.splitlines())


def test_report_markdown_has_no_trailing_whitespace(tmp_path):
    """The generated report.md must not carry trailing whitespace lines."""
    src_jsonl = tmp_path / "src.jsonl"
    mlx_jsonl = tmp_path / "mlx.jsonl"
    for path in (src_jsonl, mlx_jsonl):
        records = [
            {"id": "q1", "raw_output": "the answer  \nwith trailing ws  ",
             "truncated_suspect": False, "max_tokens": 64, "temp": 0.0,
             "gen_s": 1.0},
            {"id": "q2", "raw_output": "2", "truncated_suspect": False,
             "max_tokens": 64, "temp": 0.0, "gen_s": 1.0},
        ]
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({
        "max_tokens": 999,  # deliberately wrong: scorer must use recorded 64
        "questions": [
            {"id": "q1", "lang": "en", "category": "cat",
             "prompt": "p", "answer": "a",
             "checks": [{"type": "contains", "patterns": ["answer"]}]},
            {"id": "q2", "lang": "en", "category": "cat",
             "prompt": "p", "answer": "2",
             "checks": [{"type": "number", "value": 2}]},
        ],
    }))
    out = tmp_path / "report"
    argv = ["score", "--questions", str(questions), "--source", str(src_jsonl),
            "--mlx", str(mlx_jsonl), "--out-dir", str(out), "--title", "t",
            "--source-desc", "s", "--mlx-desc", "m"]
    monkey = pytest.MonkeyPatch()
    monkey.setattr(sys, "argv", argv)
    try:
        assert sc.main() == 0
    finally:
        monkey.undo()
    md = (out / "report.md").read_text()
    for line in md.splitlines():
        assert line == line.rstrip(), f"trailing whitespace in report line: {line!r}"
    data = json.loads((out / "report.json").read_text())
    assert data["summary"]["recorded_max_tokens"] == 64
    assert "max_tokens 64" in md


# ---------------------------------------------------------------------------
# zh-cs-02 equivalent-answer check + offline rescoring
# ---------------------------------------------------------------------------


_COMMITTED_QUESTIONS = (
    Path(__file__).resolve().parent.parent / "eval" / "questions.json"
)


def _zh_cs_02_checks() -> list[dict]:
    data = json.loads(_COMMITTED_QUESTIONS.read_text())
    return next(q["checks"] for q in data["questions"] if q["id"] == "zh-cs-02")


def test_committed_zh_cs_02_accepts_equivalent_answers():
    """The month question must accept both "12 个月" and "十二个月" styles.

    Regression for the asymmetric verdict found in the Nyx/JoyFox reports:
    the source-style answer lists every month (multiple numbers, so the
    strict ``number`` check fails) and must pass via the equivalent-answer
    contains patterns.
    """
    checks = _zh_cs_02_checks()
    assert sc.is_correct(checks, "一年有 **12 个月**。\n1 月、2 月、…、12 月。")
    assert sc.is_correct(checks, "一年有12个月。农历一年可能有12或13个月。")
    assert sc.is_correct(checks, "一年有十二个月。")
    assert not sc.is_correct(checks, "一年有 13 个月。")
    assert not sc.is_correct(checks, "一年有 11 个月。")


def _en_cs_02_checks() -> list[dict]:
    data = json.loads(_COMMITTED_QUESTIONS.read_text())
    return next(q["checks"] for q in data["questions"] if q["id"] == "en-cs-02")


def test_committed_en_cs_02_accepts_equivalent_answers():
    """The leap-year question must accept an explanatory correct answer.

    Regression for the same asymmetry as zh-cs-02: the source answer
    mentions February's 29 and the usual 28 days (three numbers, strict
    ``number`` check fails) and must pass via the "366 days" pattern.
    """
    checks = _en_cs_02_checks()
    assert sc.is_correct(
        checks,
        "There are 366 days in a leap year. The extra day is added to "
        "February, making it 29 days long instead of the usual 28.")
    assert sc.is_correct(checks, "There are **366 days** in a leap year.")
    assert not sc.is_correct(checks, "A leap year has 365 days.")
    assert not sc.is_correct(checks, "It takes 365.24 days to orbit the Sun.")


def test_rescore_cli_recomputes_verdicts_and_records_provenance(tmp_path):
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"max_tokens": 64, "questions": [
        {"id": "q1", "lang": "zh", "category": "cat", "prompt": "p", "answer": "12",
         "checks": [{"type": "number", "value": 12},
                    {"type": "contains", "patterns": ["十二", "12个月"]}]},
    ]}))
    stored = {
        "title": "t",
        "summary": {"recorded_max_tokens": 64, "recorded_temp": 0.0,
                    "note": "original note", "source_desc": "s", "mlx_desc": "m",
                    "source_accuracy": 0.0, "mlx_accuracy": 1.0,
                    "answer_agreement_rate": 0.0, "verdict_agreement_rate": 0.0,
                    "verdict_flips": ["q1"]},
        "per_question": [{
            "id": "q1", "lang": "zh", "category": "cat", "prompt": "p", "gold": "12",
            "source_output": "一年有 **12 个月**。1 月、2 月、…、12 月。",
            "mlx_output": "一年有12个月。",
            "source_correct": False, "mlx_correct": True,
            "source_blocked": [], "mlx_blocked": [],
            "source_anomalies": [], "mlx_anomalies": [],
            "agree": False, "source_gen_s": 1.0, "mlx_gen_s": 1.0,
            "source_eval_s": 0.1, "source_load_s": 0.1,
        }],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(stored))
    out = tmp_path / "out"
    argv = ["score", "--questions", str(questions),
            "--rescore-report", str(report_path), "--out-dir", str(out),
            "--note", "test reason"]
    monkey = pytest.MonkeyPatch()
    monkey.setattr(sys, "argv", argv)
    try:
        assert sc.main() == 0
    finally:
        monkey.undo()

    data = json.loads((out / "report.json").read_text())
    assert data["per_question"][0]["source_correct"] is True
    assert data["summary"]["source_accuracy"] == 1.0
    # generation-time fields are preserved, not recomputed
    assert data["summary"]["note"] == "original note"
    rescoring = data["summary"]["rescoring"]
    assert rescoring["reason"] == "test reason"
    assert rescoring["changed_verdicts"] == [
        {"id": "q1", "source": [False, True], "mlx": [True, True]}]
    assert rescoring["previous"]["source_accuracy"] == 0.0
    md = (out / "report.md").read_text()
    assert "test reason" in md
    assert "q1 (src False->True, mlx True->True)" in md


def test_rescore_cli_is_exclusive_with_jsonl_inputs(tmp_path):
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"questions": []}))
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"title": "t", "summary": {}, "per_question": []}))
    argv = ["score", "--questions", str(questions),
            "--rescore-report", str(report_path),
            "--source", str(tmp_path / "x.jsonl"),
            "--out-dir", str(tmp_path / "out")]
    monkey = pytest.MonkeyPatch()
    monkey.setattr(sys, "argv", argv)
    try:
        with pytest.raises(SystemExit, match="exclusive"):
            sc.main()
    finally:
        monkey.undo()
