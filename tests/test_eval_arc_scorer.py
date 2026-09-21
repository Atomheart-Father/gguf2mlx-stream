"""Unit tests for the ARC benchmark scorer's protocol validation."""

import importlib.util
import json
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent.parent / "eval" / "bench"
_spec = importlib.util.spec_from_file_location("score_arc", _BENCH / "score_arc.py")
sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa)


def _rec(qid: str, letter: str | None = "A", answer_key: str = "A",
         extra_text: bool = False, max_tokens: int = 64, temp: float = 0.0,
         prompt_sha: str = "abc") -> dict:
    return {
        "id": qid, "answer_key": answer_key, "prompt_sha256": prompt_sha,
        "raw_output": ("A" if letter else "no letter here 1 2 3") + (" more" if extra_text else ""),
        "letter": letter, "truncated": False, "repetition": False,
        "extra_text": extra_text, "max_tokens": max_tokens, "temp": temp,
        "gen_s": 0.5,
    }


def _write(tmp_path, name, recs) -> Path:
    p = tmp_path / f"{name}.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return p


def _two_sides(tmp_path, mutate_src=None, mutate_cand=None):
    base = [_rec(f"q{i}", prompt_sha=f"p{i}") for i in range(5)]
    src = [dict(r) for r in base]
    cand = [dict(r) for r in base]
    if mutate_src:
        mutate_src(src)
    if mutate_cand:
        mutate_cand(cand)
    return _write(tmp_path, "src", src), _write(tmp_path, "cand", cand)


def test_letter_only_requires_letter_and_no_extra(tmp_path):
    recs = [
        _rec("a", letter="B", extra_text=False),
        _rec("b", letter="C", extra_text=True),   # letter but extra text
        _rec("c", letter=None, extra_text=False),  # no letter at all
    ]
    p = _write(tmp_path, "s", recs)
    s = sa.score_side(p)
    assert s["letter_only_compliance"] == pytest.approx(1 / 3)


def test_duplicate_id_rejected(tmp_path):
    _src, cand = _two_sides(tmp_path, mutate_cand=lambda r: r.append(dict(r[0])))
    with pytest.raises(SystemExit, match="duplicate"):
        sa.load_records(cand, "cand")


def test_missing_question_rejected(tmp_path):
    src, cand = _two_sides(tmp_path, mutate_cand=lambda r: r.pop())
    a = sa.load_records(src, "src")
    b = sa.load_records(cand, "cand")
    with pytest.raises(SystemExit, match="coverage mismatch"):
        sa.assert_side_consistency({"src": a, "cand": b})


def test_answer_key_mismatch_rejected(tmp_path):
    src, cand = _two_sides(
        tmp_path,
        mutate_cand=lambda r: r[2].update(answer_key="C", letter="C"),
    )
    a, b = sa.load_records(src, "src"), sa.load_records(cand, "cand")
    with pytest.raises(SystemExit, match="answer key differs"):
        sa.assert_side_consistency({"src": a, "cand": b})


def test_prompt_hash_mismatch_rejected(tmp_path):
    src, cand = _two_sides(
        tmp_path, mutate_cand=lambda r: r[1].update(prompt_sha256="different"))
    a, b = sa.load_records(src, "src"), sa.load_records(cand, "cand")
    with pytest.raises(SystemExit, match="prompt hash differs"):
        sa.assert_side_consistency({"src": a, "cand": b})


def test_temp_mismatch_rejected(tmp_path):
    src, cand = _two_sides(tmp_path, mutate_cand=lambda r: [x.update(temp=0.7) for x in r])
    a, b = sa.load_records(src, "src"), sa.load_records(cand, "cand")
    with pytest.raises(SystemExit, match="temp differs"):
        sa.assert_side_consistency({"src": a, "cand": b})


def test_nonuniform_temp_within_side_rejected(tmp_path):
    def mixed(recs):
        recs[0]["temp"] = 0.5
    src, cand = _two_sides(tmp_path, mutate_cand=mixed)
    a, b = sa.load_records(src, "src"), sa.load_records(cand, "cand")
    with pytest.raises(SystemExit, match="not uniform"):
        sa.assert_side_consistency({"src": a, "cand": b})


def test_max_tokens_mismatch_rejected(tmp_path):
    src, cand = _two_sides(tmp_path, mutate_cand=lambda r: [x.update(max_tokens=32) for x in r])
    a, b = sa.load_records(src, "src"), sa.load_records(cand, "cand")
    with pytest.raises(SystemExit, match="max_tokens differs"):
        sa.assert_side_consistency({"src": a, "cand": b})


def test_consistent_sides_pass(tmp_path):
    src, cand = _two_sides(tmp_path)
    a, b = sa.load_records(src, "src"), sa.load_records(cand, "cand")
    max_tokens, temp = sa.assert_side_consistency({"src": a, "cand": b})
    assert (max_tokens, temp) == (64, 0.0)
