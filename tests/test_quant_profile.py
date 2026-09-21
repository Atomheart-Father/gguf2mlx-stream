"""Tests for the opt-in --quant-profile overlay (mixed quantization research)."""

import json

import pytest

from gguf2mlx_stream.config.schema import arch_config_from_dict
from gguf2mlx_stream.constants import SUPPORTED_BITS
from gguf2mlx_stream.errors import ConversionError
from gguf2mlx_stream.quant_profile import (
    apply_quant_profile,
    load_quant_profile,
    resolve_default_bits,
)


def _minimal_config():
    return arch_config_from_dict(
        {
            "architecture": {"id": "tiny", "gguf_arch": "tiny"},
            "dims": {"hidden": "gguf:embedding_length"},
            "rules": [
                {"match": "w\\.embed", "dest": "lm_head.weight"},
                {"match": "w\\.attn", "dest": "model.layers.{n}.self_attn.q_proj.weight"},
                {"match": "w\\.down", "dest": "model.layers.{n}.mlp.down_proj.weight"},
                {"match": "w\\.copy", "dest": "copied.weight", "quantize": False},
                {"match": "junk\\.drop.*", "drop": True},
            ],
            "output": {"model_type": "tiny"},
        }
    )


def _write_profile(tmp_path, doc):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(doc))
    return path


# ---------------------------------------------------------------------------
# load_quant_profile validation
# ---------------------------------------------------------------------------


def test_load_happy_path(tmp_path):
    profile = load_quant_profile(
        _write_profile(
            tmp_path,
            {
                "name": "mixed_3_4_g64",
                "default": {"bits": 3, "group_size": 64},
                "rules": [{"match": "down_proj", "bits": 4}],
            },
        )
    )
    assert profile.name == "mixed_3_4_g64"
    assert profile.default_bits == 3
    assert profile.default_group_size == 64
    assert len(profile.overlays) == 1
    assert profile.overlays[0].bits == 4
    assert profile.overlays[0].group_size is None


def test_load_unknown_top_level_key_rejected(tmp_path):
    with pytest.raises(ConversionError, match="unknown keys"):
        load_quant_profile(_write_profile(tmp_path, {"name": "x", "evil": "code"}))


def test_load_unknown_rule_key_rejected(tmp_path):
    with pytest.raises(ConversionError, match="unknown keys"):
        load_quant_profile(
            _write_profile(
                tmp_path,
                {"name": "x", "rules": [{"match": "a", "bits": 4, "lambda": "boom"}]},
            )
        )


@pytest.mark.parametrize("bits", [1, 5, 7, 16, True, "4"])
def test_load_bad_bits_rejected(tmp_path, bits):
    doc = {"name": "x", "default": {"bits": bits}}
    with pytest.raises(ConversionError):
        load_quant_profile(_write_profile(tmp_path, doc))


def test_load_group_size_must_be_supported(tmp_path):
    with pytest.raises(ConversionError, match="group_size"):
        load_quant_profile(
            _write_profile(tmp_path, {"name": "x", "default": {"bits": 3, "group_size": 48}})
        )


def test_load_bad_regex_rejected(tmp_path):
    with pytest.raises(ConversionError, match="match"):
        load_quant_profile(
            _write_profile(tmp_path, {"name": "x", "rules": [{"match": "(unclosed", "bits": 4}]})
        )


def test_load_empty_spec_rejected(tmp_path):
    with pytest.raises(ConversionError, match="at least one"):
        load_quant_profile(_write_profile(tmp_path, {"name": "x", "rules": [{"match": "a"}]}))


def test_load_invalid_json_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ConversionError, match="not valid JSON"):
        load_quant_profile(path)


def test_load_missing_name_rejected(tmp_path):
    with pytest.raises(ConversionError, match="name"):
        load_quant_profile(_write_profile(tmp_path, {"rules": []}))


# ---------------------------------------------------------------------------
# apply_quant_profile
# ---------------------------------------------------------------------------


def test_apply_matches_dest_and_replaces_bits(tmp_path):
    profile = load_quant_profile(
        _write_profile(
            tmp_path,
            {
                "name": "p",
                "rules": [
                    {"match": "lm_head", "bits": 8},
                    {"match": "down_proj", "bits": 4, "group_size": 32},
                ],
            },
        )
    )
    cfg, apps, unmatched = apply_quant_profile(_minimal_config(), profile)
    by_dest = {r.dest: r for r in cfg.rules}
    assert by_dest["lm_head.weight"].bits == 8
    assert by_dest["model.layers.{n}.mlp.down_proj.weight"].bits == 4
    assert by_dest["model.layers.{n}.mlp.down_proj.weight"].group_size == 32
    # untouched rules keep conversion-level defaults (None overrides)
    assert by_dest["model.layers.{n}.self_attn.q_proj.weight"].bits is None
    assert len(apps) == 2
    assert unmatched == ()


def test_apply_first_match_wins(tmp_path):
    profile = load_quant_profile(
        _write_profile(
            tmp_path,
            {
                "name": "p",
                "rules": [
                    {"match": "q_proj", "bits": 6},
                    {"match": "self_attn", "bits": 8},
                ],
            },
        )
    )
    cfg, apps, _ = apply_quant_profile(_minimal_config(), profile)
    by_dest = {r.dest: r for r in cfg.rules}
    assert by_dest["model.layers.{n}.self_attn.q_proj.weight"].bits == 6
    assert len(apps) == 1


def test_apply_null_overlay_field_keeps_rule_value(tmp_path):
    cfg0 = arch_config_from_dict(
        {
            "architecture": {"id": "tiny", "gguf_arch": "tiny"},
            "dims": {"hidden": "gguf:embedding_length"},
            "rules": [
                {"match": "w\\.a", "dest": "a.weight", "bits": 4, "group_size": 64},
            ],
            "output": {"model_type": "tiny"},
        }
    )
    profile = load_quant_profile(
        _write_profile(tmp_path, {"name": "p", "rules": [{"match": "a\\.weight", "bits": 6}]})
    )
    cfg, _, _ = apply_quant_profile(cfg0, profile)
    rule = cfg.rules[0]
    assert rule.bits == 6
    assert rule.group_size == 64


def test_apply_skips_drop_and_non_quantized_rules(tmp_path):
    profile = load_quant_profile(
        _write_profile(
            tmp_path,
            {"name": "p", "rules": [{"match": "\\.weight$", "bits": 6}]},
        )
    )
    cfg, apps, _unmatched = apply_quant_profile(_minimal_config(), profile)
    by_dest = {r.dest: r for r in cfg.rules}
    drop_rule = next(r for r in cfg.rules if r.drop)
    assert drop_rule.bits is None
    assert by_dest["copied.weight"].quantize is False
    assert by_dest["copied.weight"].bits is None
    assert all(app.rule_display_name != drop_rule.display_name for app in apps)


def test_apply_reports_unmatched_patterns(tmp_path):
    profile = load_quant_profile(
        _write_profile(
            tmp_path,
            {"name": "p", "rules": [{"match": "no_such_tensor", "bits": 4}]},
        )
    )
    _, apps, unmatched = apply_quant_profile(_minimal_config(), profile)
    assert apps == []
    assert unmatched == ("no_such_tensor",)


def test_apply_keeps_original_config_unchanged(tmp_path):
    cfg0 = _minimal_config()
    profile = load_quant_profile(
        _write_profile(tmp_path, {"name": "p", "rules": [{"match": "lm_head", "bits": 8}]})
    )
    apply_quant_profile(cfg0, profile)
    assert all(r.bits is None for r in cfg0.rules)


def test_as_record_roundtrip(tmp_path):
    profile = load_quant_profile(
        _write_profile(
            tmp_path,
            {"name": "p", "default": {"bits": 3}, "rules": [{"match": "x", "bits": 4}]},
        )
    )
    record = profile.as_record()
    assert record["name"] == "p"
    assert record["default_bits"] == 3
    assert record["rules"] == [{"match": "x", "bits": 4, "group_size": None}]


# ---------------------------------------------------------------------------
# resolve_default_bits
# ---------------------------------------------------------------------------


def _profile_with_default(tmp_path, bits=3):
    return load_quant_profile(
        _write_profile(tmp_path, {"name": "p", "default": {"bits": bits}})
    )


def test_default_resolves_from_auto(tmp_path):
    profile = _profile_with_default(tmp_path, 3)
    assert resolve_default_bits("auto", profile) == 3


def test_explicit_bits_conflicts_with_default(tmp_path):
    profile = _profile_with_default(tmp_path, 3)
    with pytest.raises(ConversionError, match="conflicts"):
        resolve_default_bits(4, profile)


def test_no_default_passes_through(tmp_path):
    profile = load_quant_profile(
        _write_profile(tmp_path, {"name": "p", "rules": [{"match": "x", "bits": 4}]})
    )
    assert resolve_default_bits("auto", profile) == "auto"
    assert resolve_default_bits(6, profile) == 6


def test_none_profile_passes_through():
    assert resolve_default_bits("auto", None) == "auto"
    assert resolve_default_bits(8, None) == 8


def test_profile_bits_within_supported(tmp_path):
    profile = _profile_with_default(tmp_path, 6)
    assert resolve_default_bits("auto", profile) in SUPPORTED_BITS
