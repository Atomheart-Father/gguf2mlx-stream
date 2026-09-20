"""Config schema validation tests."""

import textwrap

import pytest

from gguf2mlx_stream.config.schema import arch_config_from_dict, load_arch_config
from gguf2mlx_stream.errors import ConfigError


def minimal_config(**overrides):
    raw = {
        "architecture": {"id": "tiny", "gguf_arch": "tiny"},
        "dims": {"hidden": "gguf:embedding_length"},
        "rules": [
            {"match": "w\\.weight", "dest": "out.weight"},
        ],
        "output": {"model_type": "tiny"},
    }
    raw.update(overrides)
    return raw


def test_valid_minimal():
    cfg = arch_config_from_dict(minimal_config())
    assert cfg.architecture.id == "tiny"
    assert len(cfg.rules) == 1
    assert cfg.output.model_type == "tiny"


def test_missing_sections():
    with pytest.raises(ConfigError):
        arch_config_from_dict({"architecture": {"id": "x"}})
    with pytest.raises(ConfigError):
        arch_config_from_dict({"rules": []})


def test_unknown_op_rejected():
    raw = minimal_config(
        rules=[{"match": "a", "dest": "b", "steps": [{"op": "no_such_op"}]}]
    )
    with pytest.raises(ConfigError, match="unknown operator"):
        arch_config_from_dict(raw)


def test_step_input_validation():
    raw = minimal_config(
        rules=[{"match": "a", "dest": "b", "steps": [{"op": "neg", "input": "ghost"}]}]
    )
    with pytest.raises(ConfigError, match="not declared"):
        arch_config_from_dict(raw)


def test_multi_input_requires_declared_inputs():
    raw = minimal_config(
        rules=[
            {
                "match": "a",
                "dest": "b",
                "inputs": {"x": {"slice": {"axis": 0, "lo": 0}}},
                "steps": [{"op": "concat", "inputs": ["x", "y"], "args": {"axis": 0}}],
            }
        ]
    )
    with pytest.raises(ConfigError, match="not declared"):
        arch_config_from_dict(raw)


def test_drop_rule_must_not_have_dest():
    raw = minimal_config(rules=[{"match": "a", "drop": True, "dest": "b"}])
    with pytest.raises(ConfigError):
        arch_config_from_dict(raw)


def test_non_drop_requires_dest():
    raw = minimal_config(rules=[{"match": "a"}])
    with pytest.raises(ConfigError):
        arch_config_from_dict(raw)


def test_invalid_regex_rejected():
    raw = minimal_config(rules=[{"match": "a(", "dest": "b"}])
    with pytest.raises(ConfigError, match="regex"):
        arch_config_from_dict(raw)


def test_dim_placeholders_are_valid_regex():
    raw = minimal_config(
        rules=[{"match": "blk\\.{n_layers}\\..*", "drop": True}]
    )
    arch_config_from_dict(raw)  # must not raise


def test_bad_dim_spec_rejected():
    raw = minimal_config(dims={"hidden": "1 + 2"})
    with pytest.raises(ConfigError, match="invalid scalar spec"):
        arch_config_from_dict(raw)


def test_arith_spec_accepted():
    raw = minimal_config(dims={"a": "gguf:x", "b": {"mul": [2, "a"]}})
    arch_config_from_dict(raw)


def test_bad_arith_rejected():
    raw = minimal_config(dims={"a": {"mod": [2, 3]}})
    with pytest.raises(ConfigError):
        arch_config_from_dict(raw)


def test_unknown_rule_and_output_keys():
    with pytest.raises(ConfigError, match="unknown"):
        arch_config_from_dict(minimal_config(rules=[{"match": "a", "dest": "b", "zap": 1}]))
    with pytest.raises(ConfigError, match="unknown output key"):
        arch_config_from_dict(minimal_config(output={"model_type": "t", "bogus": 1}))


def test_duplicate_rule_names():
    raw = minimal_config(
        rules=[
            {"name": "dup", "match": "a", "dest": "b"},
            {"name": "dup", "match": "c", "dest": "d"},
        ]
    )
    with pytest.raises(ConfigError, match="duplicate rule names"):
        arch_config_from_dict(raw)


def test_yaml_load_safe_and_file(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        textwrap.dedent(
            """
            architecture: {id: tiny, gguf_arch: tiny}
            dims: {hidden: 'gguf:embedding_length'}
            rules:
              - {match: 'w\\.weight', dest: 'out.weight'}
            output: {model_type: tiny}
            """
        )
    )
    cfg = load_arch_config(str(p))
    assert cfg.rules[0].dest == "out.weight"
    # YAML object-injection is rejected by safe_load
    p2 = tmp_path / "evil.yaml"
    p2.write_text(
        "!!python/object/apply:os.system ['echo hacked']\n"
    )
    with pytest.raises(ConfigError, match="YAML"):
        load_arch_config(str(p2))
