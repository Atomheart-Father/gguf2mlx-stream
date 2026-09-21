"""Planner tests: matching, templating, conflicts, drops, coverage, dims."""

import numpy as np
import pytest

from gguf2mlx_stream.config.schema import arch_config_from_dict
from gguf2mlx_stream.errors import PlanError
from gguf2mlx_stream.planner import plan_conversion
from gguf2mlx_stream.source.gguf import GGUFSource

from conftest import write_gguf


def base_config(**over):
    raw = {
        "architecture": {"id": "tiny", "gguf_arch": "tiny"},
        "dims": {
            "n_layers": "gguf:block_count",
            "hidden": "gguf:embedding_length",
        },
        "rules": [
            {"match": "blk\\.{n_layers}\\..*", "drop": True},
            {"match": "tok\\.weight", "dest": "model.tok.weight"},
            {
                "match": "blk\\.(?P<n>\\d+)\\.attn\\.weight",
                "dest": "model.layers.{n}.attn.weight",
            },
            {
                "match": "blk\\.(?P<n>\\d+)\\.norm\\.weight",
                "dest": "model.layers.{n}.norm.weight",
                "quantize": False,
                "dtype": "float32",
                "expect_shape": ["hidden"],
            },
        ],
        "coverage": {
            "per_layer_required": ["model.layers.{layer}.norm.weight"],
            "per_layer_alternatives": [["model.layers.{layer}.attn.weight"]],
        },
    }
    raw.update(over)
    return raw


def make_source(tmp_path, n_layers=2):
    tensors = [
        ("tok.weight", np.zeros((8, 4), np.float32)),
    ]
    for i in range(n_layers):
        tensors.append((f"blk.{i}.attn.weight", np.zeros((4, 4), np.float32)))
        tensors.append((f"blk.{i}.norm.weight", np.zeros((4,), np.float32)))
    tensors.append((f"blk.{n_layers}.mtp.weight", np.zeros((4,), np.float32)))
    return GGUFSource(
        write_gguf(
            tmp_path / "m.gguf",
            arch="tiny",
            metadata={"block_count": n_layers, "embedding_length": 4},
            f32_tensors=tensors,
        )
    )


def test_plan_basic(tmp_path):
    src = make_source(tmp_path)
    cfg = arch_config_from_dict(base_config())
    plan = plan_conversion(cfg, src)
    dests = {j.dest for j in plan.jobs}
    assert "model.tok.weight" in dests
    assert "model.layers.0.attn.weight" in dests
    assert "model.layers.1.norm.weight" in dests
    assert not any("layers.2" in d for d in dests)  # MTP dropped
    assert len(plan.dropped) == 1
    assert plan.unmatched == ()
    assert plan.layer_count == 2


def test_plan_mtp_must_be_explicit(tmp_path):
    src = make_source(tmp_path)
    cfg_raw = base_config()
    cfg_raw["rules"] = [r for r in cfg_raw["rules"] if not r.get("drop")]
    with pytest.raises(PlanError, match="blk\\.2\\.mtp\\.weight"):
        # without the drop rule the MTP tensor matches nothing -> hard error
        plan_conversion(arch_config_from_dict(cfg_raw), src)


def test_plan_dest_conflict(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        rules=[
            {"match": "blk\\.{n_layers}\\..*", "drop": True},
            {"match": "tok\\.weight", "dest": "same.weight"},
            {"match": "blk\\.(?P<n>\\d+)\\.attn\\.weight", "dest": "same.weight"},
        ],
        coverage={},
    )
    with pytest.raises(PlanError, match="conflict"):
        plan_conversion(arch_config_from_dict(raw), src)


def test_plan_unmatched_error_policy(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        rules=[{"match": "tok\\.weight", "dest": "t.weight"}],
        coverage={},
    )
    with pytest.raises(PlanError, match="matched no rule"):
        plan_conversion(arch_config_from_dict(raw), src)


def test_plan_unmatched_warn_policy(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        rules=[{"match": "tok\\.weight", "dest": "t.weight"}],
        coverage={},
        unmatched_tensors="warn",
    )
    plan = plan_conversion(arch_config_from_dict(raw), src)
    assert len(plan.unmatched) == 5


def test_plan_unused_required_rule_fails(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        rules=[{"match": "never\\.matches", "dest": "x.weight", "optional": False}],
        coverage={},
    )
    with pytest.raises(PlanError, match="matched no tensor"):
        plan_conversion(arch_config_from_dict(raw), src)


def test_plan_optional_rule_ok(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        unmatched_tensors="warn",
        rules=[{"match": "never\\.matches", "dest": "x.weight", "optional": True}],
        coverage={},
    )
    plan = plan_conversion(arch_config_from_dict(raw), src)
    assert plan.jobs == ()  # optional rule matched nothing, nothing else declared


def test_plan_shape_mismatch(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        rules=[
            {"match": "tok\\.weight", "dest": "t.weight", "expect_shape": [999, 999]},
        ],
        coverage={},
    )
    with pytest.raises(PlanError, match="shape mismatch"):
        plan_conversion(arch_config_from_dict(raw), src)


def test_plan_coverage_missing(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        coverage={"per_layer_required": ["model.layers.{layer}.missing.weight"]},
    )
    with pytest.raises(PlanError, match="coverage"):
        plan_conversion(arch_config_from_dict(raw), src)


def test_plan_coverage_alternatives(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        coverage={
            "per_layer_alternatives": [
                ["model.layers.{layer}.attn.weight", "model.layers.{layer}.ghost.weight"]
            ]
        },
    )
    plan = plan_conversion(arch_config_from_dict(raw), src)
    assert plan.jobs


def test_dims_fallback_chain_and_arith(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        unmatched_tensors="warn",
        dims={
            "n_layers": ["gguf:not_there", "gguf:block_count"],
            "hidden": "gguf:embedding_length",
            "double_hidden": {"mul": [2, "hidden"]},
        },
        rules=[
            {
                "match": "tok\\.weight",
                "dest": "t.weight",
                "expect_shape": ["vocab", "hidden"],
            },
        ],
        coverage={},
    )
    raw["dims"]["vocab"] = {"mul": [2, "hidden"]}
    plan = plan_conversion(arch_config_from_dict(raw), src)
    assert plan.dims["double_hidden"] == 8
    assert plan.dims["vocab"] == 8
    assert len(plan.jobs) == 1  # only the matching expect_shape survives validation


def test_step_args_literal_list_spec(tmp_path):
    """{list: [...]} resolves to a literal list, not a fallback chain."""
    src = make_source(tmp_path)
    raw = base_config(
        unmatched_tensors="warn",
        dims={"hidden": "gguf:embedding_length", "vocab": {"mul": [2, "hidden"]},
              "half": {"div": ["vocab", 2]}},
        rules=[
            {
                "match": "tok\\.weight",
                "dest": "t.weight",
                "steps": [
                    {"op": "reshape",
                     "args": {"shape": {"list": [2, "half", "half"]}}},
                    {"op": "permute",
                     "args": {"axes": {"list": [1, 0, 2]}}},
                    {"op": "reshape",
                     "args": {"shape": {"list": ["vocab", "hidden"]}}},
                ],
            },
        ],
        coverage={},
    )
    plan = plan_conversion(arch_config_from_dict(raw), src)
    assert plan.dims["half"] == 4
    (job,) = plan.jobs
    assert job.steps[0].args["shape"] == [2, 4, 4]
    assert job.steps[1].args["axes"] == [1, 0, 2]
    assert job.steps[2].args["shape"] == [8, 4]


def test_slot_slices_resolved(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        unmatched_tensors="warn",
        dims={
            "n_layers": "gguf:block_count",
            "hidden": "gguf:embedding_length",
            "half": {"div": ["hidden", 2]},
        },
        rules=[
            {
                "match": "tok\\.weight",
                "dest": "t.weight",
                "inputs": {
                    "a": {"slice": {"axis": 0, "lo": 0, "hi": "half"}},
                    "b": {"slice": {"axis": 0, "lo": "half", "hi": "hidden"}},
                },
                "steps": [
                    {"op": "concat", "inputs": ["a", "b"], "args": {"axis": 0}}
                ],
            }
        ],
        coverage={},
    )
    plan = plan_conversion(arch_config_from_dict(raw), src)
    job = plan.jobs[0]
    assert job.slots[0][2] == (0, 2)
    assert job.slots[1][2] == (2, 4)
    assert not job.chunkable  # slots present


def test_arch_mismatch_strict(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(architecture={"id": "tiny", "gguf_arch": "other", "strict_arch": True})
    with pytest.raises(PlanError, match="general.architecture"):
        plan_conversion(arch_config_from_dict(raw), src)


def test_arch_mismatch_lenient(tmp_path, capsys):
    src = make_source(tmp_path)
    raw = base_config(architecture={"id": "tiny", "gguf_arch": "other"})
    plan = plan_conversion(arch_config_from_dict(raw), src)
    assert plan.jobs
    assert "WARNING" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# declarative range drops (P1): no hardcoded block-name prefix
# ---------------------------------------------------------------------------

def test_range_drop_names_declared_by_match_template(tmp_path):
    """A range drop rule matches exactly what its {i} match template
    declares — any name pattern, not a hardcoded blk.{i}. prefix."""
    tensors = [("tok.weight", np.zeros((8, 4), np.float32))]
    for i in range(2):
        tensors.append((f"mtp.{i}.hidden.weight", np.zeros((4,), np.float32)))
        tensors.append((f"mtp.{i}.ffn.weight", np.zeros((4, 4), np.float32)))
    src = GGUFSource(write_gguf(
        tmp_path / "m.gguf", arch="tiny",
        metadata={"block_count": 2, "embedding_length": 4},
        f32_tensors=tensors,
    ))
    raw = base_config(
        rules=[
            {"match": "mtp\\.{i}\\..*", "drop": True,
             "range": {"start": 0, "end": 2}},
            {"match": "tok\\.weight", "dest": "model.tok.weight"},
        ],
        coverage={},
    )
    plan = plan_conversion(arch_config_from_dict(raw), src)
    dropped = [name for name, _ in plan.dropped]
    assert dropped == ["mtp.0.hidden.weight", "mtp.0.ffn.weight",
                       "mtp.1.hidden.weight", "mtp.1.ffn.weight"]
    assert {j.dest for j in plan.jobs} == {"model.tok.weight"}


def test_range_drop_respects_index_bounds(tmp_path):
    tensors = [("tok.weight", np.zeros((8, 4), np.float32))]
    for i in range(3):
        tensors.append((f"mtp.{i}.ffn.weight", np.zeros((4, 4), np.float32)))
    src = GGUFSource(write_gguf(
        tmp_path / "m.gguf", arch="tiny",
        metadata={"block_count": 3, "embedding_length": 4},
        f32_tensors=tensors,
    ))
    raw = base_config(
        unmatched_tensors="warn",
        rules=[
            # only block index 1 is inside [1, 2)
            {"match": "mtp\\.{i}\\..*", "drop": True,
             "range": {"start": 1, "end": 2}},
            {"match": "tok\\.weight", "dest": "model.tok.weight"},
        ],
        coverage={},
    )
    plan = plan_conversion(arch_config_from_dict(raw), src)
    assert [name for name, _ in plan.dropped] == ["mtp.1.ffn.weight"]
    assert plan.unmatched == ("mtp.0.ffn.weight", "mtp.2.ffn.weight")


# ---------------------------------------------------------------------------
# plan-time pipeline shape validation (P1)
# ---------------------------------------------------------------------------

def test_invalid_reshape_fails_at_plan_time(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        unmatched_tensors="warn",
        rules=[{
            "match": "tok\\.weight",
            "dest": "t.weight",
            "steps": [{"op": "reshape", "args": {"shape": {"list": [100, 4]}}}],
        }],
        coverage={},
    )
    with pytest.raises(PlanError, match="reshape"):
        plan_conversion(arch_config_from_dict(raw), src)


def test_invalid_permute_args_fail_at_plan_time(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        unmatched_tensors="warn",
        rules=[{
            "match": "tok\\.weight",
            "dest": "t.weight",
            "steps": [
                {"op": "reshape", "args": {"shape": {"list": [2, 2, 2, 4]}}},
                {"op": "permute", "args": {"axes": {"list": [0, 1, 2, 9]}}},
            ],
        }],
        coverage={},
    )
    with pytest.raises(PlanError, match="permute|axes"):
        plan_conversion(arch_config_from_dict(raw), src)


def test_invalid_unsqueeze_axis_fails_at_plan_time(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        unmatched_tensors="warn",
        rules=[{
            "match": "tok\\.weight",
            "dest": "t.weight",
            "steps": [{"op": "unsqueeze", "args": {"axis": 99}}],
        }],
        coverage={},
    )
    with pytest.raises(PlanError, match="unsqueeze"):
        plan_conversion(arch_config_from_dict(raw), src)


def test_incompatible_concat_fails_at_plan_time(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        unmatched_tensors="warn",
        rules=[{
            "match": "tok\\.weight",
            "dest": "t.weight",
            "inputs": {
                "a": {"slice": {"axis": 0, "lo": 0, "hi": 2}},
                "b": {"slice": {"axis": 0, "lo": 2, "hi": 4}},
            },
            "steps": [{"op": "concat", "inputs": ["a", "b"], "args": {"axis": 5}}],
        }],
        coverage={},
    )
    with pytest.raises(PlanError, match="concat|axis"):
        plan_conversion(arch_config_from_dict(raw), src)


def test_quantized_rule_producing_rank1_fails_at_plan_time(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        unmatched_tensors="warn",
        rules=[{
            "match": "tok\\.weight",
            "dest": "t.weight",
            "steps": [{"op": "reshape", "args": {"shape": {"list": [16]}}}],
        }],
        coverage={},
    )
    with pytest.raises(PlanError, match="rank >= 2"):
        plan_conversion(arch_config_from_dict(raw), src)


def test_quantized_rule_allows_nd_output(tmp_path):
    """N-D quantization semantics: 3-D output is quantized along its last axis."""
    src = make_source(tmp_path)
    raw = base_config(
        unmatched_tensors="warn",
        rules=[{
            "match": "tok\\.weight",
            "dest": "t.weight",
            "steps": [{"op": "reshape", "args": {"shape": {"list": [2, 2, 4]}}}],
        }],
        coverage={},
    )
    plan = plan_conversion(arch_config_from_dict(raw), src)
    (job,) = plan.jobs
    assert job.quantize and job.out_shape == (2, 2, 4)


def test_quantized_rule_group_override_validated_at_plan_time(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        unmatched_tensors="warn",
        rules=[{
            "match": "tok\\.weight",
            "dest": "t.weight",
            "group_size": 64,
        }],
        coverage={},
    )
    with pytest.raises(PlanError, match="group_size"):
        plan_conversion(arch_config_from_dict(raw), src)


def test_valid_pipeline_shape_records_out_shape(tmp_path):
    src = make_source(tmp_path)
    raw = base_config(
        unmatched_tensors="warn",
        rules=[{
            "match": "tok\\.weight",
            "dest": "t.weight",
            "quantize": False,
            "steps": [
                {"op": "reshape", "args": {"shape": {"list": [2, "half", "half"]}}},
                {"op": "permute", "args": {"axes": {"list": [1, 0, 2]}}},
                {"op": "reshape", "args": {"shape": {"list": ["vocab", "hidden"]}}},
            ],
        }],
        coverage={},
        dims={
            "hidden": "gguf:embedding_length",
            "vocab": {"mul": [2, "hidden"]},
            "half": {"div": ["vocab", 2]},
        },
    )
    plan = plan_conversion(arch_config_from_dict(raw), src)
    (job,) = plan.jobs
    assert job.out_shape == (8, 4)
