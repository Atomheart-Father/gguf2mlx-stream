"""Declarative architecture config: dataclasses, parsing, and schema validation.

A config is *data*, not a program. The YAML grammar is deliberately closed:

* tensor matching: anchored regular expressions with named groups;
* dimension arithmetic: a fixed vocabulary of ``mul``/``add``/``sub``/``div``
  over integers and named dimensions (no free-form expressions, no eval);
* transforms: ordered lists of registered operator names with constant args;
* no imports, no shell, no code execution — ``yaml.safe_load`` plus the
  checks below enforce that.

Static (source-independent) validation happens here at load time.
Source-dependent resolution (GGUF metadata / reference config lookups)
happens in the planner, right before a conversion plan is compiled.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Mapping

import yaml

from ..errors import ConfigError
from ..ops import all_ops

# A scalar value spec: int | float | bool | None | str | list | {mul/add/sub/div/not: [...]}
# String forms: "gguf:<key>" (GGUF metadata, arch-prefixed fallback),
# "ref:<dotted.path>" (reference config), "tshape:<tensor>.<axis>" (source
# tensor shape), "has:<tensor>" (tensor existence), a named dimension, or a
# plain constant.
_ARITH_OPS = ("mul", "add", "sub", "div", "not")


def _err(where: str, msg: str) -> ConfigError:
    return ConfigError(f"{where}: {msg}")


# ---------------------------------------------------------------------------
# scalar value specs
# ---------------------------------------------------------------------------


def _validate_scalar_spec(v: Any, where: str, allow_mapping: bool = False) -> None:
    if v is None or isinstance(v, (bool, int, float)):
        return
    if isinstance(v, str):
        if v.startswith(("gguf:", "ref:", "tshape:", "has:")):
            if len(v) <= 5:
                raise _err(where, f"empty reference {v!r}")
            return
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v):
            return  # named dimension reference
        raise _err(
            where,
            f"invalid scalar spec {v!r}: use an int, 'gguf:<key>', 'ref:<path>', "
            "a dimension name, or an {mul/add/sub/div: [...]} form",
        )
    if isinstance(v, list):
        for i, item in enumerate(v):
            _validate_scalar_spec(item, f"{where}[{i}]", allow_mapping)
        return
    if isinstance(v, dict):
        # explicit literal-list form: {list: [<scalar spec>, ...]}
        if set(v) == {"list"}:
            items = v["list"]
            if not isinstance(items, list) or not items:
                raise _err(where, "{list: [...]} requires a non-empty list of scalar specs")
            _validate_scalar_spec(items, where, allow_mapping)
            return
        if len(v) == 1 and next(iter(v)) in _ARITH_OPS:
            operands = next(iter(v.values()))
            if not isinstance(operands, list) or len(operands) < 1:
                raise _err(where, "arithmetic operands must be a non-empty list")
            _validate_scalar_spec(operands, where, allow_mapping)
            return
        # plain nested mapping (e.g. rope_parameters, rope_scaling): every
        # value must itself be a scalar spec; arithmetic keys stay reserved.
        for k, item in v.items():
            if k in _ARITH_OPS:
                raise _err(
                    f"{where}.{k}",
                    "arithmetic keys are reserved; use exactly one "
                    "mul/add/sub/div/not key with a list of operands",
                )
            _validate_scalar_spec(item, f"{where}.{k}", allow_mapping)
        return
    raise _err(where, f"unsupported value spec type {type(v).__name__}")


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SliceSpec:
    axis: int
    lo: Any = 0
    hi: Any = None  # None = end of axis


@dataclasses.dataclass(frozen=True)
class InputSlot:
    source: str = "@"  # "@" refers to the tensor captured by the rule's match
    slice: SliceSpec | None = None


@dataclasses.dataclass(frozen=True)
class OpStep:
    op: str
    input: str | None = None  # single-input name (default: previous output)
    inputs: tuple[str, ...] = ()  # multi-input names, in order
    args: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    output: str | None = None  # optional name binding the result for later steps


@dataclasses.dataclass(frozen=True)
class Rule:
    name: str
    match: str  # anchored regex; {dim} placeholders substituted at plan time
    dest: str | None = None
    drop: bool = False
    optional: bool = False
    quantize: bool = True
    dtype: str = "float32"
    expect_shape: tuple[Any | None, ...] | None = None
    inputs: Mapping[str, InputSlot] | None = None
    steps: tuple[OpStep, ...] = ()
    # drop rules may instead declare an explicit half-open block-index range
    # [start, end): the rule's match pattern (which must contain the reserved
    # {i} placeholder) declares how block tensor names are built — {i} is
    # substituted with each index in the range and the result must fullmatch
    # the tensor name. end <= start matches nothing (e.g. nextn=0).
    block_range: tuple[Any, Any] | None = None

    @property
    def display_name(self) -> str:
        return self.name or self.match


@dataclasses.dataclass(frozen=True)
class OutputSpec:
    model_type: str | None = None
    architectures: tuple[str, ...] = ()
    nest_config_under: str | None = None
    top_level: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    text_config: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    ref_drop_fields: tuple[str, ...] = ()
    tokenizer_files: tuple[str, ...] = ()
    max_shard_bytes: int = 4 * 2**30
    # dotted paths that must resolve to a non-null value in the final
    # config.json (e.g. "text_config.hidden_size"). Conversion fails loudly
    # instead of silently relying on mlx-lm defaults.
    required_fields: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Coverage:
    per_layer_required: tuple[str, ...] = ()
    per_layer_alternatives: tuple[tuple[str, ...], ...] = ()


@dataclasses.dataclass(frozen=True)
class Architecture:
    id: str
    aliases: tuple[str, ...]
    # accepted general.architecture identifier(s) in the source GGUF.
    # A tuple lists accepted variants (current llama.cpp identifier first).
    gguf_arch: str | tuple[str, ...] | None
    strict_arch: bool
    description: str

    def accepts(self, gguf_arch: str) -> bool:
        if self.gguf_arch is None:
            return True
        if isinstance(self.gguf_arch, str):
            return gguf_arch == self.gguf_arch
        return gguf_arch in self.gguf_arch


@dataclasses.dataclass(frozen=True)
class ArchConfig:
    path: str
    architecture: Architecture
    dims: Mapping[str, Any]  # insertion-ordered name -> scalar spec
    rules: tuple[Rule, ...]
    coverage: Coverage
    output: OutputSpec
    unmatched_policy: str  # "error" | "warn"

    def rules_index(self) -> Mapping[str, Rule]:
        return {r.name: r for r in self.rules if r.name}


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

_DIM_SUBST = "[0-9]+"  # placeholder used when validating match patterns


def _parse_slice(raw: Any, where: str) -> SliceSpec:
    if not isinstance(raw, dict):
        raise _err(where, "slice must be a mapping with axis/lo/hi")
    axis = raw.get("axis")
    if not isinstance(axis, int):
        raise _err(where, "slice.axis must be an int")
    for key in raw:
        if key not in ("axis", "lo", "hi"):
            raise _err(where, f"unknown slice key {key!r}")
    lo, hi = raw.get("lo", 0), raw.get("hi")
    _validate_scalar_spec(lo, f"{where}.lo")
    _validate_scalar_spec(hi, f"{where}.hi")
    return SliceSpec(axis=axis, lo=lo, hi=hi)


def _parse_inputs(raw: Any, where: str) -> dict[str, InputSlot]:
    if not isinstance(raw, dict) or not raw:
        raise _err(where, "inputs must be a non-empty mapping of name -> slot")
    slots: dict[str, InputSlot] = {}
    for name, slot in raw.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", str(name)):
            raise _err(where, f"invalid input slot name {name!r}")
        if slot == "@":
            slots[str(name)] = InputSlot()
            continue
        if not isinstance(slot, dict):
            raise _err(f"{where}.{name}", "input slot must be a mapping or '@'")
        source = slot.get("source", "@")
        if source != "@":
            raise _err(
                f"{where}.{name}",
                "input slot 'source' must be '@' (the matched tensor); "
                "cross-tensor sources are expressed via concat jobs",
            )
        sl = slot.get("slice")
        slots[str(name)] = InputSlot(
            source="@", slice=_parse_slice(sl, f"{where}.{name}.slice") if sl else None
        )
    return slots


def _parse_steps(raw: Any, where: str, inputs: Mapping[str, InputSlot] | None) -> tuple[OpStep, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise _err(where, "steps must be a list of operator steps")
    known = set(inputs or {}) | {"x", "_"}  # "_" refers to the previous step's output
    steps: list[OpStep] = []
    prev_single: str | None = "x" if not inputs else None
    for i, step in enumerate(raw):
        w = f"{where}[{i}]"
        if not isinstance(step, dict) or "op" not in step:
            raise _err(w, "step must be a mapping with an 'op' key")
        op = str(step["op"])
        if op not in all_ops():
            raise _err(w, f"unknown operator {op!r}")
        for key in step:
            if key not in ("op", "input", "inputs", "args", "output"):
                raise _err(w, f"unknown step key {key!r}")
        args = step.get("args", {})
        if not isinstance(args, dict):
            raise _err(w, "args must be a mapping")
        out_name = step.get("output")
        if out_name is not None:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", str(out_name)):
                raise _err(w, f"invalid step output name {out_name!r}")
            out_name = str(out_name)
        multi = tuple(str(x) for x in step.get("inputs", ()))
        single = step.get("input")
        if multi:
            for name in multi:
                if name not in known:
                    raise _err(w, f"input {name!r} not declared in rule inputs or step outputs")
            if single is not None:
                raise _err(w, "step cannot have both 'input' and 'inputs'")
            steps.append(OpStep(op=op, inputs=multi, args=args, output=out_name))
            prev_single = None
        else:
            if single is None:
                single = prev_single
                if single is None:
                    raise _err(
                        w,
                        "cannot infer input: previous step produced a multi-input "
                        "combination; set 'input: \"_\"' to consume it explicitly",
                    )
            if single not in known:
                raise _err(w, f"input {single!r} not declared in rule inputs or step outputs")
            steps.append(OpStep(op=op, input=str(single), args=args, output=out_name))
            prev_single = "_"
        if out_name is not None:
            known.add(out_name)
    return tuple(steps)


_RULE_KEYS = (
    "name", "match", "dest", "drop", "optional", "quantize", "dtype",
    "expect_shape", "inputs", "steps", "range",
)


def _parse_rule(raw: Any, idx: int) -> Rule:
    where = f"rules[{idx}]"
    if not isinstance(raw, dict):
        raise _err(where, "rule must be a mapping")
    for key in raw:
        if key not in _RULE_KEYS:
            raise _err(where, f"unknown rule key {key!r}")
    match = raw.get("match")
    if not isinstance(match, str) or not match:
        raise _err(where, "rule.match must be a non-empty regex string")
    try:
        re.compile(match.replace("{", "(").replace("}", ")") if "{" in match else match)
    except re.error as exc:
        # placeholders make it uncompilable here; validate after substituting dims
        try:
            re.compile(re.sub(r"\{[A-Za-z_][A-Za-z0-9_]*\}", _DIM_SUBST, match))
        except re.error:
            raise _err(where, f"invalid regex: {exc}") from None
    drop = bool(raw.get("drop", False))
    dest = raw.get("dest")
    if drop:
        if dest is not None:
            raise _err(where, "drop rules must not have 'dest'")
    else:
        if not isinstance(dest, str) or not dest:
            raise _err(where, "non-drop rules require a 'dest' template")
    if "dest" in raw and dest is not None and drop:
        raise _err(where, "drop rules must not have 'dest'")
    block_range = None
    if "range" in raw:
        if not drop:
            raise _err(where, "'range' is only valid on drop rules")
        rng = raw["range"]
        if not isinstance(rng, dict) or set(rng) != {"start", "end"}:
            raise _err(where, "range must be a mapping with exactly 'start' and 'end'")
        _validate_scalar_spec(rng["start"], f"{where}.range.start")
        _validate_scalar_spec(rng["end"], f"{where}.range.end")
        block_range = (rng["start"], rng["end"])
        if "{i}" not in match:
            raise _err(
                where,
                "range drop rules must use the reserved {i} placeholder in 'match'",
            )
    elif drop and "{i}" in match:
        raise _err(where, "{i} is reserved for range drop rules")
    name = raw.get("name")
    if name is not None and not isinstance(name, str):
        raise _err(where, "rule.name must be a string")
    inputs = raw.get("inputs")
    inputs_p = _parse_inputs(inputs, f"{where}.inputs") if inputs else None
    steps = _parse_steps(raw.get("steps"), f"{where}.steps", inputs_p)
    dtype = raw.get("dtype", "float32")
    if dtype not in ("float32", "float16"):
        raise _err(where, f"unsupported dtype {dtype!r} (use float32/float16)")
    expect_shape = raw.get("expect_shape")
    if expect_shape is not None:
        if not isinstance(expect_shape, list):
            raise _err(where, "expect_shape must be a list")
        for i, dim in enumerate(expect_shape):
            if dim is not None:
                _validate_scalar_spec(dim, f"{where}.expect_shape[{i}]")
        expect_shape = tuple(expect_shape)
    return Rule(
        name=name,
        match=match,
        dest=None if drop else dest,
        drop=drop,
        optional=bool(raw.get("optional", False)),
        quantize=bool(raw.get("quantize", True)),
        dtype=dtype,
        expect_shape=expect_shape,
        inputs=inputs_p,
        steps=steps,
        block_range=block_range,
    )


def _parse_output(raw: Any, where: str) -> OutputSpec:
    if raw is None:
        return OutputSpec()
    if not isinstance(raw, dict):
        raise _err(where, "output must be a mapping")
    for key in raw:
        if key not in (
            "model_type",
            "architectures",
            "nest_config_under",
            "top_level",
            "text_config",
            "reference_config",
            "tokenizer_files",
            "max_shard_bytes",
            "required_fields",
        ):
            raise _err(where, f"unknown output key {key!r}")
    archs = raw.get("architectures", [])
    if not isinstance(archs, list) or not all(isinstance(a, str) for a in archs):
        raise _err(f"{where}.architectures", "must be a list of strings")
    tc = raw.get("text_config", {})
    if not isinstance(tc, dict):
        raise _err(f"{where}.text_config", "must be a mapping")
    for k, v in tc.items():
        _validate_scalar_spec(v, f"{where}.text_config.{k}", allow_mapping=True)
    top = raw.get("top_level", {})
    if not isinstance(top, dict):
        raise _err(f"{where}.top_level", "must be a mapping")
    ref = raw.get("reference_config", {})
    if not isinstance(ref, dict):
        raise _err(f"{where}.reference_config", "must be a mapping")
    drops = tuple(ref.get("drop_fields", ()))
    if not all(isinstance(d, str) for d in drops):
        raise _err(f"{where}.reference_config.drop_fields", "must be a list of strings")
    tok_files = tuple(raw.get("tokenizer_files", ()))
    if not all(isinstance(f, str) for f in tok_files):
        raise _err(f"{where}.tokenizer_files", "must be a list of strings")
    msb = int(raw.get("max_shard_bytes", 4 * 2**30))
    if msb <= 0:
        raise _err(f"{where}.max_shard_bytes", "must be positive")
    req_fields = raw.get("required_fields", [])
    if not isinstance(req_fields, list) or not all(
        isinstance(f, str) and f for f in req_fields
    ):
        raise _err(
            f"{where}.required_fields",
            "must be a list of non-empty dotted config.json paths",
        )
    return OutputSpec(
        model_type=raw.get("model_type"),
        architectures=tuple(archs),
        nest_config_under=raw.get("nest_config_under"),
        top_level=dict(top),
        text_config=dict(tc),
        ref_drop_fields=drops,
        tokenizer_files=tok_files,
        max_shard_bytes=msb,
        required_fields=tuple(req_fields),
    )


def arch_config_from_dict(raw: Any, path: str = "<config>") -> ArchConfig:
    """Parse and statically validate a raw config mapping."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    for section in ("architecture", "rules"):
        if section not in raw:
            raise ConfigError(f"{path}: missing required section {section!r}")

    arch_raw = raw["architecture"]
    if not isinstance(arch_raw, dict) or "id" not in arch_raw:
        raise ConfigError(f"{path}: 'architecture' must be a mapping with an 'id'")
    for key in arch_raw:
        if key not in ("id", "aliases", "gguf_arch", "strict_arch", "description"):
            raise ConfigError(f"{path}: unknown architecture key {key!r}")
    gguf_arch = arch_raw.get("gguf_arch")
    if isinstance(gguf_arch, list):
        if not gguf_arch or not all(isinstance(a, str) and a for a in gguf_arch):
            raise ConfigError(
                f"{path}: gguf_arch list must be non-empty strings (current identifier first)"
            )
        gguf_arch = tuple(str(a) for a in gguf_arch)
    elif gguf_arch is not None and not isinstance(gguf_arch, str):
        raise ConfigError(f"{path}: gguf_arch must be a string or list of strings")
    arch = Architecture(
        id=str(arch_raw["id"]),
        aliases=tuple(str(a) for a in arch_raw.get("aliases", ())),
        gguf_arch=gguf_arch,
        strict_arch=bool(arch_raw.get("strict_arch", False)),
        description=str(arch_raw.get("description", "")),
    )

    dims_raw = raw.get("dims", {})
    if not isinstance(dims_raw, dict):
        raise ConfigError(f"{path}: 'dims' must be a mapping")
    for name, spec in dims_raw.items():
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", str(name)):
            raise ConfigError(f"{path}: invalid dim name {name!r}")
        _validate_scalar_spec(spec, f"dims.{name}")

    rules_raw = raw["rules"]
    if not isinstance(rules_raw, list) or not rules_raw:
        raise ConfigError(f"{path}: 'rules' must be a non-empty list")
    rules = tuple(_parse_rule(r, i) for i, r in enumerate(rules_raw))
    names = [r.name for r in rules if r.name]
    if len(names) != len(set(names)):
        raise ConfigError(f"{path}: duplicate rule names")

    cov_raw = raw.get("coverage", {})
    if not isinstance(cov_raw, dict):
        raise ConfigError(f"{path}: 'coverage' must be a mapping")
    req = tuple(cov_raw.get("per_layer_required", ()))
    alts = tuple(
        tuple(alt) for alt in cov_raw.get("per_layer_alternatives", ())
    )
    for tpl in req:
        if not isinstance(tpl, str):
            raise ConfigError(f"{path}: coverage.per_layer_required entries must be strings")
    for alt in alts:
        if not alt or not all(isinstance(t, str) for t in alt):
            raise ConfigError(f"{path}: coverage.per_layer_alternatives must be non-empty string lists")

    policy = raw.get("unmatched_tensors", "error")
    if policy not in ("error", "warn"):
        raise ConfigError(f"{path}: unmatched_tensors must be 'error' or 'warn'")

    return ArchConfig(
        path=path,
        architecture=arch,
        dims=dict(dims_raw),
        rules=rules,
        coverage=Coverage(per_layer_required=req, per_layer_alternatives=alts),
        output=_parse_output(raw.get("output"), "output"),
        unmatched_policy=policy,
    )


def load_arch_config(path: str) -> ArchConfig:
    """Load a YAML architecture config and validate its schema."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read config: {exc}") from exc
    return arch_config_from_dict(raw, path=path)
