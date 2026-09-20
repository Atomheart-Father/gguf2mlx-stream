"""Execution planner.

Compiles a validated architecture config plus a concrete GGUF source into a
:class:`ConversionPlan` *before* any large tensor is read. The planner
resolves dimensions, matches tensor rules, detects destination conflicts,
unmatched source tensors, unused rules, shape mismatches, and coverage
failures — a broken config fails here, cheaply.

The plan is a flat, ordered list of :class:`PlannedJob` objects, one per
output tensor. Jobs never load weights; they describe bounded work.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Iterable, Mapping

from .config.schema import ArchConfig, InputSlot, OpStep, Rule, SliceSpec
from .errors import PlanError
from .ops import all_ops
from .source.gguf import GGUFSource, TensorInfo

# ---------------------------------------------------------------------------
# dimension resolution
# ---------------------------------------------------------------------------

_ARITH = {"mul", "add", "sub", "div"}
_FN = {
    "mul": lambda xs: _fold(xs, lambda a, b: a * b),
    "add": lambda xs: sum(xs),
    "sub": lambda xs: _fold(xs, lambda a, b: a - b),
    "div": lambda xs: _fold(xs, lambda a, b: a // b),
}


def _fold(xs: list[int], fn) -> int:
    out = xs[0]
    for x in xs[1:]:
        out = fn(out, x)
    return out


class DimResolver:
    """Resolves scalar specs against GGUF metadata and an optional reference config."""

    def __init__(self, source: GGUFSource, ref_config: Mapping[str, Any] | None):
        self._source = source
        self._ref = ref_config or {}
        self._resolved: dict[str, int] = {}

    def resolve(self, spec: Any, where: str, dims_so_far: Mapping[str, int]) -> Any:
        """Resolve a scalar spec to an int/float/str/bool/list for use at runtime.

        A list spec is an ordered *fallback chain*: entries are tried in
        order and the first one that resolves successfully wins. (Literal
        list values are reachable via ``ref:`` paths into the reference
        config.)
        """
        if spec is None or isinstance(spec, (bool, int, float)):
            return spec
        if isinstance(spec, list):
            last_err: Exception | None = None
            for candidate in spec:
                try:
                    return self.resolve(candidate, where, dims_so_far)
                except PlanError as exc:
                    last_err = exc
            raise PlanError(f"{where}: no fallback resolved (last error: {last_err})")
        if isinstance(spec, str):
            if spec.startswith("gguf:"):
                key = spec[len("gguf:"):]
                value = self._source.metadata_value(key)
                if value is None:
                    raise PlanError(f"{where}: metadata key {key!r} not found in GGUF")
                return value
            if spec.startswith("ref:"):
                path = spec[len("ref:"):].split(".")
                node: Any = self._ref
                for part in path:
                    if not isinstance(node, Mapping) or part not in node:
                        raise PlanError(
                            f"{where}: reference config path {'.'.join(path)!r} not found "
                            "(is --source-config required by this architecture?)"
                        )
                    node = node[part]
                return node
            if spec in dims_so_far:
                return dims_so_far[spec]
            return spec  # plain string constant
        if isinstance(spec, list):
            return [self.resolve(item, where, dims_so_far) for item in spec]
        if isinstance(spec, dict):
            (op, operands), = spec.items()
            values = [self.resolve(o, where, dims_so_far) for o in operands]
            if not all(isinstance(v, int) for v in values):
                raise PlanError(f"{where}: arithmetic over non-integers: {values}")
            return _FN[op](values)
        raise PlanError(f"{where}: cannot resolve value spec {spec!r}")

    def resolve_dims(self, specs: Mapping[str, Any]) -> dict[str, int]:
        dims: dict[str, int] = {}
        for name, spec in specs.items():
            value = self.resolve(spec, f"dims.{name}", dims)
            if not isinstance(value, int):
                raise PlanError(f"dims.{name}: expected int, got {value!r}")
            dims[name] = value
        return dims


# ---------------------------------------------------------------------------
# plan objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PlannedJob:
    """One output tensor: bounded reads -> operator steps -> write."""

    dest: str  # final safetensors key (prefix included)
    rule: Rule
    source: TensorInfo
    slots: tuple[tuple[str, str, tuple[int, int] | None], ...]  # (slot, source name, (lo, hi) rows) — axis-0 slices only precomputed; other axes slice in memory
    slot_specs: tuple[tuple[str, str, SliceSpec | None], ...]
    steps: tuple[OpStep, ...]
    quantize: bool
    dtype: str
    est_source_bytes: int
    est_output_bytes: int

    @property
    def chunkable(self) -> bool:
        """Row-chunked streaming is possible for simple whole-tensor pipelines."""
        if self.slot_specs and any(s for _, _, s in self.slot_specs):
            return False
        if not self.steps:
            return True
        return all(all_ops()[s.op].streaming for s in self.steps)


@dataclasses.dataclass(frozen=True)
class ConversionPlan:
    config: ArchConfig
    source: GGUFSource
    dims: Mapping[str, int]
    jobs: tuple[PlannedJob, ...]
    dropped: tuple[tuple[str, str], ...]  # (tensor name, rule display name)
    unmatched: tuple[str, ...]
    unused_rules: tuple[str, ...]
    layer_count: int | None
    est_output_bytes: int
    dest_map: Mapping[str, str]  # dest -> rule display name

    def summary_lines(self) -> list[str]:
        """Human-readable dry-run summary."""
        lines: list[str] = []
        a = self.config.architecture
        lines.append(f"architecture : {a.id} (gguf_arch={a.gguf_arch})")
        lines.append(f"source       : {self.source.path}")
        lines.append(f"source arch  : {self.source.arch} | tensors: {len(self.source.tensors)}")
        lines.append("dims         : " + ", ".join(f"{k}={v}" for k, v in self.dims.items()))
        lines.append("")
        lines.append(f"jobs ({len(self.jobs)}) — dest <- rule [source] :")
        for job in self.jobs:
            tail = ""
            if job.slot_specs:
                tail = " slots=[" + ", ".join(n for n, _, _ in job.slot_specs) + "]"
            if job.steps:
                tail += " ops=[" + " -> ".join(s.op for s in job.steps) + "]"
            tail += " quantize" if job.quantize else f" dtype={job.dtype}"
            lines.append(f"  {job.dest}  <- {job.rule.display_name} [{job.source.name}]{tail}")
        if self.dropped:
            lines.append("")
            lines.append(f"dropped tensors ({len(self.dropped)}):")
            for name, rule in self.dropped:
                lines.append(f"  {name}  ({rule})")
        if self.unused_rules:
            lines.append("")
            lines.append("rules that matched nothing:")
            for r in self.unused_rules:
                lines.append(f"  {r}")
        if self.unmatched:
            lines.append("")
            lines.append("UNMATCHED source tensors:")
            for name in self.unmatched:
                lines.append(f"  {name}")
        lines.append("")
        lines.append(f"estimated output: {self.est_output_bytes / 2**30:.2f} GiB")
        return lines


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------


def _substitute_dims(pattern: str, dims: Mapping[str, int], where: str) -> str:
    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name not in dims:
            raise PlanError(f"{where}: pattern placeholder {{{name}}} is not a known dim")
        return str(dims[name])

    return re.sub(r"\{([a-z_][a-z0-9_]*)\}", repl, pattern)


def _resolve_arg_values(args: Mapping[str, Any], resolver: DimResolver, dims: Mapping[str, int], where: str) -> dict[str, Any]:
    out = {}
    for k, v in args.items():
        out[k] = resolver.resolve(v, f"{where}.args.{k}", dims)
    return out


def _slot_rows(source: GGUFSource, tensor: TensorInfo, sl: SliceSpec | None, resolver: DimResolver, dims: Mapping[str, int], where: str) -> tuple[tuple[int, int] | None, SliceSpec | None]:
    """Resolve a slot slice to (row_lo, row_hi) for fast mmap slicing when possible."""
    if sl is None:
        return None, None
    axis = sl.axis if isinstance(sl.axis, int) else int(resolver.resolve(sl.axis, where, dims))
    lo = resolver.resolve(sl.lo, where, dims) if sl.lo is not None else 0
    hi = resolver.resolve(sl.hi, where, dims) if sl.hi is not None else None
    if not isinstance(lo, int) or (hi is not None and not isinstance(hi, int)):
        raise PlanError(f"{where}: slice bounds must resolve to ints")
    if axis == 0:
        if hi is None:
            hi = tensor.n_rows
        if not (0 <= lo < hi <= tensor.n_rows):
            raise PlanError(f"{where}: axis-0 slice [{lo}:{hi}) invalid for {tensor.n_rows} rows")
        return (lo, hi), None
    # non-row slices are applied in memory after a full read
    return None, SliceSpec(axis=axis, lo=lo, hi=hi)


def plan_conversion(
    config: ArchConfig,
    source: GGUFSource,
    ref_config: Mapping[str, Any] | None = None,
) -> ConversionPlan:
    """Compile ``config`` against ``source`` into a validated ConversionPlan."""
    a = config.architecture
    if a.gguf_arch and source.arch != a.gguf_arch:
        msg = (
            f"GGUF general.architecture is {source.arch!r}, config {a.id!r} expects "
            f"{a.gguf_arch!r}"
        )
        if a.strict_arch:
            raise PlanError(msg)
        print(f"[plan] WARNING: {msg}")

    resolver = DimResolver(source, ref_config)
    dims = resolver.resolve_dims(config.dims)

    # compile rules
    compiled: list[tuple[Rule, re.Pattern]] = []
    for rule in config.rules:
        where = f"rule {rule.display_name!r}"
        pattern = _substitute_dims(rule.match, dims, where)
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise PlanError(f"{where}: invalid regex after dim substitution: {exc}") from None
        compiled.append((rule, rx))

    jobs: list[PlannedJob] = []
    dest_map: dict[str, str] = {}
    matched_by_rule: dict[str, list[str]] = {}
    dropped: list[tuple[str, str]] = []
    consumed: set[str] = set()

    for tensor in source.iter_infos():
        if tensor.name in consumed:
            continue
        job_for_tensor = None
        for rule, rx in compiled:
            m = rx.fullmatch(tensor.name)
            if not m:
                continue
            if rule.drop:
                dropped.append((tensor.name, rule.display_name))
                consumed.add(tensor.name)
                matched_by_rule.setdefault(rule.display_name, []).append(tensor.name)
                break
            groups = {k: v for k, v in m.groupdict().items()}
            where = f"rule {rule.display_name!r} tensor {tensor.name!r}"

            # dest templates: match groups take precedence over dim names
            dest_tpl = rule.dest or ""
            if dest_tpl:
                mapping = dict(dims)
                mapping.update(groups)
                try:
                    dest = dest_tpl.format(**mapping)
                except KeyError as exc:
                    raise PlanError(
                        f"{where}: dest template placeholder {exc} is neither a "
                        "match group nor a known dim"
                    ) from None
            else:
                dest = ""
            if dest in dest_map:
                raise PlanError(
                    f"destination conflict for {dest!r}: rules {dest_map[dest]!r} and "
                    f"{rule.display_name!r} both produce it"
                )

            slots: list[tuple[str, str, tuple[int, int] | None]] = []
            slot_specs: list[tuple[str, str, SliceSpec | None]] = []
            slot_inputs = rule.inputs or {"x": InputSlot()}
            est_src = 0
            for sname, slot in slot_inputs.items():
                if slot.source != "@":
                    raise PlanError(f"{where}: unsupported slot source {slot.source!r}")
                rows, in_mem = _slot_rows(source, tensor, slot.slice, resolver, dims, f"{where}.inputs.{sname}")
                slots.append((sname, tensor.name, rows))
                slot_specs.append((sname, tensor.name, in_mem))
                if rows:
                    est_src += (rows[1] - rows[0]) * tensor.row_bytes
                else:
                    est_src += tensor.n_bytes

            steps = tuple(
                OpStep(
                    op=s.op,
                    input=s.input,
                    inputs=s.inputs,
                    args=_resolve_arg_values(s.args, resolver, dims, f"{where}.steps"),
                    output=s.output,
                )
                for s in rule.steps
            )
            if len(slot_inputs) > 1 and not steps:
                raise PlanError(
                    f"{where}: multi-slot rules require operator steps "
                    "(e.g. concat) to combine their inputs"
                )

            # shape checks
            if rule.expect_shape is not None:
                expect = []
                for i, spec in enumerate(rule.expect_shape):
                    if spec is None:
                        expect.append(tensor.hf_shape[i])
                        continue
                    v = resolver.resolve(spec, f"{where}.expect_shape[{i}]", dims)
                    if not isinstance(v, int):
                        raise PlanError(f"{where}.expect_shape[{i}]: must resolve to int")
                    expect.append(v)
                if tuple(expect) != tuple(tensor.hf_shape):
                    raise PlanError(
                        f"{where}: shape mismatch, expected {tuple(expect)}, "
                        f"got {tuple(tensor.hf_shape)}"
                    )

            est_out = _estimate_output_bytes(tensor, rule)
            job = PlannedJob(
                dest=dest,
                rule=rule,
                source=tensor,
                slots=tuple(slots),
                slot_specs=tuple(slot_specs),
                steps=steps,
                quantize=rule.quantize,
                dtype=rule.dtype,
                est_source_bytes=est_src,
                est_output_bytes=est_out,
            )
            jobs.append(job)
            dest_map[dest] = rule.display_name
            consumed.add(tensor.name)
            matched_by_rule.setdefault(rule.display_name, []).append(tensor.name)
            job_for_tensor = job
            break
        if job_for_tensor is None:
            continue

    unused = tuple(
        r.display_name
        for r in config.rules
        if not r.drop and not r.optional and not matched_by_rule.get(r.display_name)
    )
    if unused:
        raise PlanError(f"required rule(s) matched no tensor: {list(unused)}")

    unmatched = tuple(sorted(t.name for t in source.iter_infos() if t.name not in consumed))
    if unmatched and config.unmatched_policy == "error":
        raise PlanError(
            f"{len(unmatched)} source tensor(s) matched no rule, e.g. {list(unmatched[:5])}"
        )

    # coverage checks
    layer_count = dims.get("n_layers")
    _check_coverage(config, jobs, layer_count)

    est_out_total = sum(j.est_output_bytes for j in jobs)
    return ConversionPlan(
        config=config,
        source=source,
        dims=dims,
        jobs=tuple(jobs),
        dropped=tuple(dropped),
        unmatched=unmatched,
        unused_rules=unused,
        layer_count=layer_count,
        est_output_bytes=est_out_total,
        dest_map=dest_map,
    )


def _estimate_output_bytes(tensor: TensorInfo, rule: Rule) -> int:
    n = tensor.n_elements
    if rule.drop:
        return 0
    if rule.quantize:
        return n  # refined at runtime; ~0.5-0.9 bytes/elem for 4/6-bit
    return n * (4 if rule.dtype == "float32" else 2)


def _check_coverage(config: ArchConfig, jobs: Iterable[PlannedJob], layer_count: int | None) -> None:
    cov = config.coverage
    dests = {j.dest for j in jobs}
    if not (cov.per_layer_required or cov.per_layer_alternatives):
        return
    if layer_count is None:
        raise PlanError(
            "coverage checks require a dims entry for the layer count "
            "(default dim name 'n_layers')"
        )
    for i in range(layer_count):
        for tpl in cov.per_layer_required:
            name = tpl.replace("{layer}", str(i))
            if name not in dests:
                raise PlanError(f"coverage: layer {i} missing required tensor {name}")
        for alt_group in cov.per_layer_alternatives:
            if not any(tpl.replace("{layer}", str(i)) in dests for tpl in alt_group):
                raise PlanError(
                    f"coverage: layer {i} missing all alternatives: {list(alt_group)}"
                )
