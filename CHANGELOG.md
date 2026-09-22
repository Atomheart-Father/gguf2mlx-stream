# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### End-to-end case study: Nyx-RP-9B (2026-09-22)

- Re-converted `Nyx-RP-9B-Instruct-2608-v1.Q4_K_M.gguf` (9.2B Qwen3.5
  hybrid) with `--bits auto` → MLX affine 4-bit: 4.69 GiB / 2 shards /
  149 s / peak RSS 9.56 GiB, verify ALL OK (427 numeric / 427 shape /
  927 finite).
- Source-vs-MLX capability evaluation (45-question set, temp 0, recorded
  params, strict anomaly gating) committed at
  `eval/reports/nyx9b-q4km/`: gated accuracy 65.0% (source, llama.cpp
  0.4.1) vs 60.0% (MLX), verdict agreement 85%, throughput 31.2 vs
  31.8 tok/s (Apple M4 Pro). The −5 pp delta is anomaly-gating-driven at
  the shared 1536-token thinking budget, not knowledge loss.
- README restructured: Quickstart moved to the top, validation evidence
  ordered from the newest end-to-end case study down to the unit-level
  proofs.

### Phase-1 closing (mainline consolidation)

- **`--bits auto` policy finalized to same-bit conversion**: the MLX target
  bit magnitude equals the source GGUF's byte-dominant quant family
  (IQ2/Q2→2, IQ3/Q3→3, IQ4/Q4→4, Q6→6, Q8→8); explicit `--bits` always wins.
  The former blocking experimental guard is replaced by a **fidelity
  warning**: auto-derived 3-bit targets convert normally and print the
  paired-oracle warning to stderr, also recorded in the output
  `config.json` (`quantization_selection.fidelity_warning`). The
  `--allow-experimental` flag is removed (no longer needed).
- **Validation / Quantization Fidelity documentation** in the README:
  BF16 paired-oracle converter-correctness proofs (Qwen3.5-0.8B 320/320,
  Llama-3.2-1B 146/146), quantizer equivalence with mlx-lm (4+ significant
  digits), 3-bit fidelity findings table (uniform 3-bit, mixed 3/4, mixed
  3/6, uniform 4-bit, BF16→3-bit control), generation-based capability
  gates, and the QAT control. Research tooling and reports now live in the
  repository under `research/paired_oracle/`.
- **Linux CI fix**: `mlx[cpu]` is declared for Linux (the plain `mlx` wheel
  ships no compute backend there); CI push trigger fixed to `master`.
- **Lint brought to ruff 0.16.8-clean**: import ordering, deprecated
  `typing` imports, unused variables/`noqa`s, narrow or annotated broad
  exception handlers, `subprocess` `check` explicitness, executable bits
  for scripts. No tests were skipped or weakened to achieve this.

### Added (auto-bits + capability gate + research tooling)

- **`--bits auto` target-bits selection** from a byte-weighted source
  quant-family histogram over the plan's quantized jobs; evidence (dominant
  type, histogram, reason) recorded in `--report-json` and the output
  `config.json` under `quantization_selection`. Unmappable dominant
  families (Q5/IQ1/TQ) fail with an explicit error; auto never guesses.
- **ARC-Challenge benchmark harness** (`eval/bench/`): pinned 100-question
  subset (canonical sha256), raw-completion protocol with recorded params,
  strict anomaly taxonomy (loop/empty/format), source-vs-MLX scoring, and
  committed result reports (`eval/bench/results/`, `eval/reports/`).
- **Source-vs-MLX capability regression harness** (`eval/run_eval.py`,
  `eval/score_eval.py`) with committed reports for Qwen3.6-35B-A3B and
  Llama-3.2-1B.
- **Opt-in `--quant-profile` overlay**: JSON profiles overriding per-rule
  bits/group_size by destination-name regex (first match wins; drop/copy
  rules untouched; unmatched patterns are reported). Data only — no code
  execution from profiles. Used to reproduce official mixed 3/4 and 3/6
  MLX builds in the paired-oracle study.
- **Paired-oracle research tooling** (`research/paired_oracle/`): pinned
  asset manifest, BF16 tensor comparison, MLX profile extraction, per-module
  error attribution, and the fast NLL/KL + ARC-MC calibration gate, plus
  the full reports (`REPORT_BF16_ORACLE.md`,
  `REPORT_PAIRED_ORACLE.md`).

### Added

- **`qwen35moe` architecture config** (`configs/qwen3_5_moe.yaml`): Qwen3.5
  hybrid GDN + full attention with 256-expert sparse MoE. Same GDN/v-head
  semantics as `qwen3_5`; MoE tensors quantized directly in 3-D (experts,
  tokens, inner) without ever materializing the full FP32 expert tensor.
- **N-D quantization**: affine group quantization generalized from 2-D to
  arbitrary rank ≥ 2. Leading dims are preserved; the last dim is
  group-packed. Chunked conversion reshapes concatenated row chunks back to
  N-D, so huge expert tensors stream in bounded memory. Chunked and
  unchunked outputs are byte-identical (tested).
- **Config-declared per-rule quantization overrides**: rules may declare
  `bits`/`group_size`; overrides are validated at plan time, written as
  mlx-lm per-module keys (module path without the trailing `.weight`) in
  config.json's `quantization` mapping, and cross-checked by `verify`.
  The `qwen35moe` config uses this for the MoE router and shared-expert
  gate (8-bit).
- **GGUF metadata array reads**: metadata values that are arrays (e.g.
  `rope.dimension_sections`) are decoded with a bounded element cap and can
  be referenced declaratively in output config templates
  (`gguf:` references and list-valued `ref:` chains).

### Fixed

- Trailing blank line at EOF in the qwen3.5 plugin operator module.

## [0.1.0a1] - 2026-09-21

Initial alpha release.

### Release-gate hardening (P1, folded into this release before publication)

- **Built-in architecture configs**: the repository-root `configs/` remains
  the authoritative source; hatchling force-includes it into the wheel as
  `gguf2mlx_stream/configs/`. `gguf2mlx-stream list-configs` lists them;
  `--arch-config` now accepts a built-in name, an explicit YAML path, or can
  be omitted entirely (auto-detect from the GGUF's `general.architecture`
  when exactly one built-in config accepts it). Editable/source checkouts
  fall back to the repository `configs/` directory. Wheel content is tested
  for byte-parity with the repo configs and version agreement.
- **Tokenizer output contract**: a conversion may only succeed if the output
  directory will contain a loadable tokenizer (`tokenizer.json` or
  `tokenizer.model`, plus `tokenizer_config.json`). Enforced before any
  tensor is read and again on the staged output before the commit; failures
  abort the transaction and never replace an existing output.
- **Verifier full numeric coverage by default**: every quantized output
  tensor is recomputed from the source and compared. Sampling (first tensor
  per rule + small tensors) is an explicit opt-in (`verify --sampled`).
- **Bidirectional index/shard validation**: unindexed keys inside shards,
  stale index entries and unreferenced shard files are all rejected, in
  addition to plan key-set parity.
- **Quantization-parameter validation**: `verify` reads `bits`,
  `group_size`, `mode` from the output's config.json; explicit CLI values
  conflicting with the recorded metadata fail, and weights that cannot be
  dequantized under the recorded parameters are reported failures instead
  of crashing.
- **Plan-time pipeline shape inference**: every registered operator now
  carries a shape-inference function; the planner validates each job's
  operator pipeline (axes, divisibility, reshape products, concat
  compatibility, final shape; quantized rules must be 2-D) with shape
  tuples only, before any tensor data is read. Invalid dims/rank/operator
  parameters raise `PlanError` instead of runtime
  `TypeError`/`IndexError`/`ValueError`.
- New tests: verifier tamper/parity/quant-parameter regressions,
  tokenizer-contract negative transaction tests, planner range/shape tests,
  packaging tests (wheel contents, clean resolution, auto-detect).

### Changed (P1)

- **Range drop rules are fully declarative**: the rule's own `match`
  template (reserved `{i}` placeholder, `fullmatch` per block index) defines
  which tensors a range drops — the previous hardcoded `blk.{i}.` prefix is
  gone. Existing configs already declare `blk\.{i}\..*` and are unchanged in
  behavior.
- `verify` CLI: `--all` replaced by `--sampled` (inverted, explicit
  opt-in); `--group-size` and the new `--mode` default to the output
  config.json instead of assuming values.
- Version management: `__version__` in `src/gguf2mlx_stream/__init__.py` is
  the single source of truth (hatch dynamic versioning), aligned to
  `0.1.0a1`.

### Removed (P1)

- The placeholder `Homepage` URL (no canonical public repository exists
  yet). `pyproject.toml` carries no `[project.urls]` until a real upstream
  exists.

## [0.1.0a1] - 2026-09-21

Initial alpha release.

### Added

- Compiler-like core: declarative YAML architecture configs compiled into a
  validated conversion plan, executed by a bounded-memory runner with
  tensor-bounded streaming dequantization (no full FP16/BF16 staging).
- GGUF source layer: mmap-backed reader with bounded tensor/row-chunk
  dequantization for Q4_K, Q6_K, Q8_0, F32, F16, BF16 (dequant math via the
  MIT-licensed `gguf` package).
- Declarative config grammar: anchored regex tensor rules with named groups,
  dest templating, explicit drop rules, multi-slot slicing jobs, closed
  dim-arithmetic vocabulary, dim fallback chains (`gguf:` / `ref:` /
  `tshape:` / `has:` / constants), coverage assertions; schema-validated,
  `yaml.safe_load` only.
- Operator registry with uniform pure-function signature and declared
  chunk-safety: 19 generic operators plus the `qwen35_v_head_unpermute`
  plugin (a thin geometry-resolving wrapper over the generic
  `reorder_grouped_heads` operator).
- Execution planner: dimension resolution, rule matching, destination
  conflict detection, unmatched-tensor policy, `expect_shape` assertions,
  per-layer coverage checks, estimated output size, dry-run summary.
- Bounded-memory runner: per-tensor or row-chunked
  dequantize → transform → MLX affine quantize → shard write → release;
  peak-RSS and timing reporting.
- MLX-LM writer: sharded safetensors with `model-0000N-of-0000M` naming,
  `model.safetensors.index.json`, MLX-LM `config.json`, tokenizer file
  copying.
- Explicit `config.json` emission with `required_fields`: every field the
  runtime reads is declared and presence-checked after emission (explicit
  null is a legal value); a broken or over-permissive config fails the
  conversion instead of silently relying on mlx-lm defaults.
- NextN/MTP block removal via block-range drop rules
  (`drop: true` + `range: {start, end}` with the reserved `{i}` placeholder;
  `n_layers = block_count − nextn_predict_layers`).
- New architecture configs alongside `qwen3_5`:
  - `qwen3` — dense, tied-embedding aware, q/k-norm pass-through;
  - `llama` — undoes the llama.cpp convert-time q/k out-axis storage
    permutation with a generic reshape→permute→reshape chain; drops the
    derived `rope_freqs.weight`;
  - `gemma3` — subtracts the llama.cpp-baked `+1` from RMSNorm weights
    (mlx-lm `gemma3_text` re-adds 1.0 at runtime); sliding/global attention
    pattern literals.
- `{list: [...]}` literal-list scalar spec (distinct from fallback-chain
  lists) for operator args such as reshape shapes and permute axes.
- Independent oracle test layer: `tests/oracle_impl.py` (pure-numpy
  reimplementations sharing no code with production operators) driven
  through the real pipeline — grouped-head reorder ratio matrix 1–4, A_log
  formula, MTP key-set oracle, q+gate fusion, conv1d, llama q/k unpermute.
- Transactional output tests: injected-failure leaves no partial output;
  existing output requires `--overwrite`; missing required config fields
  fail conversion.
- Integration matrix harness: pinned fixture manifest
  (`tests/integration/models.yaml`), fetch script with sha256 logging,
  full stage pipeline (inspect → validate-config → dry-run → convert →
  structural → verify → mlx generation → llama.cpp comparison → optional
  isolated oMLX server), reports under `reports/`; 8/8 variants PASS
  (4 families × Q4_K_M + Q6_K).
- CI: Linux (CPU MLX backend smoke test) + macOS jobs, ruff, config
  validation smoke test, `list-ops` check.
- Test suite: 84 passing, 3 skipped (env-gated real-GGUF tests) — unit
  tests with real hand-built Q4_K/Q6_K block bytes, a synthetic end-to-end
  pipeline, oracle-independence tests, transactional tests, and structural
  key-set parity (927/927) against the proven Nyx reference outputs.
- CLI: `inspect`, `list-ops`, `validate-config`, `convert` (with
  `--dry-run`), `verify`.

### Fixed

- Architecture identifier and SSM metadata mapping: accept the current
  llama.cpp `qwen35` identifier with `qwen35.ssm.*` metadata keys (and
  keep the legacy `qwen3_5_text` identifier with `linear_*` keys as an
  accepted alias); SSM dims resolve through `gguf:` → `ref:` fallback
  chains in both layouts.

### Changed

- Generalized `reorder_grouped_heads` (ratio 1–4) replaces the ratio-2-only
  un-interleave as the data-movement primitive behind the qwen35 plugin
  (ratio=1 identity, ratio=2 equivalent to `unzip_blocks`).
- Transactional runner: conversion output is staged into a temporary
  sibling directory and atomically swapped into place; failed or
  interrupted conversions leave no partial artifacts; existing non-empty
  outputs require `--overwrite`.
- `--report-json` writes run statistics (output bytes, shards, dims, peak
  RSS, elapsed); `-q/--quiet` for machine-driven runs.
- Flat-vs-nested `config.json` emission is an explicit per-family config
  choice (`nest_config_under`): llama/qwen3/gemma3_text emit flat fields;
  qwen3_5 nests under `text_config` (mlx-lm 0.31.x requirement).
- `pyproject.toml`: version 0.1.0a1, alpha development-status classifier,
  dependency floors pinned to validated minimums
  (numpy≥1.26, gguf≥0.19.0, pyyaml≥6.0, mlx≥0.29.1, safetensors≥0.4).
