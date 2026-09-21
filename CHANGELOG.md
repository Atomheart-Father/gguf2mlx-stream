# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
