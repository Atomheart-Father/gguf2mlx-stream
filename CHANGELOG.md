# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-09-21

Initial release.

### Added

- Compiler-like core: declarative YAML architecture configs compiled into a
  validated conversion plan, executed by a bounded-memory streaming runner.
- GGUF source layer: mmap-backed reader with bounded tensor/row-chunk
  dequantization for Q4_K, Q6_K, Q8_0, F32, F16, BF16 (dequant math via the
  MIT-licensed `gguf` package).
- Declarative config grammar: anchored regex tensor rules with named groups,
  dest templating, explicit drop rules, multi-slot slicing jobs, closed
  dim-arithmetic vocabulary, dim fallback chains (`gguf:` / `ref:` /
  constants), coverage assertions; schema-validated, `yaml.safe_load` only.
- Operator registry with uniform pure-function signature and declared
  chunk-safety: 19 generic operators plus the first plugin,
  `qwen35_v_head_unpermute` (inverse of llama.cpp's GDN v-head storage
  permutation).
- Execution planner: dimension resolution, rule matching, destination
  conflict detection, unmatched-tensor policy, `expect_shape` assertions,
  per-layer coverage checks, estimated output size, dry-run summary.
- Bounded-memory runner: per-tensor or row-chunked
  dequantize → transform → MLX affine quantize → shard write → release;
  peak-RSS and timing reporting.
- MLX-LM writer: sharded safetensors with `model-0000N-of-0000M` naming,
  `model.safetensors.index.json`, MLX-LM `config.json` (nested
  `text_config`, quantization entries, optional reference-config merge),
  tokenizer file copying.
- Verifier: key-set parity, shape and NaN/inf checks, scale-aware numeric
  spot checks against the source GGUF, structural quantization checks,
  optional `mlx_lm.load()` + generation smoke test.
- CLI: `inspect`, `list-ops`, `validate-config`, `convert` (with `--dry-run`),
  `verify`.
- Qwen3.5 hybrid architecture config (`configs/qwen3_5.yaml`) reproducing the
  proven Nyx/Qwen3.5 conversion semantics (RMSNorm +1 convention, GDN v-head
  unpermutation, A_log domain transform, fused q|k|v and [q; gate] layouts,
  conv1d `(dim,k) → (dim,k,1)`, explicit MTP block drop, `model_type:
  qwen3_5` + nested `text_config` compatibility).
- Test suite: 66 passing tests — unit tests with real hand-built Q4_K/Q6_K
  block bytes, a synthetic end-to-end pipeline (read → plan → transform →
  quantize → write → reload), structural key-set parity (927/927) against
  the proven Nyx reference outputs, and env-gated optional integration tests
  for real source GGUFs.
- CI (Ubuntu + macOS), docs (ARCHITECTURE, CONFIG_SPEC, TESTING, ROADMAP),
  README, MIT license, changelog.
