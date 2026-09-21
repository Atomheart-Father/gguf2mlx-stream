# Roadmap

## Done (0.1.0a1)

* Compiler-like core: declarative config → validated plan → bounded-memory
  (tensor-bounded streaming) executor.
* **Qwen3.5 reference refactor** — the proven Nyx/Qwen3.5 conversion is now
  expressed entirely as config + generic operators + one thin plugin
  (structural 927-key parity + numeric fixtures).
* **P0 correctness fixes** — current llama.cpp `qwen35` architecture id +
  `qwen35.ssm.*` metadata mapping (legacy `qwen3_5_text`/`linear_*` kept as
  alias); NextN/MTP handling.
* **4-family config matrix** — `qwen3_5`, `qwen3`, `llama`, `gemma3`
  (including the llama q/k storage-permutation inverse and the gemma3
  RMSNorm `+1` convention, both with generic operators only).
* **Integration evidence** — 8 real-GGUF conversions (4 families ×
  Q4_K_M + Q6_K) pass the full stage pipeline (convert → structural →
  verify → mlx_lm load → chat generation → llama.cpp comparison), plus an
  isolated oMLX server discovering all outputs; 84 passing tests.
* **Transactional outputs** — staged conversion + atomic swap;
  `--overwrite` gate; explicit `required_fields` config emission.
* **Independent oracle test layer** — pure-numpy reimplementations drive
  the pipeline against non-circular expectations.
* **CI** — Linux (CPU MLX backend smoke) + macOS matrix, ruff, config
  validation smoke test.
* Version 0.1.0a1 published metadata (alpha classifier, dependency floors).

## Next

1. **Second stress-test architecture** — chosen deliberately to stress the
   abstraction (criteria below), not to grow the supported-models list.
2. **oMLX as a first-class integration target** — the isolated-server stage
   exists in the integration matrix; promote it to a documented, routinely
   run compatibility gate.
3. **Publication checklist** — license/attribution review, secret scan,
   large-file scan, local-path leakage scan, README claims vs verified
   architecture matrix.

## Later candidates

* **Chunk-aware permutations** — allow permute-class ops in the streaming
  path when chunk boundaries align to the block span.
* **More GGUF formats** — Q3_K/Q5_K/Q2_K etc. (mostly free via gguf-py
  dequant; needs fixtures + tests).
* **Single-file outputs** (no shard split) for small models.
* **bfloat16 non-quantized output** (needs an ml_dtypes dependency or a
  manual bf16 writer).
* **`inspect --plan-mapping`** — overlay config rules onto a GGUF
  inventory without a full plan compile.

## Second-architecture selection criteria

The next config should differ structurally from the families already
covered while remaining verifiable:

* at least one **multi-source combine** (real concat of two GGUF tensors
  into one destination) to exercise dependency grouping;
* an **axis order change** (transpose-class) or **dimension rearrangement**
  not yet covered;
* ideally another **norm convention** to prove +1-style handling is
  config-driven, not engine-driven (llama ships verbatim, gemma3 subtracts
  1 — a third variant would strengthen this);
* a public reference checkpoint + GGUF with known-good conversions for
  verification.

Candidates: a MoE (many-to-one combines) or an architecture with
non-trivial weight fusion/splitting beyond the current q|k|v patterns.

## Explicit non-goals (for now)

* Reverse direction (MLX → GGUF).
* Building `tokenizer.json` from GGUF-embedded vocabularies.
* GPU-sharded or distributed conversion.
* Support claims for architectures without an end-to-end verified config.
* Qwen3.8 is not a claimed target.
