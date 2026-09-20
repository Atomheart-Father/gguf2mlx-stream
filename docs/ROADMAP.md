# Roadmap

## Done (v0.1.0)

* Compiler-like core: declarative config → plan → bounded streaming executor.
* GGUF source layer (Q4_K/Q6_K/Q8_0/F32/F16/BF16) with bounded row reads.
* 19 generic operators + first plugin (`qwen35_v_head_unpermute`).
* Planner diagnostics: conflicts, unmatched, coverage, shapes, dry-run.
* MLX-LM writer (shards, index, config, tokenizer) + verifier.
* Qwen3.5 hybrid support validated against the proven golden conversion
  (structural 927-key parity + numeric fixtures).

## Next

1. **Second architecture** — chosen deliberately to stress the abstraction
   (see below), not to grow the supported-models list.
2. **Chunk-aware permutations** — allow permute-class ops in the streaming
   path when chunk boundaries align to the block span.
3. **More GGUF formats** — Q3_K/Q5_K/Q2_K etc. (mostly free via gguf-py
   dequant; needs fixtures + tests).
4. **Single-file outputs** (no shard split) for small models.
5. **bfloat16 non-quantized output** (needs an ml_dtypes dependency or a
   manual bf16 writer).
6. **`inspect --plan-mapping`** — overlay config rules onto a GGUF
   inventory without a full plan compile.

## Second-architecture selection criteria

The next config should differ structurally from Qwen3.5 while remaining
verifiable:

* at least one **multi-source combine** (real concat of two GGUF tensors
  into one destination) to exercise dependency grouping;
* an **axis order change** (transpose-class) or **dimension rearrangement**
  not present in Qwen3.5;
* ideally a **different norm convention** (e.g. one that must NOT copy
  verbatim) to prove the +1 handling is config-driven, not engine-driven;
* a public reference checkpoint + GGUF with known-good conversions for
  verification.

Candidates: a small dense Qwen2/Llama-class model (mostly copy rules — good
smoke test but low stress), Gemma-class (transpose + norm differences), or a
MoE (many-to-one combines). Preference: **Gemma-class** first, MoE second.

## Explicit non-goals (for now)

* Reverse direction (MLX → GGUF).
* Building `tokenizer.json` from GGUV-embedded vocabularies.
* GPU-sharded or distributed conversion.
* Support claims for architectures without an end-to-end verified config.
