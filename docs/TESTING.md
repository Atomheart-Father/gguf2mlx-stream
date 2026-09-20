# Testing

Three layers, all CI-friendly (no multi-GB checkpoints required):

```bash
pytest                  # everything below, ~10 s
pytest -m integration   # only the env-gated real-GGUF tests (usually skipped)
ruff check src tests
```

## A. Unit tests (synthetic tensors only)

| file | covers |
|---|---|
| `test_source_gguf.py` | GGUF reader: **real hand-built Q4_K/Q6_K block bytes** with analytically derived values (non-circular: bytes are constructed from the GGML k-quant spec, not from the library under test), row-range reads, metadata fallback, error paths |
| `test_ops_generic.py` | every generic operator: semantics, ordering, purity (inputs unmutated), chunk-safety classification, validation errors |
| `test_plugin_qwen35.py` | v-head unpermute: rows/cols, block=1 vectors, dim resolution, round-trip against a `zip` reference |
| `test_config_schema.py` | config grammar: unknown ops/keys, bad regex, bad dims, drop-rule constraints, duplicate names, YAML object-injection rejection |
| `test_planner.py` | matching, dest templating (groups vs dims), conflicts, unmatched policy, unused required rules, `expect_shape`, coverage, dim fallback chains, slot resolution, arch checks |
| `test_quantize_writer.py` | 4/6-bit quantize→dequantize round-trips, packing shapes, bad group rejection; shard splitting, `-of-N` renaming, index totals |

## B. Synthetic end-to-end pipeline (`test_pipeline_synthetic.py`)

Builds a tiny GGUF with the **real qwen3_5 tensor-name inventory** — hybrid
GDN/full-attention layers, fused `attn_qkv`, fused `attn_q` gate rows,
permuted v-head tensors, f32 norms/conv1d/A_log, **Q6_K + Q4_K quantized
globals**, and a 15-tensor MTP block — then runs the public CLI:

```
read → plan → transform → quantize → shard write → reload
```

and asserts:

* conversion and `verify` both exit 0;
* output config (`model_type`, nesting, quantization entries, dims);
* MTP keys absent; expected keys present; 927-key parity is covered
  separately by the structural tests;
* exact float32 outputs (A_log = log(−unperm(ssm_a)), dt_bias, conv1d
  `(dim, k, 1)`, norms);
* quantized outputs match a reference quantization of the analytically
  expected values (requantization idempotence), including unpermuted
  `in_proj_qkv` v-rows and `out_proj` columns;
* the row-chunked streaming path is exercised (`--chunk-mb 1`).

## C. Structural regression vs. proven reference (`test_nyx_parity.py`)

The original source GGUFs were deleted after the golden conversion, so the
*proven outputs* anchor the regression:

1. **Skeleton parity** — a tiny f32 GGUF carrying the exact real Nyx tensor
   name inventory (32 layers, 24 GDN + 8 full-attention, MTP block) is
   planned with `configs/qwen3_5.yaml`; the planned destination key set
   (including `.scales`/`.biases` companions) must equal the reference
   `model.safetensors.index.json` weight map **exactly (927 keys)**.
2. **Reference structure** — the reference 6-bit/4-bit directories are
   checked directly: index consistency, shard presence, quantization triple
   shapes/dtypes per bit width, config fields, finiteness of all
   float tensors, special tensor shapes (conv1d `(8192,4,1)`, A_log/dt_bias
   `(32,)`, embed packing `(248320, 768|512)`).

Reference locations (env-overridable, tests skip when absent):

```
GGUF2MLX_NYX_REFERENCE_6BIT  (default ~/Models/MLX/Nyx-RP-9B-Instruct-2608-v1-MLX-6bit)
GGUF2MLX_NYX_REFERENCE_4BIT  (default ~/Models/MLX/Nyx-RP-9B-Instruct-2608-v1-MLX-4bit)
```

## D. Optional local integration tests (`test_integration_nyx.py`)

Full bounded-memory regression against real source GGUFs. **Skipped unless
assets are provided via environment variables:**

```bash
GGUF2MLX_TEST_Q6_GGUF=/path/model.Q6_K.gguf \
GGUF2MLX_TEST_Q4_GGUF=/path/model.Q4_K_M.gguf \
GGUF2MLX_TEST_SOURCE_DIR=/path/original-hf-dir \   # config.json + tokenizer files
GGUF2MLX_TEST_LOAD=1 \                             # optional mlx_lm.load + generation
pytest tests/test_integration_nyx.py -s
```

Asserts: 137 planned jobs, 15 dropped MTP tensors, 927 output keys, output
size plausibility (~6.8 GiB 6-bit / ~4.7 GiB 4-bit), verifier pass, and
optionally an `mlx_lm.load()` + generation smoke test. Sources are only
read; outputs go to a temp directory.
