# Testing

Three levels, all CI-friendly (no multi-GB checkpoints required), plus an
oracle-independence layer, transactional-output tests, and an opt-in
real-model integration matrix.

```bash
.venv/bin/python -m pytest tests/ -q
# current baseline: 110 passed, 3 skipped
# (the 3 skips are the env-gated real-GGUF tests, absent assets)

python -m pytest -m integration -q   # only the env-gated real-GGUF tests
ruff check src tests scripts
```

## A. Unit tests (tiny synthetic tensors)

| file | covers |
|---|---|
| `test_source_gguf.py` | GGUF reader: **real hand-built Q4_K/Q6_K block bytes** with analytically derived values (non-circular: bytes are constructed from the GGML k-quant spec, not from the library under test), row-range reads, metadata fallback, error paths |
| `test_ops_generic.py` | every generic operator: semantics, ordering, purity (inputs unmutated), chunk-safety classification, validation errors |
| `test_plugin_qwen35.py` | v-head unpermute: rows/cols, block=1 vectors, dim resolution, round-trip against a `zip` reference |
| `test_config_schema.py` | config grammar: unknown ops/keys, bad regex, bad dims, drop-rule constraints, duplicate names, `{list: ...}` forms, YAML object-injection rejection |
| `test_planner.py` | matching, dest templating (groups vs dims), conflicts, unmatched policy, unused required rules, `expect_shape`, coverage, dim fallback chains, slot resolution, range-drop compilation (declared by the `{i}` match template, not a hardcoded prefix), arch checks, plan-time pipeline shape validation (bad reshape/permute/unsqueeze/concat args, non-2-D quantized outputs, `out_shape` recording) |
| `test_quantize_writer.py` | 4/6-bit quantize→dequantize round-trips, packing shapes, bad group rejection; shard splitting, `-of-N` renaming, index totals |
| `test_verifier.py` | verifier credibility: tampered later-layer tensor of an already-seen rule is caught (full numeric coverage by default; sampling is `--sampled` opt-in), extra key inside a shard rejected, unindexed shard file rejected, `bits`/`group_size`/`mode` read from config.json with CLI-conflict failures, sampled opt-in still passes clean outputs |
| `test_transactional.py` | failed conversions leave no partial output and no staging leftovers; existing outputs need `--overwrite`; missing required config fields fail loudly; **tokenizer output contract**: missing tokenizer files / missing `tokenizer_config.json` / invalid `tokenizer.json` each fail the transaction and preserve the previous output |
| `test_packaging.py` | built-in config discovery + loading, `--arch-config` name/path/auto-detect resolution, `list-configs` CLI, wheel content: the five official configs byte-identical to the repo-root `configs/`, wheel metadata version == runtime `__version__`, auto-detected conversion without `--arch-config` |

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

## C. Oracle-independence layer (`tests/oracle_impl.py` + `test_oracles.py`)

`tests/oracle_impl.py` contains **pure-numpy reimplementations that share no
code with production operators** (no `gguf2mlx_stream` imports; explicit
index loops instead of the production vectorized gathers). They are written
from the GGUF layout specification, so a bug in a production operator cannot
hide behind a verifier that shares the same code.

`tests/test_oracles.py` drives the **real pipeline** (planner + runner) over
tiny synthetic GGUFs and compares every expectation against oracle-built
values:

* grouped-head reorder **ratio matrix 1–4** (rows, cols, and block-1
  vectors per ratio), plus a forward/inverse round-trip for all ratios;
* the documented `A_log = log(−unpermute(ssm_a))` formula;
* the **MTP key-set oracle** (which keys must exist / be absent after
  block-range drops);
* q+gate fusion semantics and conv1d `(dim, k, 1)` semantics;
* the **llama q/k unpermute** (oracle storage permutation vs the
  reshape→permute→reshape pipeline, round-trip + pipeline equivalence).

## D. Transactional output tests (`test_transactional.py`)

* an **injected failure** mid-conversion leaves no partial (or replaced)
  model directory at the output path, and no staging directories survive;
* an existing non-empty output is never replaced without `--overwrite`
  (an empty pre-created directory is);
* a config that cannot produce a **required config.json field** fails the
  conversion loudly instead of silently relying on mlx-lm defaults;
* the **tokenizer output contract** is transactional: a tokenizer source
  without a vocab file (`tokenizer.json`/`tokenizer.model`), without
  `tokenizer_config.json`, or with an invalid `tokenizer.json` fails the
  conversion — before any tensor is read or at commit time — and never
  replaces the previous output.

## E. Structural regression vs. proven reference (`test_nyx_parity.py`)

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

Note: the golden-reference directories (produced by the `gguf2mlx-nyx`
reference converter) and the local re-converted Nyx output were deleted
from the local model library after the parity work and the end-to-end
case study concluded; all three parity tests skip cleanly without these
assets. Point the env overrides at any Nyx MLX 4-bit/6-bit directory to
re-enable the structural checks.

## F. Optional env-gated local integration (`test_integration_nyx.py`)

Full bounded-memory regression against real source GGUFs. **Skipped unless
assets are provided via environment variables:**

```bash
GGUF2MLX_TEST_Q6_GGUF=/path/model.Q6_K.gguf \
GGUF2MLX_TEST_Q4_GGUF=/path/model.Q4_K_M.gguf \
GGUF2MLX_TEST_SOURCE_DIR=/path/original-hf-dir \   # config.json + tokenizer files
GGUF2MLX_TEST_LOAD=1 \                             # optional mlx_lm.load + generation
python -m pytest tests/test_integration_nyx.py -s
```

Asserts: 137 planned jobs, 15 dropped MTP tensors, 927 output keys, output
size plausibility (~6.8 GiB 6-bit / ~4.7 GiB 4-bit), verifier pass, and
optionally an `mlx_lm.load()` + generation smoke test. Sources are only
read; outputs go to a temp directory.

## G. Real-model integration matrix (release gate)

`scripts/run_integration_matrix.py` runs the full release-gate pipeline over
every pinned model/variant in `tests/integration/models.yaml`:

```
inspect → validate-config → dry-run → convert → structural
        → verify → mlx_generation → llamacpp (raw + chat)
        → optional isolated oMLX stage (--omlx)
```

* **Fixtures**: `scripts/fetch_integration_models.py` downloads only the
  pinned small GGUFs + tokenizer files (manifest
  `tests/integration/models.yaml` with pinned revisions; sha256 + size are
  recorded in `download-log.json`). Everything lands under
  `.integration-models/` (git-ignored). Reference model libraries are never
  touched, and outputs never overwrite proven references.
* **mlx_generation**: `mlx_lm.load()` + chat-template generation at temp 0
  with a runtime bar — no crash, no empty/immediately-EOS output, printable
  ratio > 0.9, and the longest generation ≥ 30 chars.
* **llamacpp**: the *source* GGUF is generated with `llama-completion`
  (raw continuations + chat turns) for semantic comparison records;
  token-for-token identity is not expected after requantization.
* **verify**: the release gate runs the default verifier — full numeric
  coverage (every quantized tensor recomputed from the source), index/shard
  parity, quantization-parameter validation against the output's
  config.json, finiteness checks. On the 8-variant matrix this adds a
  bounded per-tensor recompute pass (peak memory unchanged: one tensor at a
  time).
* **oMLX stage** (`--omlx`): spins up `omlx-cli serve` on a dedicated port
  with its own model directory — it never touches a user's running oMLX
  instance — then checks discovery of all converted outputs and
  `/v1/chat/completions` for the per-family probe models.
* **Reports** land in `reports/integration-matrix.{json,md}` (generated
  locally; run artifacts are git-ignored, not committed);
  per-variant convert statistics are written alongside the fixtures.

Latest published run: 8/8 variants PASS (4 families × Q4_K_M + Q6_K),
all outputs discovered by the isolated oMLX server.

Large-model one-off regression (`qwen35moe`, same pipeline stages, run
manually outside the pinned matrix): JoyFox Qwen3.6-35B-A3B-RP-Aggressive.
**The historical 4-bit run below is superseded history — wrong target for
an IQ3-dominant source, not a recommended result.** It remains the
reference run for the N-D (3-D expert) quantization and per-rule
bits/group_size override paths: 733 planned jobs, 20 dropped MTP tensors,
5 shards / 18.17 GiB output, peak RSS 16.48 GiB, verify ALL OK (733
numeric / 733 shape / 1757 finite), `mlx_lm.load()` + coherent temp-0
generation.

The current JoyFox conversion is a **3-bit `--bits auto` run** (14.14 GiB /
4 shards, peak RSS 12.06 GiB, verify ALL OK) whose capability-gate verdict
was **FAIL — experimental only, never a recommendation**: the ARC-Challenge
100q gate and the Llama-1B 3-bit calibration both reject a 3-bit target
for this source class (see `eval/bench/results/`). As of the phase-1
closing, same-bit auto is the final policy: auto-derived 3-bit targets
**convert and print a fidelity warning** (also recorded in the output
`config.json`); they are no longer blocked. See README
"Validation / Quantization Fidelity" for the evidence and the exact
warning text.

## H. Clean-install acceptance (release gate)

After building the wheel:

```bash
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
python -m venv /tmp/fresh-venv && /tmp/fresh-venv/bin/pip install dist/*.whl
/tmp/fresh-venv/bin/gguf2mlx-stream list-configs          # 5 built-ins, no clone
/tmp/fresh-venv/bin/gguf2mlx-stream validate-config llama  # builtin by name
/tmp/fresh-venv/bin/gguf2mlx-stream convert tiny.gguf --output out \
    --tokenizer-source tokenizer/ --bits 4 --quiet         # --arch-config omitted:
/tmp/fresh-venv/bin/gguf2mlx-stream verify tiny.gguf out   # auto-detect + full verify
```

`test_packaging.py` covers the same guarantees from inside pytest (wheel
contents byte-parity with the repo `configs/`, version agreement, config
resolution).
