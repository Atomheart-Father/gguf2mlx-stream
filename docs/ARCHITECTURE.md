# Architecture

gguf2mlx-stream is a small compiler-like pipeline:

```
                 YAML (declarative architecture config)
                                │  validate-config (schema)
                                ▼
GGUF file ──► GGUFSource ──► Planner ──► ConversionPlan ──► ConversionRunner ──► MLX-LM dir
             (mmap,              (match,        (jobs)          (bounded stream)     (shards +
              dequant)            resolve,                        └─ quantize.py      index +
                                  check)                          └─ writer.py        config)
```

## Layers

### 1. Source layer (`source/gguf.py`)

Knows GGUF and GGML quantization formats — and *nothing about models*.

* `GGUFSource`: opens the file with `gguf.GGUFReader` (mmap), exposes
  metadata (scalar lookups with `{arch}.`-prefix fallback) and `TensorInfo`
  (name, ne, qtype, hf-convention shape, per-row byte size).
* Bounded reads: `read_rows(name, lo, hi)` dequantizes an outer-row range;
  `read_matrix(name)` reads one whole tensor. Row slicing uses absolute
  byte offsets into the mmap, so only the requested bytes are touched.
* Dequantization math is delegated to `gguf.quants` (llama.cpp, MIT).

### 2. Config layer (`config/schema.py`)

Declarative data, validated statically. The YAML grammar is closed:

* rules: anchored regex (named groups) → dest template (groups + dims);
* dims: named integers resolved from `gguf:<key>`, `ref:<path>` (optional
  reference config), constants, or a closed arithmetic vocabulary
  (`mul/add/sub/div`); list values are ordered fallback chains;
* transforms: ordered operator steps referencing *registered op names*;
* `yaml.safe_load` rejects object construction; no eval/import paths exist.

### 3. Operator registry (`ops/`)

Operators are pure functions with one uniform signature:

```python
fn(inputs: dict[str, np.ndarray], order: list[str], args: dict,
   ctx: OpContext) -> np.ndarray
```

* No global state; inputs never mutated; float32 semantics on HF-convention
  arrays (2-D = `(out, in)`).
* Each op declares a `kind` (elementwise / shape / combine / permute);
  `kind` determines whether the op is chunk-safe (streaming) or needs the
  whole tensor.
* Built-ins are registered at import; plugins live in `ops/plugins/` and are
  imported (i.e. registered) by the package — never dynamically by configs.
* The first plugin is `qwen35_v_head_unpermute`, which resolves head count /
  block size from the architecture dims and delegates to the generic
  `unzip_blocks` primitive (the inverse of llama.cpp's
  `concat(hf[0::2], hf[1::2])` v-head storage permutation).

### 4. Planner (`planner.py`)

Compiles config + source into a `ConversionPlan` *before* reading weights:

* resolves dims (failing fast on missing metadata/refs);
* substitutes `{dim}` placeholders in match patterns, compiles anchored
  regexes, matches every GGUF tensor to the first matching rule;
* resolves dest templates (match groups take precedence over dims), slot
  slices, and operator args;
* detects: destination conflicts, unmatched source tensors (error/warn
  policy), required rules that matched nothing, shape mismatches
  (`expect_shape`), coverage failures (`per_layer_required` /
  `per_layer_alternatives` over the layer range);
* emits a dry-run summary (`convert --dry-run`) with every job, drop, and
  estimate — so mapping mistakes never require a real conversion to surface.

### 5. Runner (`runner.py`) + Quantizer (`quantize.py`)

Executes jobs in plan order under a strict memory contract:

* per job: read slot slices (bounded), run operator steps, quantize
  (`mx.quantize`, weights→f16→affine) or cast, hand to the writer, release;
* jobs whose pipeline is all-elementwise and slice-free are eligible for the
  row-chunked streaming path (`chunk_elements`, default ~512 MiB of f32) —
  used for vocab-sized embeddings/lm_heads;
* tracks wall time and peak RSS (`ru_maxrss`) per job and overall.

### 6. Writer (`writer.py`)

Buffers tensors into shards, flushing at `max_shard_bytes`; finalizes with
`model-0000N-of-0000M.safetensors` naming, `model.safetensors.index.json`,
the MLX-LM `config.json` (model_type + optional nested `text_config` +
quantization entries; reference config merge with drop-field policy), and
tokenizer file copying.

### 7. Verifier (`verifier.py`)

Verification is part of the product, not an afterthought:

* index/key-set parity with the plan (including `.scales`/`.biases`
  companions);
* per-tensor shape checks and global NaN/inf checks;
* numeric spot checks: each sampled job is recomputed from the GGUF through
  the same operator pipeline and compared to the saved tensor (dequantized
  first for quantized outputs) with a scale-aware tolerance (one quant step
  of the tensor's value range, floored per bit width);
* optional `mlx_lm.load()` + short generation smoke test;
* structural quantization checks (packed uint32 / f16 scales / f16 biases
  shapes) for every quantized tensor.

## Design invariants

1. **Bounded memory** — no step materializes more than one tensor/chunk of
   dequantized weights; no full-FP16 staging anywhere.
2. **Standard MLX-LM output** — not a private runtime format.
3. **Declarative mapping** — architecture knowledge lives in YAML + small
   named plugins; the engine has zero model-specific code.
4. **Generic operators first** — a plugin is added only when generic ops
   cannot express a transform clearly.
5. **Fail before weights move** — the planner turns config/mapping mistakes
   into cheap errors.
