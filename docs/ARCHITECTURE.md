# Architecture

gguf2mlx-stream is a small compiler-like pipeline:

```
                 YAML (declarative architecture config)
                                │  validate-config (schema)
                                ▼
GGUF file ──► GGUFSource ──► Planner ──► ConversionPlan ──► ConversionRunner ──► MLX-LM dir
             (mmap,              (match,        (jobs)          (bounded stream)     (shards +
              dequant)            resolve,                       └─ quantize.py      index +
                                  check)                         └─ writer.py        config)
                                                                 (transactional swap)
```

## Module map

| module | responsibility |
|---|---|
| `source/gguf.py` | GGUF reading + bounded dequantization (no model knowledge) |
| `config/schema.py` | config dataclasses, parsing, static schema validation |
| `planner.py` | `DimResolver`, plan compilation, pre-weight validation incl. pipeline shape inference |
| `ops/generic.py`, `ops/registry.py`, `ops/base.py` | generic operators + registry (every op carries a plan-time shape-inference function) |
| `ops/plugins/qwen35.py` | thin architecture plugin (delegates to a generic op) |
| `quantize.py` | MLX affine quantization |
| `writer.py` | sharding, index, config.json emission, tokenizer copying + tokenizer output contract |
| `runner.py` | bounded execution loop, transactional output staging |
| `verifier.py` | post-conversion verification (full numeric coverage by default) |
| `builtin.py` | built-in architecture config discovery (wheel-packaged `configs/`, repo fallback, auto-detect) |
| `cli.py` | `inspect` / `list-ops` / `list-configs` / `validate-config` / `convert` / `verify` |

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
* dims: named numbers resolved from `gguf:<key>`, `ref:<path>` (optional
  reference config), `tshape:`/`has:` probes, constants, or a closed
  arithmetic vocabulary (`mul/add/sub/div/not`); list values are ordered
  fallback chains; `{list: [...]}` is the literal-list escape hatch;
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
* Each op declares a `kind` — **elementwise / shape / combine / permute**.
  Only elementwise ops are row-chunk streaming-safe; shape and permute jobs
  run whole-tensor (the planner only selects the chunked path when every
  step is elementwise and the job is slice-free).
* Built-ins are registered at import; plugins live in `ops/plugins/` and are
  imported (i.e. registered) by the package — never dynamically by configs.
* The `qwen35` plugin (`qwen35_v_head_unpermute`) is a thin wrapper: it
  resolves head geometry from the architecture dims and delegates the data
  movement to the generic `reorder_grouped_heads` operator (ratio 1–4;
  ratio=1 is identity, ratio=2 equals `unzip_blocks`).

#### Worked example: generic operators express family transforms

**Llama q/k unpermute.** llama.cpp bakes a convert-time storage permutation
into the OUT axis of `attn_q`/`attn_k`: the natural row
`d + (head_dim/2)*c` within each head block is stored at `2*d + c`. The
inverse needs no family-specific code — a reshape → permute → reshape chain
of generic operators (see `configs/llama.yaml`):

```
rows (n_heads, head_dim, hidden)
  → reshape  (n_heads, head_dim/2, 2, hidden)   # split stored row index
  → permute  (0, 2, 1, 3)                       # swap the halves back
  → reshape  (n_heads*head_dim, hidden)         # natural order
```

(`attn_k` is identical with `n_kv_heads`; `attn_v`/`attn_output` are not
permuted. The derived `rope_freqs.weight` buffer is dropped by a drop rule.)

**Grouped v-head reorder.** llama.cpp stores GDN v-head tensors in
group-interleaved order `concat(natural[j::ratio] for j in range(ratio))`
over v-head blocks, where `ratio = value heads / kv-head groups`. The
generic `reorder_grouped_heads(axis, block, ratio)` operator restores
natural order for any ratio 1–4; the `qwen35` plugin only resolves
`ratio`/`block` from the dims and forwards.

### 4. Planner (`planner.py`)

Compiles config + source into a `ConversionPlan` *before* reading weights:

* `DimResolver` resolves dims and all scalar specs (failing fast on missing
  metadata/refs; fallback chains, literal lists, nested mappings);
* substitutes `{dim}` placeholders in match patterns, compiles anchored
  regexes, matches every GGUF tensor to the first matching rule, and
  compiles range-drop rules per block index: the rule's own match template
  (with the reserved `{i}` placeholder) declares the name pattern for each
  index in `[start, end)` — no name prefix is hardcoded;
* resolves dest templates (match groups take precedence over dims), slot
  slices (axis-0 → row ranges for bounded reads), and operator args;
* validation before any tensor data moves: destination conflicts, unmatched
  source tensors (error/warn policy), required rules that matched nothing,
  shape mismatches (`expect_shape`), coverage failures
  (`per_layer_required` / `per_layer_alternatives` over the layer range),
  and arch-identifier checks;
* **pipeline shape inference**: every registered operator carries a
  shape-inference function; the planner walks each job's operator pipeline
  with shape tuples only (source shapes, slot slices, resolved args) and
  verifies every step's parameters (axes in range, divisibility, reshape
  products, concat compatibility, permutation validity) and the final
  output shape — invalid dims/rank/args surface as `PlanError` at plan
  time, not as raw `TypeError`/`IndexError`/`ValueError` mid-conversion.
  Quantized rules must produce 2-D outputs;
* emits a dry-run summary (`convert --dry-run`) with every job, drop, and
  estimate — so mapping mistakes never require a real conversion to surface.

### 5. Runner (`runner.py`) + Quantizer (`quantize.py`)

Executes jobs in plan order under a strict memory contract:

* per job: read slot slices (bounded), run operator steps, quantize
  (`mx.quantize`, weights→f16→affine) or cast, hand to the writer, release;
* jobs whose pipeline is all-elementwise and slice-free are eligible for the
  row-chunked streaming path (`chunk_elements`, default ~512 MiB of f32) —
  used for vocab-sized embeddings/lm_heads;
* output is **transactional**: everything is written into a staging sibling
  directory (`<out>.tmp-<uuid>`), then atomically swapped into place only
  after shards, index, config.json, and tokenizer files are complete. A
  failure never leaves partial artifacts, and an existing output is never
  replaced without `--overwrite`;
* tracks wall time and peak RSS (`ru_maxrss`) per job and overall
  (`--report-json`).

### 6. Writer (`writer.py`)

Buffers tensors into shards, flushing at `max_shard_bytes`; finalizes with
`model-0000N-of-0000M.safetensors` naming, `model.safetensors.index.json`,
the MLX-LM `config.json` (flat or nested under `text_config` per the
config's emission contract + quantization entries; reference config merge
with drop-field policy; `required_fields` presence check), and tokenizer
file copying.

The **tokenizer output contract** is enforced here and in the runner: a
conversion may only succeed if the output directory will contain one
vocab-capable tokenizer file (`tokenizer.json` or `tokenizer.model`) plus
`tokenizer_config.json`. The check runs twice — on the tokenizer source
before any tensor is read, and on the staged output before the transactional
commit — so a conversion without a usable tokenizer fails and never replaces
an existing output.

### 7. Built-in configs (`builtin.py`)

The repository-root `configs/` directory is the single authoritative source;
hatchling force-includes it into the wheel as `gguf2mlx_stream/configs/`.
Discovery order: packaged resource (installed wheel) → repository-root
`configs/` (editable/source checkouts). `--arch-config` accepts a built-in
name, an explicit YAML path, or nothing (auto-detect from the GGUF's
`general.architecture` when exactly one built-in config accepts it).

### 8. Verifier (`verifier.py`)

Verification is part of the product, not an afterthought:

* bidirectional index/shard key-set parity: every index entry exists in its
  shard, every key in an indexed shard is listed in the index, and every
  `*.safetensors` file is referenced by the index — unindexed tensors, stale
  index entries and unreferenced shard files are all rejected;
* key-set parity with the plan (including `.scales`/`.biases` companions);
* quantization parameters (`bits`, `group_size`, `mode`) are read from the
  output's `config.json` and validated; explicit CLI values conflicting
  with the recorded metadata are a hard failure, and weights that cannot be
  dequantized under the recorded parameters are a reported failure;
* per-tensor shape checks and global NaN/inf checks;
* numeric checks with **full coverage by default**: every quantized tensor
  is recomputed from the GGUF through the same operator pipeline and
  compared to the saved tensor (dequantized first) with a scale-aware
  tolerance (one quant step of the tensor's value range, floored per bit
  width). Sampling (first tensor per rule + small tensors) is an explicit
  opt-in (`--sampled`);
* structural quantization checks (packed uint32 / f16 scales / f16 biases
  shapes) for every quantized tensor;
* optional `mlx_lm.load()` + short generation smoke test.

## Execution / memory contract

1. **Bounded memory** — no step materializes more than one tensor (or one
   row-chunk of a huge tensor) of dequantized weights; no full-FP16 staging
   anywhere.
2. Evidence: measured peak RSS across the 4-family integration matrix is
   3.4–4.5 GiB for 0.27–1B models (including evictable mmap page cache);
   the 9B Qwen3.5 reference conversion peaks around 13–15 GiB — far below
   the ~18 GiB a full FP16 staging of that model alone would require.
3. Peak RSS includes the mmap'd source page cache; it is not "live" memory.

## Design invariants

1. **Bounded memory** — per-tensor/chunk buffers only (see above).
2. **Standard MLX-LM output** — not a private runtime format.
3. **Declarative mapping** — architecture knowledge lives in YAML + small
   named plugins; the engine has zero model-specific code.
4. **Generic operators first** — a plugin is added only when generic ops
   cannot express a transform clearly (see the worked examples above: the
   llama permutation and the grouped v-head reorder are generic; the
   `qwen35` plugin is only geometry resolution).
5. **Fail before weights move** — the planner turns config/mapping mistakes
   into cheap errors; the runner turns partial outputs into none.
