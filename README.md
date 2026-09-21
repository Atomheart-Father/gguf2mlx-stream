# gguf2mlx-stream

**Declarative, bounded-memory GGUF → MLX-LM checkpoint transcoder.**

Convert llama.cpp GGUF models into standard MLX-LM checkpoints on Apple
Silicon using tensor-bounded streaming dequantization: one quantized GGUF
tensor (or one row-chunk of a huge tensor) is read at a time, dequantized
in bounded chunks, transformed, quantized, written to the shard, and
released. A complete FP16/BF16 checkpoint is never materialized.

("Streaming" here means bounded-memory *tensor* streaming — not token-level
streaming.)

```python
from mlx_lm import load
model, tokenizer = load("./my-model-mlx-4bit")   # standard MLX-LM output
```

---

## Why this exists

The naive conversion path — load the whole GGUF dequantized to
FP16/BF16, then quantize to MLX — needs roughly:

```
source bytes  +  full FP16 copy  +  quantized output
```

simultaneously in memory. For a 9B model that is ~18 GiB of weights alone
plus the quantized output on a 24 GB machine: instant swap, often OOM.

`gguf2mlx-stream` never builds that intermediate. Per tensor (or row-chunk):

```
GGUF quantized tensor/chunk  (mmap, quantized bytes)
        ↓  bounded dequantization
float32 tensor or row-chunk
        ↓  semantic transform (declarative op pipeline)
transformed float32
        ↓  MLX affine quantization
packed uint32 + f16 scales/biases
        ↓  write shard
safetensors shard (flushed at a size limit)
        ↓  release
```

## Supported architectures

| family | config | highlights |
|---|---|---|
| `qwen3_5` | `configs/qwen3_5.yaml` | Qwen3.5 hybrid GDN + full attention; generic grouped v-head reorder for any heads/kv-heads ratio 1–4; NextN/MTP block removal via block-range drop rules; fused q\|k\|v; full-attention q+gate fusion pass-through; `A_log = log(−unpermute(ssm_a))`; conv1d `(dim,k) → (dim,k,1)`; output config nested under `text_config` |
| `qwen3` | `configs/qwen3.yaml` | dense transformer; tied-embedding aware (`tie_word_embeddings` derived from `output.weight` presence); q/k-norm pass-through |
| `llama` | `configs/llama.yaml` | Llama family; undoes the llama.cpp convert-time q/k out-axis storage permutation with a generic reshape→permute→reshape chain; drops the derived `rope_freqs.weight` buffer |
| `gemma3` | `configs/gemma3.yaml` | Gemma 3 text; subtracts the llama.cpp-baked `+1` from all RMSNorm weights (mlx-lm `gemma3_text` re-adds 1.0 at runtime); sliding/global attention pattern literals in output config |

Configs are data: matching, transforms, and output metadata are declared in
YAML and compiled into a validated plan. See docs/CONFIG_SPEC.md.

## Integration evidence

Release validation ran the full stage pipeline for **8 real-GGUF
conversions (4 families × Q4_K_M + Q6_K), all PASS**:

```
convert → structural checks → verify → mlx_lm.load → chat generation (temp 0)
        → llama.cpp source comparison
```

All 8 converted outputs were discovered by an isolated oMLX server
(`omlx-cli serve`, dedicated port, its own model directory) with working
`/v1/chat/completions` for the per-family probe models.

Measured on Apple M4 Pro (24 GB), mlx-lm 0.31.3:

| model | quant | source GiB | output GiB | peak RSS GiB | convert s |
|---|---|---|---|---|---|
| Qwen3.5-0.8B (hybrid) | Q4_K_M | 0.50 | 0.40 | 4.22 | 90.4 |
| Qwen3.5-0.8B (hybrid) | Q6_K | 0.60 | 0.57 | 4.45 | 92.8 |
| Qwen3-0.6B | Q4_K_M | 0.37 | 0.31 | 3.40 | 55.2 |
| Qwen3-0.6B | Q6_K | 0.46 | 0.45 | 3.48 | 54.3 |
| Llama-3.2-1B | Q4_K_M | 0.75 | 0.65 | 4.20 | 34.6 |
| Llama-3.2-1B | Q6_K | 0.95 | 0.94 | 4.29 | 35.4 |
| Gemma-3-270M | Q4_K_M | 0.24 | 0.14 | 3.49 | 56.4 |
| Gemma-3-270M | Q6_K | 0.26 | 0.20 | 3.54 | 56.5 |

Numbers come from the converter's own `--report-json` statistics
(regenerated locally by `scripts/run_integration_matrix.py` into
`reports/` and the integration working directory; run artifacts are not
committed). Peak RSS includes the mmap'd source page
cache, which the OS can evict. See docs/TESTING.md for how the matrix is
reproduced.

## Scope boundaries

* **One source GGUF per conversion.** Tokenizer files and reference-config
  augmentation come from a single declared sibling source directory
  (`--source-config` / `--tokenizer-source`). No multi-source merging is
  performed or planned.
* **Qwen3.8 is not a supported or claimed target.** No Qwen3.8 config
  exists and none is implied. Qwen3.5-architecture distills (e.g. 2B-class)
  may convert via the `qwen3_5` config; that is a compatibility item to
  verify per model, not a claim.

## Quickstart

```bash
pip install gguf2mlx-stream            # wheel: the four official configs are built in
pip install -e '.[loadtest]'           # source checkout + mlx-lm for the load/generation contract

# list the built-in architecture configs shipped with the package
gguf2mlx-stream list-configs

# optional: fetch the pinned small integration GGUFs + tokenizers (network)
python scripts/fetch_integration_models.py

# inspect a GGUF (metadata + tensor inventory)
gguf2mlx-stream inspect ~/models/qwen.gguf --tensors

# compile the conversion plan without touching tensor data (dry-run/plan view)
gguf2mlx-stream convert ~/models/qwen.gguf --arch-config qwen3_5 --dry-run

# convert (bounded memory; transactional output; standard MLX-LM dir)
gguf2mlx-stream convert ~/models/qwen.gguf \
  --arch-config qwen3_5 \
  --output ./my-model-mlx-6bit \
  --bits 6 --group-size 64 --mode affine \
  --source-config ~/models/qwen-hf/config.json \
  --tokenizer-source ~/models/qwen-hf

# verify output against source: every quantized tensor is numerically
# recomputed and compared; shapes, finiteness, index/shard parity, and the
# output's recorded quantization parameters are all checked
gguf2mlx-stream verify ~/models/qwen.gguf ./my-model-mlx-6bit \
  --arch-config qwen3_5 --bits 6
```

`--arch-config` accepts a built-in config name (no clone needed), a YAML
path, or nothing at all — when omitted, the config is auto-detected from the
GGUF's `general.architecture` if exactly one built-in config accepts it.
`--dry-run` prints the full plan (every job, drop, and estimate) so mapping
mistakes are caught before any weights move.

The output contract is `mlx_lm.load(path)`. A conversion fails
transactionally — never replacing an existing output — when the tokenizer
source cannot supply a loadable tokenizer (`tokenizer.json` or
`tokenizer.model`, plus `tokenizer_config.json`).

## The non-negotiable memory rule

Never implement the primary conversion path as
`GGUF → complete FP16/BF16 checkpoint → MLX quantization`. The pipeline is
always:

```
read quantized GGUF tensor/chunk → bounded dequantization → transform
→ MLX quantize → write shard → release
```

## Memory expectations

* Bounded by the largest single tensor (dequantized float32) plus the
  current shard buffer (~`max_shard_bytes`, default 4 GiB).
* Huge vocab-embedding jobs are processed in row chunks (`--chunk-mb`,
  default 512 MiB of float32 per chunk) — only for pipelines whose
  operators are elementwise (chunk-safe) and slice-free.
* Peak RSS includes the mmap'd source file's page cache, which the OS can
  evict; it is not "live" memory.

## Limitations

* GGUF → MLX only (no reverse direction yet).
* Tokenizer files are copied from a source directory; GGUF-embedded
  vocabularies are not rebuilt into `tokenizer.json`.
* Conv1d/Norm etc. non-quantized outputs are float32 (float16 opt-in per rule).
* One tensor is never split across shards; a single tensor larger than the
  shard limit still lands in its own shard.
* llama.cpp (`llama-completion`) is an *optional* external semantic
  cross-check — this tool never requires it.
* No canonical public repository URL is advertised yet; the project is
  distributed as a local/private package until a real upstream exists.

## Development

```bash
pip install -e '.[dev]'
python -m pytest tests/ -q    # unit + oracle + synthetic pipeline + structural parity
ruff check src tests scripts
```

## License

MIT — see LICENSE. GGUF container/quant handling delegates to the
[`gguf`](https://pypi.org/project/gguf/) package (MIT, part of llama.cpp).
