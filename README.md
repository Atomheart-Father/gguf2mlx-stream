# gguf2mlx-stream

**Declarative, bounded-memory GGUF → MLX-LM checkpoint streaming transcoder.**

Convert llama.cpp GGUF models into standard MLX-LM checkpoints on Apple
Silicon without ever materializing a full dequantized copy of the model.

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

`gguf2mlx-stream` never builds that intermediate. It processes **one tensor
(or one row-chunk of a huge tensor) at a time**:

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

Reference numbers for a 9B hybrid model on an M4 Pro (24 GB):
peak RSS ≈ 13–15 GiB (including evictable mmap page cache), conversion time
≈ 2 minutes, 6-bit output ≈ 6.8 GiB, 4-bit output ≈ 4.7 GiB.

## Supported

| | |
|---|---|
| GGUF inputs | `Q4_K` (incl. Q4_K_M), `Q6_K`, `Q8_0`, `F32`, `F16`, `BF16` |
| MLX output quantization | affine, bits 2/3/4/6/8, any group size dividing the row |
| Architectures | `qwen3_5` (hybrid GDN + full attention) via `configs/qwen3_5.yaml` |
| Output | standard MLX-LM directory (config.json, sharded safetensors, index, tokenizer files) |

**Tested models:** Qwen3.5-text 9B dense-hybrid finetunes (the converted
outputs were verified numerically against the original HF BF16 checkpoint
and load with `mlx_lm.load()`).
**Untested:** everything else. The engine is architecture-agnostic, but only
the Qwen3.5 config has been exercised end-to-end. See docs/ROADMAP.md.

## CLI

```bash
# inspect a GGUF (metadata + tensor inventory)
gguf2mlx-stream inspect model.gguf --tensors

# list registered transformation operators
gguf2mlx-stream list-ops

# validate an architecture config before any conversion
gguf2mlx-stream validate-config configs/qwen3_5.yaml

# compile the conversion plan without touching tensor data
gguf2mlx-stream convert model.gguf --arch-config configs/qwen3_5.yaml --dry-run

# convert (bounded memory; output is a standard MLX-LM dir)
gguf2mlx-stream convert model.gguf \
  --arch-config configs/qwen3_5.yaml \
  --output ./model-mlx-6bit \
  --bits 6 --group-size 64 --mode affine \
  --source-config /path/to/original/config.json \
  --tokenizer-source /path/to/original/dir

# verify output against source: coverage, shapes, finiteness, numeric spot checks
gguf2mlx-stream verify model.gguf ./model-mlx-6bit --arch-config configs/qwen3_5.yaml --bits 6
```

`--source-config` supplies the original HF/text `config.json` (used for
architecture dimensions that GGUF metadata may not carry and for output
config generation). `--tokenizer-source` names the directory to copy
tokenizer files from; it defaults to the GGUF's own directory.

## Adding a new architecture

In order of preference:

1. **Write a config.** Copy `configs/qwen3_5.yaml`, adjust the tensor
   rules/dims/output mapping. Most "simple layout" architectures need
   nothing else. Run `validate-config` and `--dry-run` to iterate cheaply.
2. **Reuse operators.** 19 generic operators (copy/cast/reshape/transpose/
   slice/concat/zip-unzip blocks/neg/log/exp/add/sub/mul/div/…) cover most
   transformations. See `gguf2mlx-stream list-ops`.
3. **Only if truly unavoidable, add a plugin operator**: a small, named,
   deterministic, tested function registered under a fixed name, referenced
   from YAML by that name only. Configs can never import or execute code.

See docs/CONFIG_SPEC.md for the full config grammar and
docs/ARCHITECTURE.md for how the pieces fit.

## Memory expectations

* Bounded by the largest single tensor (dequantized float32) plus the
  current shard buffer (~`max_shard_bytes`, default 4 GiB).
* Huge vocab-embedding jobs are processed in row chunks (`--chunk-mb`,
  default 512 MiB of float32 per chunk).
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

## Development

```bash
pip install -e '.[dev]'
pytest                       # unit + synthetic pipeline + structural parity
ruff check src tests
```

Optional local integration tests (real GGUF regression) are env-gated and
skipped when the assets are absent — see tests/test_integration_nyx.py.

## License

MIT — see LICENSE. GGUF container/quant handling delegates to the
[`gguf`](https://pypi.org/project/gguf/) package (MIT, part of llama.cpp).
