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

## Practical guidance (read this first)

* GGUF → MLX **streaming conversion is supported** for the architectures
  listed below, with **bounded-memory, chunked conversion** — a complete
  FP16/BF16 checkpoint is never materialized.
* `--bits auto` (the default) picks the MLX bit magnitude matching the
  source GGUF's byte-dominant quant family: **IQ2/Q2 → 2, IQ3/Q3 → 3,
  IQ4/Q4 → 4, Q6 → 6, Q8 → 8**. An explicit `--bits` always wins.
* **4-bit or higher is recommended for practical model quality.**
* **3-bit is supported but can cause substantial quality degradation.**
  Converting an auto-derived 3-bit target works out of the box and prints a
  fidelity warning; the evidence behind that warning is summarized in
  [Validation / Quantization Fidelity](#validation--quantization-fidelity)
  below.
* GGUF quantization and MLX affine quantization are **different encodings**:
  "same-bit" means the same bit *magnitude*, **not** bit-for-bit equivalent
  weights. This project does not claim lossless quantized conversion.
* What *is* proven exact is the **unquantized path**: BF16 GGUF → BF16 MLX
  was verified tensor-identical (after float16 normalization) against
  official exports on two architectures — see the validation section.

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
| `qwen35moe` | `configs/qwen3_5_moe.yaml` | Qwen3.5-MoE (hybrid GDN + full attention + 256-expert sparse MoE); everything from `qwen3_5` plus N-D (3-D expert) quantization with chunk-safe row streaming; config-declared per-rule bits/group_size overrides emitted as mlx-lm per-module keys (router and shared-expert gate at 8 bits); GGUF metadata-array passthrough (`rope.dimension_sections` → `mrope_section`) |
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

Additional large-model regression (`qwen35moe`, run outside the pinned
matrix): JoyFox Qwen3.6-35B-A3B-RP-Aggressive (hybrid GDN + full attention
+ 256-expert MoE). **The historical 4-bit conversion below is superseded
history — it used the wrong target for an IQ3-dominant source and is not a
recommended result:**

| source GiB | output GiB (shards) | peak RSS GiB | convert s | verify |
|---|---|---|---|---|
| 14.72 | 18.17 (5) | 16.48 | 428 | ALL OK (733 numeric / 733 shape / 1757 finite) |

That run exercised N-D (3-D expert) quantization with chunk-safe streaming
and config-declared per-rule 8-bit overrides for the MoE router and
shared-expert gate, and loaded/produced coherent chat generations.

The current conversion of this source is the **3-bit `--bits auto` run**
(14.14 GiB / 4 shards, verify ALL OK). Its capability-gate result was
**FAIL** for that source class (see `eval/bench/results/`). Converting
IQ3/Q3 sources with `--bits auto` therefore emits a fidelity warning
(today) — the conversion itself proceeds; see the section below.

## Validation / Quantization Fidelity

Three separate questions are validated with separate evidence. **Do not
conflate them:** the unquantized path is proven exact; the quantized path is
numerically equivalent to mlx-lm but inherits the fidelity of the target
grid, and our paired-oracle experiments observed a strong 3-bit cliff.

### 1. Converter correctness — BF16 paired-oracle proofs

Method: convert the *same pinned BF16 GGUF* the official MLX export was
built from, compare every tensor (`research/paired_oracle/compare_bf16.py`).

| architecture | reference tensors matched | after f16 normalization | residual differences |
|---|---|---|---|
| Qwen3.5-0.8B (hybrid GDN) | **320/320** | 241 bit-equal + 79 norm tensors | all 79 are the *official export* rounding f32 norms to bf16; we keep f32 (strictly more faithful) |
| Llama-3.2-1B (dense) | **146/146** | **146/146 equal** | none |

Proven at BF16: tensor mapping, attention/GDN v-head reordering,
`A_log = log(−unpermute(ssm_a))`, conv1d layout, NextN/MTP block drop, tied
`lm_head`, and `config.json` semantics (27 common fields; our decoded
`eos_token_id` is the correct instruct token while the official export's
contradicts its own tokenizer config). The 153 reference-only tensors for
Qwen3.5-0.8B are `vision_tower.*` — the official repo is a VLM export; this
converter is text-only by design.

**Verdict: the transcoder mapping itself is correct; subsequent quantized
differences are attributable to quantization, not to conversion.**
Details: [research/paired_oracle/REPORT_BF16_ORACLE.md](research/paired_oracle/REPORT_BF16_ORACLE.md).

### 2. Quantizer equivalence with mlx-lm

Converting the pinned BF16 GGUF at uniform 3-bit reproduces the *official
MLX 3-bit* per-module error profile to **4+ significant digits** (e.g.
GDN `in_proj_b` rel-L2 0.20442 vs 0.20443): the MLX quantization path here
is numerically equivalent to mlx-lm's own requantization.

Requantizing an existing Q3_K_M GGUF to MLX 3-bit ("double quantization")
adds +6%…+13.5% per-module error, but behaviorally only **+0.03 nats/token**
— the 3-bit *grid* loses the same information either way.

### 3. 3-bit fidelity findings (paired-oracle calibration, Qwen3.5-0.8B)

All candidates built from the same Q3_K_M GGUF (except the BF16→3-bit
control); reference = official instruct BF16 (NLL 3.1375, ARC letter 41%).
Level-1 gate = teacher-forced NLL/KL on pinned wikitext windows + ARC-100
multiple-choice log-likelihood, no generation involved.

| candidate | size | Δ CE (nats/token) | KL | ARC letter | verdict |
|---|---|---|---|---|---|
| uniform 3-bit (g64) | 341 MB | +1.007 | 0.956 | 34% | **3-bit cliff** |
| mixed 3/4 (official-analog) | 382 MB | +0.769 | 0.699 | 26% | still 3.3× worse KL than 4-bit |
| mixed 3/4 (sensitivity-informed) | 363 MB | +0.876 | 0.836 | 23% | worse than official-analog |
| mixed 3/6 (official-analog) | 465 MB | +0.687 | 0.615 | 36% | bigger *and* worse than 4-bit |
| mixed 3/6 (sensitivity-informed) | 343 MB | +1.019 | 0.963 | 36% | ≈ uniform 3-bit |
| **uniform 4-bit (g64)** — official repo anchor | 447 MB | **+0.213** | **0.157** | 36% | **recommended** |
| BF16 → 3-bit control (no double quantization) | 341 MB | +1.037 | 0.950 | 27% | isolates the grid as the cause |

Observed conclusions from these experiments (stated as evidence, not as a
universal claim about every model):

* **A strong fidelity cliff at 3-bit**: ≈ +1.0 nat/token (≈ 2.8× perplexity)
  regardless of whether the source is BF16 or Q3_K_M.
* The cliff is **primarily attributable to the MLX affine 3-bit grid**, not
  to double quantization (+0.03 nats) and not to the transcoder (§1, §2).
* **Every 3-bit-containing profile tested is Pareto-dominated by plain
  4-bit** — mixed 3/6 spends more bytes than uniform 4-bit and is still
  3.2× worse in KL.
* **Uniform 4-bit is clearly the reliable choice**; 6-bit improves further.
* A QAT control pair (YoozLabs Qwen3.5-0.8B) decodes as
  same-weights-different-format and confirms the thesis from the other
  side: low-bit fidelity comes from grid-aligned QAT, not from PTQ.

Because of this evidence, `--bits auto` for IQ3/Q3 sources prints the
following warning while still converting:

> MLX affine 3-bit conversion is supported, but our paired-oracle
> experiments show substantial fidelity degradation at 3-bit. The
> degradation is primarily attributable to the MLX affine 3-bit quantization
> grid rather than the GGUF→MLX transcoder. For practical model quality,
> 4-bit or higher is recommended.

### 4. Generation-based capability gates (earlier experiments)

ARC-Challenge-100 strict gate on converted outputs (recorded params,
hash-verified prompts; full reports in `eval/reports/` and
`eval/bench/results/`):

Llama-3.2-1B-Instruct (Q4_K_M source), same source at several targets:

| target | clean accuracy | anomaly rate | verdict |
|---|---|---|---|
| source GGUF (reference) | 48.0% | 0.0% | — |
| 3-bit g32 | 31.0% | 3.0% | FAIL (−17 pp) |
| 3-bit g64 | 29.0% | 8.0% | FAIL (−19 pp) |
| 3-bit g128 | 31.0% | 9.0% | FAIL (−17 pp) |
| **4-bit g64** | **47.0%** | 1.0% | **PASS** |
| **6-bit g64** | **55.0%** | 0.0% | **PASS** |

Qwen3.6-35B-A3B (IQ3_M source) at `--bits auto` → 3-bit: source 90.0% vs
3-bit clean accuracy 55.0% (anomaly rate 41% vs 5%) — **FAIL**; the same
source at 4-bit/6-bit passes. All four Q4_K_M/Q6_K families in the
integration matrix above (§ "Integration evidence") load, generate coherently,
and serve through oMLX.

### Where the raw data lives

* Paired-oracle study: `research/paired_oracle/REPORT_PAIRED_ORACLE.md`
  (+ `REPORT_BF16_ORACLE.md`), profiles in
  `research/paired_oracle/profiles/`, pinned-asset manifest in
  `research/paired_oracle/manifest.py` (16 pinned assets, 16.65 GB; weights
  and dataset text are never committed).
* Capability gates: `eval/bench/results/FINAL_REPORT.md`,
  `eval/bench/results/*/`, `eval/reports/*/`.
* Reproduce the converter-vs-official comparisons with the tools in
  `research/paired_oracle/` (`manifest.py` pins every revision + sha256).

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
pip install gguf2mlx-stream            # wheel: the five official configs are built in
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

# convert with target-bits auto-selection (the default): the byte-weighted
# dominant source quant family picks the global MLX bit magnitude
# (IQ2->2, IQ3->3, IQ4/Q4->4, Q6->6, Q8->8); sources whose dominant family
# has no MLX affine equivalent (Q5/IQ1/TQ) fail with an explicit error.
# An auto-derived 3-bit target prints a fidelity warning and still converts.
gguf2mlx-stream convert ~/models/iq3-model.gguf --output ./out-3bit-auto

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

`--bits` defaults to `auto`: the target bit magnitude is derived from a
byte-weighted histogram of the source quantization families over the tensors
the plan will quantize, and the evidence (histogram, dominant type, reason)
is recorded in the dry-run view, `--report-json`, and the output
`config.json` under `quantization_selection`. The MLX output is one global
bit magnitude plus the config's declared per-rule overrides — mixed source
quantization is not replicated per tensor, and an unmappable dominant family
refuses to convert rather than guessing. Note that "source IQ3" and "MLX
affine 3-bit" are the same *target bit magnitude*, not bit-for-bit
equivalent encodings. When `auto` resolves to 3 bits, a fidelity warning is
printed to stderr and recorded in the output `config.json`
(`quantization_selection.fidelity_warning`); the conversion itself proceeds —
see [Validation / Quantization Fidelity](#validation--quantization-fidelity).

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
