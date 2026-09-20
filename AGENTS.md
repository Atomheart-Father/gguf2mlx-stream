# AGENTS.md

Guidance for AI coding agents working in the `gguf2mlx-stream` repository.

## Mission

This repository is an open-source, bounded-memory **GGUF → standard MLX-LM checkpoint transcoder**.

The project is not a collection of model-specific conversion scripts.

Its central abstraction is:

`GGUF tensor semantics → declarative transform plan → MLX tensor semantics`

New model families should normally be supported by:

1. an architecture config under `configs/`;
2. composition of existing registered tensor operators;
3. a small architecture-specific plugin operator only when a transformation cannot be expressed safely and clearly with generic operators.

The current Qwen3.5/Nyx converter is the reference implementation and regression case, not the final architecture of the project.

## Local environment

Development root:

`/Users/bozhongxiao/code`

Repository target:

`/Users/bozhongxiao/code/gguf2mlx-stream`

Golden reference implementation:

`/Users/bozhongxiao/code/gguf2mlx-nyx/`

Relevant reference files:

- `nyx_gguf_to_mlx.py`
- `verify_nyx_mlx.py`

Local model library:

`/Users/bozhongxiao/Models/MLX`

Host machine:

- Apple M4 Pro
- 24 GB unified memory
- memory, not compute, is the primary conversion constraint

Use the existing Python environment under `/Users/bozhongxiao/code`. Do not install into or modify the system Python.

Homebrew `llama.cpp` is available locally. For one-shot source-GGUF verification, `llama-completion` is preferred over interactive `llama-cli`.

## Non-negotiable memory rule

Never implement the primary conversion path as:

`GGUF → complete FP16/BF16 checkpoint → MLX quantization`

For 9B+ models that defeats the purpose of this project and can OOM the host.

The intended execution pattern is:

`read quantized GGUF tensor/chunk → bounded dequantization → transform → MLX quantize → write shard → release`

Keep memory bounded.

Use mmap where appropriate. Release large temporaries promptly. Monitor RSS during local integration tests.

Any change that materially increases peak memory must be justified and measured.

## Repository architecture

Keep these concerns separate:

- GGUF reading/dequantization
- architecture config/schema
- tensor transform IR
- operator registry
- execution planning
- MLX quantization
- safetensors/shard writing
- tokenizer/config emission
- verification
- CLI

Do not place all logic into a single converter script.

Prefer `src/` package layout with small modules and explicit public interfaces.

## Declarative configs

Architecture configs live under:

`configs/`

They describe tensor mapping and transformation, not arbitrary program logic.

Configs may contain:

- architecture aliases
- source tensor match patterns
- destination tensor templates
- ordered operator pipelines
- drop rules
- tensor dependencies/groups
- output config metadata mappings/templates
- references to registered plugin operators

Configs must be schema-validated before conversion starts.

Do not allow embedded Python, shell, arbitrary eval, or unrestricted expressions in config files.

If a config needs procedural control flow, the abstraction is wrong or the behavior belongs in a named operator.

## Operator design

Generic operators should implement reusable tensor semantics.

Examples:

- copy
- drop
- cast
- transpose
- reshape
- squeeze / unsqueeze
- slice
- split
- concat
- permute / reorder blocks
- neg / log / exp
- add / sub / mul / div
- quantize

Each operator should be independently unit-tested.

Where practical, operators should expose:

- input requirements
- output shape behavior
- dtype behavior
- whether the operation is streaming/chunk-safe
- temporary-memory expectations

Do not create architecture-specific operators for behavior that can be cleanly expressed with generic operators.

## Plugin operators

Architecture-specific transforms are allowed through an explicit operator registry.

They must be:

- small;
- named;
- deterministic;
- tested;
- documented;
- referenced from config by name.

They must not provide a back door for arbitrary code execution from YAML.

The first expected plugin is the Qwen3.5 GDN v-head unpermutation.

## Execution planning

Compile the declarative config into a conversion plan before processing large tensors.

The planner should detect, before conversion where possible:

- unmatched required source tensors
- duplicate destination tensors
- invalid shapes
- unresolved dependencies
- unsupported operators
- incompatible quantization settings
- ambiguous rules

For operations needing multiple source tensors, buffer only the minimum group needed to produce the destination tensor.

Provide a dry-run/plan view useful for debugging new architecture configs.

## Output contract

The converter writes standard MLX-LM model directories, not a private runtime format.

A successful output should include the appropriate:

- `config.json`
- tokenizer files
- sharded `*.safetensors`
- `model.safetensors.index.json` when required

The resulting model must load with:

```python
from mlx_lm import load
model, tokenizer = load(path)
```

oMLX compatibility is a valuable integration target, but the core output contract is standard MLX-LM compatibility.

## Current Qwen3.5 reference behavior

The existing Nyx/Qwen3.5 conversion is numerically verified. Preserve these semantics while refactoring.

### RMSNorm convention

For this GGUF layout, llama.cpp has already baked the RMSNorm `+1` convention into the weights used by the proven conversion path.

Do not blindly apply another `+1`.

The currently working MLX layout ships those norm values as-is.

### GDN v-head permutation

The relevant GGUF v-head order corresponds to:

`concat(hf[0::2], hf[1::2])`

over v-head blocks.

The conversion must invert that layout where applicable, including the relevant axes/segments of:

- `in_proj_qkv`
- `in_proj_z`
- `in_proj_a`
- `in_proj_b`
- `ssm_out`
- `conv1d`
- `ssm_dt.bias`

Express this through one small, tested Qwen3.5 plugin operator unless a cleaner generic permutation primitive proves sufficient.

### A_log transform

The GGUF `ssm_a` representation corresponds to:

`-exp(A_log)`

in the GGUF permutation order.

The MLX-side value therefore needs the equivalent of:

`A_log = log(-unpermute(ssm_a))`

Prefer expressing `neg` and `log` with generic operators.

### Structural transforms

The reference path also handles:

- fused q|k|v tensors;
- full-attention `attn_q` fused `[q; gate]`;
- conv1d layout `(dim, k) → (dim, k, 1)`;
- dropping the MTP/next-token-prediction block for the proven inference layout.

Do not treat Qwen3.5 as a Qwen2/Llama layout.

### mlx-lm config compatibility

With the currently installed mlx-lm line, the proven output uses:

- `model_type: "qwen3_5"`
- text-model fields nested under `text_config`

Do not emit `model_type: "qwen3_5_text"` unless the target mlx-lm version is verified to support it.

Keep version-specific compatibility code isolated and documented.

## Verification requirements

Every architecture implementation should have three levels of testing.

### Unit tests

Use tiny synthetic tensors.

Test generic operators, schema validation, mapping, planner behavior, shape logic, and architecture-specific plugins.

### Synthetic pipeline tests

Exercise the complete path with a tiny fixture:

`read → plan → transform → quantize → shard write → reload`

These tests must be CI-friendly and must not require multi-GB checkpoints.

### Optional local integration tests

Large local checkpoints may be used when present, but they must never be committed.

Known local regression assets include the Nyx Q6_K and Q4_K_M source GGUF files and the already-proven MLX outputs under `/Users/bozhongxiao/Models/MLX`.

Never overwrite those reference models.

Integration outputs must go to a distinct temporary/test directory.

## Nyx regression workflow

For the existing Qwen3.5/Nyx case:

1. validate config and mapping coverage;
2. run numeric spot checks against source/reference behavior;
3. convert with bounded memory;
4. verify output sizes are plausible;
5. load using `mlx_lm.load()`;
6. generate through the tokenizer chat template;
7. reject NaN, corrupted text, or immediate EOS failures;
8. optionally compare source GGUF output through `llama-completion` at temperature 0;
9. optionally verify oMLX discovery and `/v1/chat/completions`.

Semantic agreement is required; token-for-token identity is not expected after requantization.

## Known size/memory regression references

For the current 9B Nyx conversion:

- MLX 6-bit output: about 6.78 GiB
- MLX 4-bit output: about 4.69 GiB
- observed peak RSS: roughly 13–15 GiB

If a supposedly quantized path suddenly produces an ~18 GiB intermediate/output or materially larger RSS, investigate whether the implementation accidentally materialized FP16.

## Open-source hygiene

Never commit:

- model weights;
- generated checkpoints;
- API keys;
- tokens;
- secrets;
- local caches;
- user data;
- large temporary artifacts.

Avoid hard-coding personal absolute paths in public library code.

Local integration tests may accept environment variables or documented optional defaults.

Before copying code from another repository, inspect its license and preserve required attribution. Prefer our own clean implementation when practical.

Use type hints on public APIs.

Add tests with behavioral changes.

Keep documentation synchronized with config schema and CLI behavior.

## Scope discipline

The first supported difficult architecture is Qwen3.5.

Do not prematurely add many model families merely to claim broad support.

First prove that the architecture/config/operator abstraction reproduces the existing Qwen3.5 converter correctly and with bounded memory.

Then add a second architecture chosen specifically to stress-test the abstraction.

The goal is a small correct compiler-like core, not a long list of fragile architecture-specific scripts.

## Git / publication

It is fine to initialize a local git repository and make clean local commits.

Do not create a public GitHub repository, push to a remote, or publish a package unless the user explicitly asks for publication.

Before publication, review:

- license
- attribution
- README claims
- supported architecture matrix
- CI
- secret scan
- large-file scan
- local-path leakage

## Current status (for agents picking this up)

* The engine, Qwen3.5 config, and full test suite live here; `66 passed / 3 skipped` is the baseline (`pytest`).
* The original Nyx source GGUFs were deleted from `~/Models/MLX/Nyx-RP-9B-Instruct-2608-v1/`; the proven MLX outputs (`...-MLX-6bit`, `...-MLX-4bit`) remain and anchor the structural regression tests (`tests/test_nyx_parity.py`).
* Real-GGUF integration tests are env-gated (`GGUF2MLX_TEST_Q6_GGUF`, `GGUF2MLX_TEST_Q4_GGUF`, `GGUF2MLX_TEST_SOURCE_DIR`) and report `SKIPPED` when assets are absent.
* Docs: `docs/ARCHITECTURE.md`, `docs/CONFIG_SPEC.md`, `docs/TESTING.md`, `docs/ROADMAP.md`.
