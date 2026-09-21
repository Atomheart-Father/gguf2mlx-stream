# Config specification

Architecture configs are YAML data — not programs. The grammar is closed:
no expressions beyond a fixed arithmetic vocabulary, no imports, no shell,
no object construction (`yaml.safe_load` rejects tagged objects), and
operator references are validated against the registry at load time — a
config can never import or execute code.

A config has five sections: `architecture`, `dims`, `rules`, `coverage`,
and `output`, plus the global `unmatched_tensors` option.

## `architecture`

```yaml
architecture:
  id: qwen3_5                # canonical identifier
  aliases: [qwen3_5_text]    # accepted aliases (documentation)
  gguf_arch: [qwen35, qwen3_5_text]   # accepted general.architecture value(s)
  strict_arch: false         # true -> identifier mismatch is an error
  description: >-            # free text
    ...
```

`gguf_arch` is a string **or a list** of accepted GGUF
`general.architecture` identifiers (current llama.cpp identifier first).
Matching is by the GGUF's `general.architecture`: with `strict_arch: false`
a mismatch warns; with `true` it is a plan error. `gguf_arch: null` accepts
any identifier.

## `dims`

Named dimensions used by rules, plugins, and output mapping. Values are
*scalar specs*; resolution is by declaration order and operands may only
reference earlier dims. Anything that fails to resolve to a number is a
plan-time error.

| form | meaning |
|---|---|
| `123` / `0.5` | numeric constant |
| `"gguf:<metadata.key>"` | GGUF metadata; tries `<key>` then `{arch}.<key>` |
| `"ref:<dotted.path>"` | path into the optional `--source-config` JSON |
| `"tshape:<tensor>.<axis>"` | one axis of a source tensor's HF-convention shape |
| `"has:<tensor>"` | boolean: does the source tensor exist |
| `"dim_name"` | reference to a previously declared dim |
| `{mul: [a, b, ...]}` / `{add: [...]}` / `{sub: [...]}` / `{div: [...]}` | closed arithmetic over the above |
| `{not: [...]}` | boolean negation of the single operand |
| `[spec1, spec2, ...]` | **fallback chain** — first spec that resolves wins |
| `{list: [spec, ...]}` | **literal list** — the resolved list itself, not a fallback |
| plain nested mapping | every value is itself a scalar spec (nested output objects) |

`{div: [...]}` is exact integer division when evenly divisible; otherwise it
falls back to float division (contexts that require an integer —
`expect_shape`, slice bounds, reshape/permute args — will then fail at plan
time).

### Fallback chain vs literal list

A plain list in spec position is a **fallback chain**, never a literal
value. Where a literal list is needed (operator args such as reshape shapes
or permute axes), use the explicit `{list: [...]}` form:

```yaml
dims:
  head_dim: [gguf:attention.key_length, ref:head_dim]   # fallback chain

rules:
  - match: "blk\\.(?P<n>\\d+)\\.attn_q\\.weight"
    steps:
      # WRONG: [0, 2, 1, 3] here would be a fallback chain that tries to
      # resolve 0, then 2, ... as scalar specs.
      # RIGHT: a literal list of resolved ints:
      - {op: permute, args: {axes: {list: [0, 2, 1, 3]}}}
```

Elements of a `{list: [...]}` are themselves scalar specs, so dims and
arithmetic compose inside it:

```yaml
- {op: reshape, args: {shape: {list: [n_heads, {div: [head_dim, 2]}, 2, hidden_size]}}}
```

## `rules`

An ordered list. Every GGUF tensor is matched against the rules **in
order**; the first match wins, so put drop rules before generic ones.

```yaml
rules:
  # range drop rule: blocks [n_layers, total_blocks) are dropped explicitly
  # (NextN/MTP removal); end <= start matches nothing (nextn=0)
  - name: mtp_drop
    match: "blk\\.{i}\\..*"
    drop: true
    range: {start: n_layers, end: total_blocks}

  # simple drop rule: makes ignored tensors explicit
  - name: rope_freqs_drop
    match: "rope_freqs\\..*"
    drop: true

  # simple 1:1 map with a shape assertion
  - match: "token_embd\\.weight"
    dest: "model.embed_tokens.weight"
    expect_shape: [vocab_size, hidden_size]

  # captured group -> dest template
  - match: "blk\\.(?P<n>\\d+)\\.ffn_gate\\.weight"
    dest: "model.layers.{n}.mlp.gate_proj.weight"

  # multi-slot job: slice one source, transform a part, combine
  - match: "blk\\.(?P<n>\\d+)\\.attn_qkv\\.weight"
    dest: "model.layers.{n}.linear_attn.in_proj_qkv.weight"
    inputs:
      qk: {slice: {axis: 0, lo: 0, hi: {mul: [2, key_dim]}}}
      v:  {slice: {axis: 0, lo: {mul: [2, key_dim]}, hi: conv_dim}}
    steps:
      - {op: reorder_v_heads, input: v, output: v_nat, args: {axis: 0}}
      - {op: concat, inputs: [qk, v_nat], args: {axis: 0}}
```

Rule fields:

| field | meaning |
|---|---|
| `match` | regex; `{dim}` placeholders are substituted before compiling, then matched **anchored** (`fullmatch`); named groups feed dest templates |
| `dest` | dest template; `{group}` (from named groups, taking precedence) and `{dim}` placeholders |
| `drop: true` | drop matching tensors (explicitly, not silently); must not have `dest` |
| `range` | drop-only: `{start, end}` scalar specs; half-open block-index range over the reserved `{i}` placeholder (see below) |
| `optional: true` | rule may match nothing (default: a required rule matching nothing is an error) |
| `quantize: true/false` | emit MLX affine quantization or plain dtype |
| `dtype` | `float32`/`float16` for non-quantized output |
| `expect_shape` | per-axis shape assertion; each axis is a scalar spec (`{mul: [...]}` allowed) or `null` (wildcard) |
| `inputs` | named slots; `"@"` or `{source: "@", slice: {axis, lo, hi}}` (see below) |
| `steps` | ordered operator pipeline (see below) |

Rule names are optional but must be unique. Unknown keys are rejected.

### Block-range drop rules

`drop: true` plus `range: {start: <spec>, end: <spec>}` restricts a rule to
a **half-open block-index range**. The match pattern must contain the
reserved `{i}` placeholder; the planner compiles the rule to the prefixes
`blk.{i}.` for every `i` in `[start, end)`. `end <= start` matches nothing.
This is how NextN/MTP blocks are removed explicitly: `start` is the real
layer count and `end` the total block count, where
`n_layers = block_count − nextn_predict_layers`. A plain (non-range) drop
rule may not use `{i}`.

### Inputs (slots)

Each slot reads the tensor captured by the rule's match (`source: "@"` —
cross-tensor sources are not supported; single-source conversions only).
An optional `slice` (`axis`, `lo`, `hi` as scalar specs) selects part of it:
axis-0 slices are precomputed as row ranges so only the requested bytes are
read from the mmap; other axes are sliced in memory after a bounded read.

### Steps (the transformation IR)

Each step is `{op, input?, inputs?, args?, output?}`. Operator names must
exist in the registry (validated at load time).

* Default chaining: the **first** step consumes `x` (the whole matched
  tensor) when the rule has no named `inputs`; every subsequent step
  consumes the previous step's output (`_`) unless `input:`/`inputs:` is
  given. After a multi-input step the default is not inferable — the next
  step must name its input explicitly or the config fails validation.
* `input: name` — consume a slot, a step `output`, `"_"` (the previous
  step's output), or `x`.
* `inputs: [a, b, ...]` — multi-input ops (`concat`/`add`/`sub`/`mul`/`div`)
  consume several named values in the listed order.
* `output: name` — bind the result under a name for later steps; otherwise
  the result is the implicit previous output.
* `args` — scalar specs (constants, dim names, arithmetic, `{list: ...}`);
  resolved at plan time against the dims.

A step that transforms part of a fused tensor must bind its result via
`output:` and have later steps consume that name (not the raw slot).

## `coverage`

Post-plan assertions that every layer produced its expected tensors.

```yaml
coverage:
  per_layer_required:
    - model.layers.{layer}.input_layernorm.weight
  per_layer_alternatives:
    - - model.layers.{layer}.self_attn.q_proj.weight
      - model.layers.{layer}.linear_attn.in_proj_qkv.weight
```

`{layer}` expands over `range(n_layers)` (the dim named `n_layers`, which
coverage requires). `per_layer_required` entries must all be produced;
`per_layer_alternatives` groups pass if at least one template in the group
matches.

## `output`

```yaml
output:
  model_type: qwen3_5                    # required at conversion time
  architectures: ["Qwen3_5ForCausalLM"]
  nest_config_under: text_config         # omit for flat config.json fields
  top_level:                             # scalar specs
    tie_word_embeddings: {not: [has:output.weight]}
  text_config:                           # scalar specs; nested mappings allowed
    hidden_size: [gguf:embedding_length, ref:hidden_size]
    rope_parameters:
      rope_theta: [rope_theta]
  required_fields:                       # presence-checked after emission
    - text_config.hidden_size
    - text_config.rope_parameters.rope_theta
  reference_config:
    drop_fields: [architectures, model_name]
  tokenizer_files: [tokenizer.json, tokenizer_config.json]
  max_shard_bytes: 4294967296
```

* `text_config` values are scalar specs (fallback chains allowed); plain
  nested mappings resolve every value and are used for nested output
  objects. When `--source-config` is given, the reference JSON (minus
  `drop_fields`, `architectures`, `model_type`) is the base; resolved specs
  are overlaid on top (mismatches warn).
* `required_fields` are dotted paths presence-checked **after** emission; a
  missing field fails the conversion instead of silently relying on mlx-lm
  defaults. An explicit `null` is a legal value and satisfies presence —
  only *absence* fails.
* Quantization entries (`bits`/`group_size`/`mode`) are added under
  `quantization` when quantization is enabled.
* `tokenizer_files` are copied from `--tokenizer-source` (default: the
  GGUF's own directory); absent files are skipped.

### Flat vs nested emission contract

The flat-vs-nested shape of `config.json` is an explicit, per-family
compatibility decision:

* `llama`, `qwen3`, and `gemma3_text` outputs require **flat** text-model
  fields at the top level of `config.json` (omit `nest_config_under`).
* `qwen3_5` nests all text-model fields under `text_config`
  (`nest_config_under: text_config`) — required by the current mlx-lm
  (0.31.x) line for `model_type: qwen3_5`.

Version-specific compatibility is isolated to this choice (and the
`model_type` value); the engine has no model-version-specific code.

## Global option

```yaml
unmatched_tensors: error   # or "warn": what to do with source tensors no rule matched
```

`error` (default) forces configs to account for every tensor — including
explicit `drop` rules for intentionally ignored blocks (e.g. MTP,
`rope_freqs.weight`).

## CLI behavior

```bash
gguf2mlx-stream inspect model.gguf [--tensors]   # metadata + tensor inventory
gguf2mlx-stream list-ops                          # registered operators + kinds
gguf2mlx-stream validate-config configs/qwen3_5.yaml
gguf2mlx-stream convert model.gguf --arch-config cfg.yaml --output out/ [options]
gguf2mlx-stream verify model.gguf out/ --arch-config cfg.yaml [--bits N] [...]
```

`convert` options: `--bits {2,3,4,6,8}`, `--group-size`, `--mode affine`,
`--no-quantize` (float16 weights), `--source-config`, `--tokenizer-source`,
`--max-shard-gb`, `--chunk-mb`, `--report-json`, `-q/--quiet`, and
`--dry-run` — the dry-run/plan view prints every job, drop, unused rule,
unmatched tensor, and the estimated output size without touching tensor
data.

* `validate-config` runs the full static schema validation plus, when a GGUF
  is supplied to `convert --dry-run`, the source-dependent resolution.
* `--overwrite` makes the output **transactional**: everything is staged
  into a temporary sibling directory (`<out>.tmp-<uuid>`) and atomically
  swapped into place only after shards, index, config.json, and tokenizer
  files are complete. A failed or interrupted conversion leaves the previous
  output untouched and leaves no partial artifacts. An existing non-empty
  output directory is never replaced without `--overwrite`.
* `--report-json` writes run statistics (output bytes, shards, dims, peak
  RSS, elapsed time); pair with `-q` for machine-driven runs.
* `verify` re-derives sampled jobs from the source GGUF and compares against
  the written output (see docs/ARCHITECTURE.md); `--load-test` optionally
  runs `mlx_lm.load()` + a short generation.
