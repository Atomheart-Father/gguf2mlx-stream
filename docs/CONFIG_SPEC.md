# Config specification

Architecture configs are YAML data — not programs. The grammar is closed:
no expressions beyond a fixed arithmetic vocabulary, no imports, no shell,
no object construction (`yaml.safe_load` rejects tagged objects), and
operator references are validated against the registry at load time.

A config has five sections: `architecture`, `dims`, `rules`, `coverage`,
and `output`.

## `architecture`

```yaml
architecture:
  id: qwen3_5                # canonical identifier
  aliases: [qwen3_5_text]    # accepted aliases (documentation)
  gguf_arch: qwen3_5_text    # expected general.architecture; mismatch warns
  strict_arch: false         # true -> mismatch is an error
  description: >-            # free text
    ...
```

## `dims`

Named integer dimensions used by rules, plugins, and output mapping.
Values are *scalar specs*:

| form | meaning |
|---|---|
| `123` | integer constant |
| `"gguf:<key>"` | GGUF metadata; tries `<key>` then `{arch}.<key>` |
| `"ref:<dotted.path>"` | path into the optional `--source-config` JSON |
| `"dim_name"` | reference to a previously declared dim |
| `{mul: [a, b, ...]}` / `{add: [...]}` / `{sub: [...]}` / `{div: [...]}` | closed arithmetic over the above |
| `[spec1, spec2, ...]` | **fallback chain** — first spec that resolves wins |

```yaml
dims:
  hidden_size: [gguf:embedding_length, ref:hidden_size]
  key_dim: {mul: [linear_num_key_heads, linear_key_head_dim]}
  conv_dim: {add: [{mul: [2, key_dim]}, value_dim]}
  mtp_layer: [gguf:block_count, ref:num_hidden_layers]
```

Resolution order is declaration order; operands may only reference earlier
dims. Anything that fails to resolve to an int is a plan-time error.

## `rules`

An ordered list. Every GGUF tensor is matched against the rules **in
order**; the first match wins, so put drop rules before generic ones.

```yaml
rules:
  # drop rule: makes unmatched-by-design tensors explicit
  - name: mtp_drop
    match: "blk\\.{mtp_layer}\\..*"
    drop: true                      # must not have 'dest'

  # simple 1:1 map
  - match: "token_embd\\.weight"
    dest: "language_model.model.embed_tokens.weight"
    expect_shape: [vocab_size, hidden_size]

  # captured group -> dest template
  - match: "blk\\.(?P<n>\\d+)\\.ffn_gate\\.weight"
    dest: "language_model.model.layers.{n}.mlp.gate_proj.weight"

  # multi-slot job: slice one source, transform a part, combine
  - match: "blk\\.(?P<n>\\d+)\\.attn_qkv\\.weight"
    dest: "language_model.model.layers.{n}.linear_attn.in_proj_qkv.weight"
    inputs:
      qk: {slice: {axis: 0, lo: 0, hi: {mul: [2, key_dim]}}}
      v:  {slice: {axis: 0, lo: {mul: [2, key_dim]}, hi: conv_dim}}
    steps:
      - {op: qwen35_v_head_unpermute, input: v, output: v_nat, args: {axis: 0}}
      - {op: concat, inputs: [qk, v_nat], args: {axis: 0}}
```

Rule fields:

| field | meaning |
|---|---|
| `match` | anchored regex; `{dim}` placeholders are substituted before compiling |
| `dest` | dest template; `{group}` (from named groups) and `{dim}` placeholders |
| `drop: true` | drop matching tensors (explicitly, not silently) |
| `optional: true` | rule may match nothing (default: a required rule matching nothing is an error) |
| `quantize: true/false` | emit MLX affine quantization or plain dtype |
| `dtype` | `float32`/`float16` for non-quantized output |
| `expect_shape` | per-axis shape assertion; each axis is a scalar spec or `null` (wildcard) |
| `inputs` | named slots; each reads the matched tensor (`source: "@"`) with an optional `slice` (`axis`, `lo`, `hi` scalar specs) |
| `steps` | operator pipeline (see below) |

### Steps (the transformation IR)

Each step is `{op, input?, inputs?, args?, output?}`:

* `input: name` — consume a slot, a step `output`, or `_` (the previous
  step's output). Omitted on the first step ⇒ `x` (the whole tensor).
* `inputs: [a, b, ...]` — multi-input ops (concat/add/...) consume several
  named values in order.
* `output: name` — bind the result under a name for later steps; otherwise
  the result is addressable as `_` (previous output).
* `args` — plain values; strings that name a dim are resolved to ints.

A step that transforms part of a fused tensor must bind its result via
`output:` and have later steps consume that name (not the raw slot).

## `coverage`

Post-plan assertions that every layer produced its expected tensors.

```yaml
coverage:
  per_layer_required:
    - language_model.model.layers.{layer}.input_layernorm.weight
  per_layer_alternatives:
    - - language_model.model.layers.{layer}.self_attn.q_proj.weight
      - language_model.model.layers.{layer}.linear_attn.in_proj_qkv.weight
```

`{layer}` expands over `range(n_layers)` (the dim named `n_layers`).
`per_layer_alternatives` entries pass if at least one template matches.
Layer count comes from the dim named `n_layers`.

## `output`

```yaml
output:
  model_type: qwen3_5                    # required
  architectures: ["Qwen3_5ForCausalLM"]
  nest_config_under: text_config         # omit for flat configs
  top_level:
    tie_word_embeddings: false
  text_config:                           # scalar specs, merged over the ref config
    hidden_size: [gguf:embedding_length, ref:hidden_size]
  reference_config:
    drop_fields: [architectures, model_name, unsloth_version]
  tokenizer_files: [tokenizer.json, tokenizer_config.json, chat_template.jinja]
  max_shard_bytes: 4294967296
```

Output config generation: if `--source-config` is given, the reference JSON
(minus `drop_fields`, `architectures`, `model_type`) is the `text_config`
base; resolved `text_config` specs are overlaid on top (mismatches warn).
`top_level` values are scalar specs too. Quantization entries
(`bits`/`group_size`/`mode`) are added when quantization is enabled.

## Global option

```yaml
unmatched_tensors: error   # or "warn": what to do with source tensors no rule matched
```

`error` (default) forces configs to account for every tensor — including
explicit `drop` rules for intentionally ignored blocks (e.g. MTP).
