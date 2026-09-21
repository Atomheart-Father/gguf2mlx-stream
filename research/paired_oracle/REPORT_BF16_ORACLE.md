# Paired-oracle proof: BF16 GGUF -> unquantized MLX (Qwen3.5-0.8B)

Branch: `research/paired-oracle-profiles`. Research-only; converter defaults
unchanged (the only converter addition is the opt-in `--quant-profile` flag).

## Question

Does our converter reproduce the *semantics* of the official MLX export of the
same base model, before any quantization noise? If mapping, q/k/v and GDN
reordering, norms, conv1d, A_log, MTP handling and config emission are wrong,
they must show up here — quantized comparisons can mask structural errors.

## Paired assets (pinned, see manifest.json)

| side | asset | revision |
|---|---|---|
| source | `ggml-org/Qwen3.5-0.8B-GGUF` `Qwen3.5-0.8B-BF16.gguf` | `8fea620810c4afa23dd6443f999a48574c1611a3` |
| reference | `mlx-community/Qwen3.5-0.8B-MLX-bf16` | `7aef04e9adfd926ce0da9da376fe9610c8818a58` |

Both derive from `Qwen/Qwen3.5-0.8B` (instruct). Source: 335 tensors
(BF16 x195, F32 x140), 24 layers + `blk.24.nextn` MTP block, tied embeddings,
no vision tensors. Conversion: `--no-quantize`, 89 s, peak RSS 6.32 GiB.

## Tensor verdict (compare_bf16.py)

- **320/320 reference text tensors matched; 0 unmatched on our side.**
  All shapes equal; zero NaN/Inf.
- The reference has 153 additional tensors — all `vision_tower.*` /
  merger (the official export is a VLM; ours is text-only by design).
- After normalizing both sides to f16: **241/320 bit-equal**. The 79
  remaining are *all* norm-family tensors (`input_layernorm`,
  `post_attention_layernorm`, `linear_attn.norm`, `output_norm`): the
  official export rounds its (f32-sourced) norms to bf16; we keep f32
  from the GGUF, which is strictly more faithful. No matrix weight
  disagrees after format normalization.
- `A_log` (our `log(-unpermute(ssm_a))` pipeline): equal after both bf16
  and f16 casts; raw max abs diff 5.96e-08 (sub-ulp of the log arithmetic).
  The transform is exactly right.
- `conv1d` (dim,k)->(dim,k,1), `dt_bias`, all `in_proj_*` unpermutations,
  tied-embedding `lm_head` absence: identical values.
- MTP: both sides exclude the NextN block (equal tensor sets).
- Storage note: our `--no-quantize` contract is float16; the reference is
  bfloat16. Raw-format diffs are therefore dominated by bf16-vs-f16
  rounding (embeddings in this model are tiny; 15,948 of ~254M embedding
  values — 0.006%, all <= 6e-8 — flush to zero in f16, which bf16 could
  represent). After cast-normalization these vanish everywhere except the
  official-side norm rounding above.

## Config verdict

27 common fields agree. Differences:

1. `architectures`: ours `Qwen3_5ForCausalLM` (text) vs official
   `Qwen3_5ForConditionalGeneration` (VLM) — correct for a text output.
2. `text_config.eos_token_id`: ours 248046 (`<|im_end|>`) vs official
   248044 (`<|endoftext|>`). Both sides' `tokenizer_config.json` say
   `<|im_end|>`; ours is consistent with the instruct chat semantics and
   with the verified JoyFox 35B outputs. Not a defect.
3. `text_config.rms_norm_eps`: 9.999999974752427e-07 (f32 read of GGUF)
   vs 1e-06 — same value, float printing artifact.

Fields only the reference has are VLM/optional-default fields
(vision, `layer_types`, mtp/rope extras); fields only we have are
`bos_token_id`/`pad_token_id` (harmless, explicit).

## Conclusion

**BF16 GGUF -> BF16-semantics MLX conversion is proven correct** for the
Qwen3.5-0.8B architecture: tensor semantics are identical modulo storage
format, and the emitted config is runtime-equivalent (one deliberate,
documented eos choice). Quantization studies (3-bit profiles, mixed
profiles) can now attribute every difference to quantization, not to the
transcoder.

Raw artifact: `reports/bf16_oracle_report.json` (+ `.md`) in the oracle
cache; tensor text is not committed.
