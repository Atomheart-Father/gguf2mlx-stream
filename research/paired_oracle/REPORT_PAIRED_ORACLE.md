# Paired-oracle study: GGUF→MLX requantization fidelity and 3-bit profiles

> **Status update (phase-1 closing).** This report is the historical record
> of the study. Its recommendation in §5 was resolved as follows: the
> converter default keeps **same-bit auto** (IQ3/Q3 → MLX 3-bit); the
> blocking guard was replaced by a fidelity warning (see README
> "Validation / Quantization Fidelity"), and the IQ3→4-bit default change
> was **not** applied. `--allow-experimental` no longer exists. §5 below is
> preserved as written at study time.

Branch: `research/paired-oracle-profiles` (from PR #2 HEAD `900045a`; PR #2
remains frozen; converter defaults unchanged — the only converter addition is
the opt-in `--quant-profile` flag with tests).

## Method

Paired-oracle assets pinned by repo revision + sha256 (see
`research/paired_oracle/manifest.py`, manifest built in the oracle cache):

| pair | source | reference |
|---|---|---|
| Qwen3.5-0.8B | `ggml-org/Qwen3.5-0.8B-GGUF` BF16 (rev `8fea6208`) | `mlx-community/Qwen3.5-0.8B-MLX-bf16` (rev `7aef04e9`) |
| Llama-3.2-1B | `unsloth/...-GGUF` BF16 (rev `b69aef11`) | `mlx-community/Llama-3.2-1B-Instruct-bf16` (rev `863c846a`) |
| QAT control | `YoozLabs/Qwen3.5-0.8B-qat-GGUF` Q4_0 (rev `078b637a`) | `YoozLabs/Qwen3.5-0.8B-qat-lean-4bit-mlx` (rev `51d364e8`) |

Local Q3_K_M is produced from the same BF16 GGUF with `llama-quantize`, so
the Q3 source and the baseline share weights by construction. Corpus
(wikitext-2-raw-v1 validation, rev `b08601e0`) and the ARC-100 subset
(canonical sha256 `eb947cff…`, re-verified) are pinned in the cache.

## 1. BF16→BF16 converter-semantics proof (both architectures)

* Qwen3.5-0.8B: 320/320 reference text tensors matched, 0 unmatched ours;
  after f16-format normalization 241 bit-equal and the remaining 79 are all
  norms where the *official export* rounded f32→bf16 (we keep f32 — strictly
  more faithful). `A_log = log(-unpermute(ssm_a))` equal after cast
  (sub-ulp); conv1d/dt_bias/in_proj_*/tied lm_head/MTP drop identical.
  Config: 27 common fields agree; eos_token_id decoded (ours
  `<|im_end|>` = correct for instruct; official's `<|endoftext|>` contradicts
  its own tokenizer_config), rms_norm_eps is a float print artifact. The 153
  reference-only tensors are `vision_tower.*` (VLM export; we are text-only
  by design).
* Llama-3.2-1B: **146/146 matched, 146/146 equal after f16 cast** — a perfect
  match on the second, structurally simpler architecture.

**Verdict: mapping, attention/GDN reordering, norms, MTP handling and config
emission are proven correct at BF16 on two architectures. Every subsequent
difference is attributable to quantization, not to the transcoder.**

## 2. Quantizer equivalence and the cost of double quantization

Per-module relative-L2 error vs the same-family BF16
(`error_attribution.py`, dequantizing per-tensor bits/group_size):

* Our BF16-GGUF→uniform-3-bit reproduces the official MLX 3-bit error
  profile to 4+ significant digits (e.g. `in_proj_b` 0.20442 vs 0.20443) —
  **our quantization path is numerically equivalent to mlx-lm's**.
* Requantizing the locally-produced Q3_K_M GGUF to MLX 3-bit adds
  **+6%…+13.5%** per-module error on top of the 3-bit ceiling. Worst: GDN
  `in_proj_a`/`in_proj_b` (dt/β), `k_proj`, `gate/up` (+13%); least:
  `in_proj_qkv`, `down_proj`, `v_proj` (+6%).
* 4-bit halves the error everywhere (≈0.195 → ≈0.095), as theory predicts.

## 3. Level-1 gate (fast calibration, no generation)

Teacher-forced NLL/KL on pinned wikitext windows + ARC-100 MC
log-likelihood (`calib_ll.py`), same tokenizer, reference = official
instruct bf16 (NLL 3.1375, letter 41.0%):

| candidate (all from Q3_K_M GGUF unless noted) | size | ce Δ (nats) | KL | ARC letter |
|---|---|---|---|---|
| uniform-3 g64 | 341M | +1.007 | 0.956 | 34% |
| mixed_3_4 official-analog | 382M | +0.769 | 0.699 | 26% |
| mixed_3_4 sensitive-informed | 363M | +0.876 | 0.836 | 23% |
| mixed_3_6 official-analog | 465M | +0.687 | 0.615 | 36% |
| mixed_3_6 sensitive-informed | 343M | +1.019 | 0.963 | 36% |
| uniform-4 g64 (official repo, anchor) | 447M | **+0.213** | **0.157** | 36% |
| BF16→3b (no double quantization) | 341M | +1.037 | 0.950 | 27% |

Findings:

1. **The 3-bit affine cliff dominates everything**: ≈+1.0 nat/token
   (≈2.8× perplexity) whether the source is BF16 or Q3_K_M. Double
   quantization is behaviorally secondary (+0.03 nats) even though it adds
   6–13% module error — the information lost by Q3_K_M overlaps what the
   3-bit affine grid loses again.
2. **Every 3-bit-containing profile is Pareto-dominated by plain 4-bit**:
   even mixed_3_6 (which spends MORE bytes than uniform 4-bit) is 3.2×
   worse in KL. Size-heuristic upgrades (big classes) beat
   sensitivity-heuristic upgrades (small worst-classes) because NLL scales
   with parameter mass, but nothing recovers the cliff.
3. Per the two-level gate: no profile advanced to ARC/35B; **no promotion;
   `--bits auto` stays fail-closed for IQ3**.

## 4. QAT control (YoozLabs pair)

The QAT card targets the MLX affine g64 grid; the paired Q4_0 GGUF is a
*different* grid. Decoded by dequant-vs-dequant comparisons:

* Q4_0-f16 vs Yooz-MLX-affine: median rel-L2 0.0996 ≈ the standalone Q4_0
  grid noise baseline (PTQ Q4_0 vs bf16 = 0.0886) → **the pair is the same
  weights in different formats; our tooling recognizes this correctly**.
* Our Q4_0→affine-g64 conversion vs their MLX: 0.116 — the extra ≈0.06 is
  exactly the requantization penalty measured in §2. Self-consistent.
* QAT weights vs original instruct weights differ by 0.0926 (distillation
  shift) — QAT moves weights by a full quantization step; that is precisely
  why grid-aligned QAT works where PTQ cannot.

This control validates both our reading pipeline and the study's central
thesis from the other side: fidelity at low bits comes from aligning
weights to the target grid (QAT), not from smarter PTQ.

## 5. Conclusions and recommendations

1. **IQ3→MLX-uniform-3 has no universal fidelity solution under the current
   MLX affine kernel.** The cliff is the grid, not the source quality and
   not the transcoder. Confirmed on the 0.8B with behavioral metrics that
   generation-based scoring cannot isolate.
2. Recommended auto policy change (needs a converter default change, hence
   user sign-off; NOT applied here): map IQ3 sources to **4-bit** in
   `family_bits`, keep the existing guard text as evidence. Until then
   `--bits auto` for IQ3 remains fail-closed and explicit `--bits` remains
   available.
3. Mixed 3/x profiles are not worth the complexity for this family: they
   are Pareto-inferior to 4-bit at any budget tested. The `--quant-profile`
   mechanism stays (research/replay value, e.g. reproducing official mixed
   builds), but no default uses it.
4. The transcoder itself is exonerated end-to-end: BF16 semantics exact on
   two architectures; quantized output numerically equivalent to mlx-lm's
   on the same weights.

Raw reports (JSON/MD), profiles and the pinned manifest live in the oracle
cache (`GGUF2MLX_ORACLE_CACHE`); dataset text is not committed.
