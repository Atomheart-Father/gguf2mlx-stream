# Final report: IQ3 → 3-bit conversion — capability vs. usability (2026-09-21)

> **Status note (phase-1 closing):** the "Guard semantics" section below
> describes behavior that was later superseded — the blocking guard and
> `--allow-experimental` were replaced by a non-blocking fidelity warning
> (same-bit auto converts and warns; see README "Validation / Quantization
> Fidelity"). The measured results and gate verdicts are unchanged history.

## Decision

**"3-bit can convert" is proven. "3-bit is usable and should be a default
recommendation" is rejected by the data.**

A 3-bit target for an IQ3-dominant source converts mechanically correctly
(bounded memory, full numeric verification, loadable output, coherent
generation) — but every measured capability gate refuses to bless it:

- plain `--bits auto` for `qwen35moe` + IQ3 sources is now blocked behind
  `--allow-experimental`;
- IQ3 → 3 remains an **explicit experimental option**, not a default
  recommendation, until a calibrated configuration passes the gate;
- 4-bit and 6-bit targets of the same source **pass** the gate and remain
  the sane defaults.

## Conversion mechanics (JoyFox Qwen3.6-35B-A3B, i1-IQ3_M, `--bits auto` → 3)

| metric | 3-bit auto (current, experimental) | 4-bit (superseded history) |
|---|---|---|
| output size | **14.14 GiB (4 shards)** | 18.17 GiB (5 shards) |
| conversion time | 446 s | 428 s |
| peak RSS | **12.06 GiB** | 16.48 GiB |
| verify | ALL OK (733 numeric / 733 shape / 1757 finite) | ALL OK |
| `mlx_lm.load` + chat generation | 8.9 s load; coherent temp-0 output | yes |
| target-bits evidence | `quantization_selection` in config.json (IQ3_S 90.1% → 3) | n/a (pre-auto) |

## Capability gates

### 1. ARC-Challenge 100q (raw completion, identical prompts, hash-verified)

JoyFox 35B, source i1-IQ3_M GGUF vs MLX 3-bit (`results/qwen36-35b-joyfox-3bit/`):

| side | accuracy | clean accuracy | anomaly rate | verdict |
|---|---|---|---|---|
| source | 90.0% | 90.0% | 5.0% | reference |
| MLX 3-bit | 83.0% | 55.0% | 41.0% | **FAIL** |

Answer-level accuracy when a letter was produced: 83% vs 90% (89% among
items where thinking closed). The blocking deficit concentrates in
raw-completion protocol fragility for a thinking model: 10 items never
closed `<think>` inside the 512-token budget; 39 answered, then looped
into the cap while continuing the quiz.

### 2. Small-model 3-bit calibration (Llama-3.2-1B-Instruct, Q4_K_M source)

Same source converted at bits=3 with group sizes 32/64/128
(`results/llama32-1b-3bit-groups/`):

| candidate | clean accuracy | anomaly rate | verdict |
|---|---|---|---|
| source (reference) | 48.0% | 0.0% | — |
| 3-bit g32 | 31.0% | 3.0% | **FAIL** (−17 pp) |
| 3-bit g64 | 29.0% | 8.0% | **FAIL** (−19 pp) |
| 3-bit g128 | 31.0% | 9.0% | **FAIL** (−17 pp) |
| 4-bit g64 (reference) | 47.0% | 1.0% | PASS |
| 6-bit g64 (reference) | 55.0% | 0.0% | PASS |

**No 3-bit group size passes the clean-accuracy/anomaly gate** — finer
groups (g32) reduce the loop anomalies but do not recover the accuracy
deficit. Per the promotion rule, no JoyFox rerun with an alternative group
size is warranted: the gate stays failed for the source class.

### 3. Capability question set, 45 items, strict gate (chat protocol)

The published 35B pair is **superseded history** (its MLX side was the
deleted 4-bit baseline): gated accuracy 67.5% vs 60.0%, MLX anomaly rate
13.3% — basic-question keyword accuracy showed no drop, but this is
explicitly **not** a "no capability loss" result.

## Guard semantics

- `--bits auto` + `qwen3_5_moe` + IQ3-dominant source → conversion refused
  with `PlanError` unless `--allow-experimental` (the experimental status
  is then recorded inside the output `config.json`).
- Explicit `--bits` values are never guarded (the user owns the choice).
- Dry-run prints the guard warning without blocking.
- The guard is data-driven (`EXPERIMENTAL_AUTO_TARGETS` in
  `quant_select.py`) and is removed/updated only when new gate evidence
  flips the verdict.

## What would change the verdict

A configuration that passes the gate on this protocol: clean accuracy
≥ source − 5 pp AND anomaly rate ≤ source + 2 pp on ARC-100, on both the
small-model calibration and the target model. Candidates: other target
bit magnitudes (4/6 pass today), different quantization modes, or a
chat-protocol benchmark for thinking models (raw completion penalizes
open-ended thinking continuation on both sides asymmetrically).
