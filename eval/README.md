# Capability regression harness

Source-GGUF vs converted-MLX **capability** regression: the same fixed
question set is run through both sides under an identical chat protocol, and
rule-based scoring quantifies whether the converted model lost capability,
corrupted formatting, or broke routing — beyond the expected noise of
requantization.

This directory is evaluation tooling only; it contains no converter code.
Raw generation records (JSONL) stay in a temp directory and are never
committed — the committed artifacts are the question set, the scripts, and
the scored reports under `reports/`.

## Protocol

Both sides of a pair use:

- the **same chat template semantics** — llama.cpp renders the GGUF-embedded
  template; the MLX side uses the tokenizer files the converter derived from
  that same source;
- a single user turn, **no system prompt** (llama.cpp 0.4.1 `-sys` + `--jinja`
  crashes on the Qwen3.5 template, so the no-system form is used on both
  sides for fairness);
- **temperature 0** (greedy) and the **same max_tokens**;
- thinking enabled on both sides (template default) for thinking models —
  llama.cpp 0.4.1 has no working flag to disable it for this template
  (`--reasoning off` and `--reasoning-budget 0` are ignored), so scoring
  extracts the post-`</think>` text on both sides alike.

```
# source side (per question; conversation pre-start + EOF exit)
llama-completion -m SRC.gguf --jinja --no-display-prompt \
    -c 8192 -n MAXTOK --temp 0 -ngl 99 -p "PROMPT"

# MLX side
python eval/run_eval.py mlx --model OUT_DIR --questions eval/questions.json ...
```

## Question set (`questions.json`)

45 items: 10 zh common sense, 10 en common sense, 10 basic math, 5 logic,
5 instruction-following (scored), plus 5 open items screened for anomalies
only (garbling, empty output, immediate EOS, off-topic rambling). Every
scored item carries rule-based checks (normalized substring, number
equality, yes/no, single-token, format constraints) OR-combined; the gold
answer is recorded for readability. Checks run over the post-`<think>`
final text.

## Metrics (per pair)

- **gated accuracy** per side over the scored items, and the delta — the
  scorer reads the *actual* `max_tokens`/`temp` recorded in each run record,
  refuses to score if the two sides' recorded parameters differ, and counts
  any blocking anomaly (truncation at the cap, repetition loop, empty or
  garbled output) as incorrect **even when the expected keyword appears** —
  an answer that loops into the token cap is not an instruction success;
- **strict format constraints** — `number` requires exactly one number in
  the final text, `all_colors` requires exactly the three requested colors
  and no other color word, `one_word` / `yes_no` must match the whole final
  text;
- **anomaly taxonomy** — truncation, repetition loops, empty output, and
  garbled text are reported as separate per-type counts (plus replacement /
  control-char signals), not one lumped number;
- **verdict agreement** — fraction of scored items where both sides are
  correct or both wrong (the converter-health signal);
- **answer agreement** — surface-form equality of extracted final answers
  (strict; two checkpoints rarely phrase identically, so it is reported but
  not treated as a quality bar);
- **timing** — per-question wall and llama.cpp internal eval time.

The scorer is exercised by unit tests under `tests/test_eval_scorer.py`
(run with the repo's normal pytest suite / CI).

## Reproducing

```
python eval/run_eval.py llamacpp --gguf SRC.gguf --questions eval/questions.json \
    --out /tmp/cap_eval/<pair>-src.jsonl --max-tokens 4096 --ctx 8192
python eval/run_eval.py mlx --model OUT_DIR --questions eval/questions.json \
    --out /tmp/cap_eval/<pair>-mlx.jsonl --max-tokens 4096
python eval/score_eval.py --questions eval/questions.json \
    --source ... --mlx ... --out-dir eval/reports/<pair> \
    --title ... --source-desc ... --mlx-desc ...
```

## Published runs

> **SUPERSEDED HISTORY — not a current recommendation.** The 35B run below
> compared against the **4-bit MLX baseline**, which was the wrong target for
> an IQ3-dominant source and has been deleted (moved to Trash on 2026-09-21;
> the converter now auto-selects target bits, resolving this source to
> 3-bit). The reports are kept as scorer-regression and methodology records
> only.

Scored with the **strict gate** (blocking anomalies incorrect; strict format
checks; recorded `max_tokens` asserted identical across sides):

| pair | gated accuracy src | gated accuracy mlx | delta | verdict agreement | anomalies (src/mlx) |
|---|---|---|---|---|---|
| JoyFox Qwen3.6-35B-A3B, i1-IQ3_M → MLX 4-bit [superseded] | 67.5% (27/40) | 60.0% (24/40) | −7.5% | 87.5% | 2 (4.4%) / 6 (13.3%) |
| Llama-3.2-1B-Instruct, Q4_K_M → MLX 4-bit | 42.5% (17/40) | 45.0% (18/40) | +2.5% | 92.5% | 3 (6.7%) / 4 (8.9%) |

Under the earlier lenient scorer (keyword substring only, no anomaly gate)
the same runs scored 95% vs 100% and 75% vs 75%; those figures must not be
quoted as evidence of losslessness.

Findings (strict-gate reading):

- **Basic-question answer accuracy shows no drop, but this is NOT a
  no-capability-loss result.** On the lenient keyword scoring the basic
  (zh/en common-sense, math, logic) items showed no answer-accuracy drop
  after conversion. However the **Q4 MLX 35B anomaly rate is 13.3%**
  (6/45 items loop into the token cap under temp-0 greedy decoding, vs
  4.4% for the source), and under the strict gate the MLX side scores
  *below* the source (60.0% vs 67.5% gated accuracy; per-type anomaly
  breakdown in the report). **不可称无能力损失** — conversion quality must be
  argued from the gated benchmark (see `bench/`), not from keyword-contains
  accuracy.
- **Anomaly types are repetition loops and cap truncation — not corruption.**
  Zero garbled, empty-after-think, or immediately-EOS outputs on either
  side; an MoE routing break would show up as nonsense text and does not
  appear. Greedy decoding is chaotic after requantization and a handful of
  trajectory flips in both directions is the expected noise floor.
- The 1B pair flips a small number of items both ways (weak-model noise);
  its value is protocol calibration, not capability claims.

## Known limitations

- Qwen3.5-0.8B was rejected as the cheap secondary pair: its temp-0
  thinking loops to the token cap on **both** sides, making every item
  truncated and the run uninformative. Llama-3.2-1B (no thinking) is used
  instead, which additionally exercises the llama config path.
- The 35B MLX run executed under heavy memory pressure (~2% free, 18.6 GiB
  model on 24 GB unified); its timing figures are inflated and noisy.
  Correctness is unaffected.
- `answer_agreement_rate` is surface-form and low by construction; use
  verdict agreement for converter health.
