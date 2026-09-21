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

- **accuracy** per side over the 40 scored items, and the delta;
- **verdict agreement** — fraction of scored items where both sides are
  correct or both wrong (the converter-health signal);
- **answer agreement** — surface-form equality of extracted final answers
  (strict; two checkpoints rarely phrase identically, so it is reported but
  not treated as a quality bar);
- **anomaly rate** — empty output, garbled text (printable ratio /
  replacement chars / control chars), suspect truncation at the cap;
- **timing** — per-question wall and llama.cpp internal eval time.

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

| pair | source accuracy | MLX accuracy | delta | verdict agreement | anomalies (src/mlx) |
|---|---|---|---|---|---|
| JoyFox Qwen3.6-35B-A3B, i1-IQ3_M → MLX 4-bit | 95% (38/40) | 100% (40/40) | +5% | 95% | 2 / 6 |
| Llama-3.2-1B-Instruct, Q4_K_M → MLX 4-bit | 75% (30/40) | 75% (30/40) | ±0 | 95% | 2 / 2 |

Findings:

- **No capability loss from conversion.** The 35B pair's only verdict flips
  (`inst-03`, `inst-04`) are source-side instruction-format failures
  (including a source-side repetition loop); the converted MLX model
  answered them cleanly.
- **Anomalies are temp-0 repetition loops, not corruption.** Zero garbled,
  empty, or immediately-EOS outputs on either side; an MoE routing break
  would show up as nonsense text and does not appear. The MLX 35B loops on
  6/45 items vs 2/45 on the source — greedy decoding is chaotic after
  requantization and a handful of trajectory flips in both directions is
  the expected noise floor.
- The 1B pair flips one item each way (`zh-cs-09`, `logic-01`) — weak-model
  noise, net zero.

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
