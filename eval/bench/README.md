# ARC-Challenge capability benchmark

A lightweight, metric-bearing gate for converted checkpoints: the same fixed
100-question subset of the **ARC-Challenge validation split** is completed by
the source GGUF (llama.cpp raw completion) and by each converted MLX
candidate (mlx-lm raw completion) under identical prompts.

## Protocol

- **Fixed data**: `allenai/ai2_arc`, config `ARC-Challenge`, split
  `validation`, pinned dataset revision, deterministic 100-question subset
  (sorted by id → seeded shuffle → first 100). The committed
  `arc_config.json` records the revision, seed, selection method, prompt
  template, the sha256 of the canonical subset, and license attribution.
  The dataset TEXT is never committed; every run verifies the local subset
  against the committed hash before scoring.
- **Identical prompts**: both engines receive the *same rendered raw prompt
  string* (zero-shot, "answer with only the letter") — the rendered text is
  part of the subset records, and per-record prompt hashes are logged. No
  chat template is used on either side.
- **Strict extraction**: the answer letter must be a standalone A–E; an
  unclosed `<think>` section counts as no answer.
- **Metrics per side**: accuracy, clean accuracy (correct AND anomaly-free),
  anomaly rate (no-letter / empty / repetition loop / truncation without a
  letter), letter-only compliance, median generation latency. Conversion
  metrics (output size, peak RSS, elapsed time) come from each conversion's
  report.
- **Gate**: a candidate's clean accuracy must not be more than **5 pp below**
  the source's, and its anomaly rate must not exceed the source's by more
  than **2 pp**. Failing candidates are labeled *experimental* — never a
  recommended configuration.

## Attribution

AI2 ARC (Allen Institute for AI), <https://allenai.org/data/arc>, license
**CC-BY-SA-4.0**. Accessed via the Hugging Face dataset `allenai/ai2_arc` at
the revision pinned in `arc_config.json`.

## Reproducing

```bash
# (re)build the subset (network; deterministic given the pinned revision)
python -c "import sys; sys.path.insert(0, 'eval/bench'); \
  from pathlib import Path; from arc_subset import build_and_save; \
  print(build_and_save(Path('/tmp/cap_eval/arc/arc_challenge_val_100.jsonl')))"

# run one side
python eval/bench/run_arc.py llamacpp --gguf SRC.gguf \
  --subset /tmp/cap_eval/arc/arc_challenge_val_100.jsonl \
  --out /tmp/cap_eval/arc/<side>.jsonl --max-tokens 512 --ctx 2048 \
  --expected-sha <sha from arc_config.json>
python eval/bench/run_arc.py mlx --model OUT_DIR --subset ... --out ... \
  --max-tokens 512 --expected-sha <sha>

# score + gate
python eval/bench/score_arc.py --side src=... --side 3bit=... ... \
  --source-side src --out-dir eval/bench/results/<pair>
```

## Published results

- `results/llama32-1b-calibration/` — small-model 3/4/6-bit calibration:
  **3-bit FAILS the gate** (clean accuracy −19 pp vs source; its outputs
  loop into the token cap on 8 items after answering); 4-bit and 6-bit pass.
- `results/qwen36-35b-joyfox-3bit/` — JoyFox 35B source (i1-IQ3_M) vs MLX
  3-bit (`--bits auto`): **FAILS the gate** (clean 55% vs 90%; the deficit
  concentrates in raw-completion protocol fragility — 10 items where the
  thinking model never closed `<think>` within the 512-token budget and 39
  items that answered, then looped into the cap while continuing the quiz).
  Answer-level accuracy when a letter was produced: 83% vs source 90%, and
  89% among items where thinking closed — the conversion itself preserves
  answer quality, but the candidate run is not clean enough to promote.

**Gate conclusion (2026-09-21):** a 3-bit target is **NOT promoted** as a
default recommendation for IQ3-dominant sources. `--bits auto` still
*derives* 3-bit from the histogram (that is the declared mapping), but the
recommendation status stays **experimental** until a protocol under which
the candidate runs cleanly passes the gate.
