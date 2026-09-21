# ARC-Challenge 100q 3-bit calibration: Llama-3.2-1B-Instruct Q4_K_M source vs MLX 3-bit (group sizes 32/64/128) + 4/6-bit references

Protocol evidence: 6 side(s) × 100 questions, identical question coverage, answer keys, prompt hashes; recorded temp 0, max_tokens 64.

Gate: candidate clean accuracy ≥ source − 5 pp AND anomaly rate ≤ source + 2 pp. Failing candidates are experimental results, never recommended configurations.

| side | accuracy | clean accuracy | anomaly rate | letter-only | median gen (s) | size | conv RSS | conv time | verdict |
|---|---|---|---|---|---|---|---|---|---|
| src | 48.0% | 48.0% | 0.0% | 0.0% | 0.514 | — | — | — | source (reference) |
| 3bit-g32 | 33.0% | 31.0% | 3.0% | 1.0% | 0.432 | 0.58 | 4.12 | 35.3 | FAIL gate — experimental result only, not a recommended configuration |
| 3bit-g64 | 33.0% | 29.0% | 8.0% | 0.0% | 0.399 | 0.5 | 3.87 | 37.2 | FAIL gate — experimental result only, not a recommended configuration |
| 3bit-g128 | 32.0% | 31.0% | 9.0% | 2.0% | 0.332 | 0.47 | 4.08 | 33.4 | FAIL gate — experimental result only, not a recommended configuration |
| 4bit-g64 | 48.0% | 47.0% | 1.0% | 0.0% | 0.413 | 0.65 | None | None | PASS gate |
| 6bit-g64 | 55.0% | 55.0% | 0.0% | 2.0% | 0.502 | 0.94 | 4.08 | 35.5 | PASS gate |

Anomaly breakdown per side:

- **src**: none
- **3bit-g32**: {'repetition_loop': 3}
- **3bit-g64**: {'repetition_loop': 8}
- **3bit-g128**: {'repetition_loop': 8, 'no_letter': 1, 'truncated': 1}
- **4bit-g64**: {'repetition_loop': 1}
- **6bit-g64**: none
