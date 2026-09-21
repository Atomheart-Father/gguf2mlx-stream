# ARC-Challenge 100q calibration: Llama-3.2-1B-Instruct source (Q4_K_M GGUF) vs MLX 3/4/6-bit candidates

Gate: candidate clean accuracy ≥ source − 5 pp AND anomaly rate ≤ source + 2 pp. Failing candidates are experimental results, never recommended configurations.

| side | accuracy | clean accuracy | anomaly rate | letter-only | median gen (s) | size | conv RSS | conv time | verdict |
|---|---|---|---|---|---|---|---|---|---|
| src | 48.0% | 48.0% | 0.0% | 0.0% | 0.514 | — | — | — | source (reference) |
| 3bit | 33.0% | 29.0% | 8.0% | 0.0% | 0.399 | 0.5 | 3.87 | 37.2 | FAIL gate — experimental result only, not a recommended configuration |
| 4bit | 48.0% | 47.0% | 1.0% | 0.0% | 0.413 | 0.65 | None | None | PASS gate |
| 6bit | 55.0% | 55.0% | 0.0% | 2.0% | 0.502 | 0.94 | 4.08 | 35.5 | PASS gate |

Anomaly breakdown per side:

- **src**: none
- **3bit**: {'repetition_loop': 8}
- **4bit**: {'repetition_loop': 1}
- **6bit**: none
