# ARC-Challenge 100q: JoyFox Qwen3.6-35B-A3B source (i1-IQ3_M GGUF) vs MLX 3-bit (--bits auto)

Gate: candidate clean accuracy ≥ source − 5 pp AND anomaly rate ≤ source + 2 pp. Failing candidates are experimental results, never recommended configurations.

| side | accuracy | clean accuracy | anomaly rate | letter-only | median gen (s) | size | conv RSS | conv time | verdict |
|---|---|---|---|---|---|---|---|---|---|
| src | 86.0% | 86.0% | 0.0% | 0.0% | 1.676 | — | — | — | source (reference) |
| mlx-3bit | 76.0% | 74.0% | 3.0% | 0.0% | 2.994 | 14.14 | 12.06 | 446.3 | FAIL gate — experimental result only, not a recommended configuration |

Anomaly breakdown per side:

- **src**: none
- **mlx-3bit**: {'repetition_loop': 3}
