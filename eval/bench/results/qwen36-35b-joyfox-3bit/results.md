# ARC-Challenge 100q: JoyFox Qwen3.6-35B-A3B source (i1-IQ3_M GGUF) vs MLX 3-bit (--bits auto)

Gate: candidate clean accuracy ≥ source − 5 pp AND anomaly rate ≤ source + 2 pp. Failing candidates are experimental results, never recommended configurations.

| side | accuracy | clean accuracy | anomaly rate | letter-only | median gen (s) | size | conv RSS | conv time | verdict |
|---|---|---|---|---|---|---|---|---|---|
| src | 90.0% | 90.0% | 5.0% | 5.0% | 2.05 | — | — | — | source (reference) |
| mlx-3bit | 83.0% | 55.0% | 41.0% | 10.0% | 20.521 | 14.14 | 12.06 | 446.3 | FAIL gate — experimental result only, not a recommended configuration |

Anomaly breakdown per side:

- **src**: {'no_letter': 5, 'truncated': 5, 'repetition_loop': 2}
- **mlx-3bit**: {'repetition_loop': 39, 'no_letter': 10, 'truncated': 10}
