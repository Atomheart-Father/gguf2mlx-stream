# Integration Matrix

*Generated: 2026-09-21 05:58:50 | host: Apple M4 Pro, 24 GB unified memory | mlx 0.31.2 | mlx-lm 0.31.3 | version: 0.4.1 (build 10964, commit b29c606e2)*

| model | variant | status | src GiB | out GiB | RSS GiB | conv s | verify | load+gen | llamacpp |
|---|---|---|---|---|---|---|---|---|---|
| qwen35_08b:Q4_K_M | Q4_K_M | **PASS** | 0.50 | 0.40 | 4.26 | 90.5 | PASS | PASS [hit: Paris,391] | PASS |
| qwen35_08b:Q6_K | Q6_K | **PASS** | 0.60 | 0.57 | 4.49 | 90.6 | PASS | PASS [hit: Paris,391] | PASS |
| qwen3_06b:Q4_K_M | Q4_K_M | **PASS** | 0.37 | 0.31 | 3.50 | 54.1 | PASS | PASS [hit: Paris] | PASS |
| qwen3_06b:Q6_K | Q6_K | **PASS** | 0.46 | 0.45 | 3.53 | 55.4 | PASS | PASS [hit: Paris] | PASS |
| llama32_1b:Q4_K_M | Q4_K_M | **PASS** | 0.75 | 0.65 | 4.13 | 34.5 | PASS | PASS [hit: Paris,391] | PASS |
| llama32_1b:Q6_K | Q6_K | **PASS** | 0.95 | 0.94 | 4.37 | 34.7 | PASS | PASS [hit: Paris,391] | PASS |
| gemma3_270m:Q4_K_M | Q4_K_M | **PASS** | 0.24 | 0.14 | 3.76 | 55.5 | PASS | PASS [hit: Paris] | PASS |
| gemma3_270m:Q6_K | Q6_K | **PASS** | 0.26 | 0.20 | 3.80 | 55.6 | PASS | PASS [hit: Paris] | PASS |

## oMLX (isolated server)

status: **PASS**
- discovery: PASS (10 models)
- qwen35_08b-q4km-mlx4bit: PASS (hit) 
- qwen3_06b-q4km-mlx4bit: PASS (hit) 
- llama32_1b-q4km-mlx4bit: PASS  
- gemma3_270m-q4km-mlx4bit: PASS  

## Notes

- MLX generation is chat-template based (temp 0, max 160 tokens); `expected_hit` records whether the reference answer appears.
- llama.cpp raw completions are untemplate raw continuations; small wording differences vs MLX are expected after requantization.
- Thinking models may spend tokens on reasoning before answering.
