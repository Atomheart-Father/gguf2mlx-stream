# JoyFox 35B conversion evidence (3-bit auto)

- command: `convert JoyFox...i1-IQ3_M.gguf --output /private/tmp/moe_convert/Qwen3.6-35B-A3B-3bit-auto --tokenizer-source <3bit-reference>` (no --bits)
- dry-run resolution: `[bits] target = 3 (auto)`; dominant family IQ3_S = 90.1% of quantized source bytes
- output: 14.14 GiB, 4 shards, 733 tensors (4-bit baseline for the same source was 18.17 GiB / 5 shards)
- elapsed 446 s, peak RSS 12.06 GiB (24 GB unified machine)
- verify: ALL OK (733 numeric, 733 shape, 1757 finite checks)
- mlx_lm.load: 8.9 s; chat generation coherent (temp 0, 80 tokens, post-think text clean)
- note: 'source IQ3' and 'MLX affine 3-bit' are the same target bit magnitude, not bit-for-bit equivalent encodings
