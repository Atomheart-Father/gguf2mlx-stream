"""Pinned paired-oracle research tooling.

This package is research-only. It never changes converter defaults; the
only converter-facing addition is the opt-in ``--quant-profile`` flag.

Directory conventions (override the cache root with ``--cache`` /
``GGUF2MLX_ORACLE_CACHE``):

    <cache>/gguf/...      source GGUF files, one directory per repo
    <cache>/mlx/...       reference MLX model directories
    <cache>/out/...       our converter outputs (expendable)
    <cache>/reports/...   generated reports
    <cache>/manifest.json pinned asset manifest (revisions + sha256)
"""
