#!/usr/bin/env python
"""Fetch pinned integration-model GGUFs and tokenizer files.

Reads ``tests/integration/models.yaml``, downloads ONLY the listed files
into ``$GGUF2MLX_INTEGRATION_DIR`` (default: ``<repo>/.integration-models``,
git-ignored), skips anything already present, and records
repo/revision/filename/sha256/size in ``download-log.json`` inside the
integration dir. Never touches anything outside that directory.

Usage:
    python scripts/fetch_integration_models.py [--manifest tests/integration/models.yaml]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import yaml
from huggingface_hub import hf_hub_download

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_of(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default=os.path.join(REPO_ROOT, "tests", "integration", "models.yaml"),
    )
    args = ap.parse_args()

    root = os.environ.get(
        "GGUF2MLX_INTEGRATION_DIR", os.path.join(REPO_ROOT, ".integration-models")
    )
    gguf_dir = os.path.join(root, "gguf")
    tok_dir = os.path.join(root, "tokenizer")
    os.makedirs(gguf_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)

    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)

    log_path = os.path.join(root, "download-log.json")
    log: dict = {}
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)

    tok_files = manifest.get("tokenizer_files", [])

    for model, entry in sorted(manifest.items()):
        if model == "tokenizer_files":
            continue
        # tokenizer files (small; fetched once per model)
        tdir = os.path.join(tok_dir, model)
        got = []
        for fn in tok_files:
            try:
                hf_hub_download(
                    repo_id=entry["tokenizer_repo"],
                    filename=fn,
                    revision=entry["tokenizer_revision"],
                    local_dir=tdir,
                )
                got.append(fn)
            except Exception as exc:
                print(f"[tok] {model}/{fn}: SKIP ({type(exc).__name__}: {str(exc)[:80]})")
        log.setdefault(model, {})["tokenizer"] = {
            "repo": entry["tokenizer_repo"],
            "revision": entry["tokenizer_revision"],
            "files": got,
            "dir": tdir,
        }
        print(f"[tok] {model}: {len(got)} file(s) ready")

        for quant, var in sorted(entry["variants"].items()):
            dest = os.path.join(gguf_dir, var["file"])
            key = f"{model}:{quant}"
            if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                print(f"[gguf] {key}: already present, skipping download")
            else:
                print(f"[gguf] {key}: downloading {var['gguf_repo']}@{var['file']} ...")
                hf_hub_download(
                    repo_id=var["gguf_repo"],
                    filename=var["file"],
                    revision=var["revision"],
                    local_dir=gguf_dir,
                )
            digest = sha256_of(dest)
            log.setdefault(model, {})[quant] = {
                "repo": var["gguf_repo"],
                "revision": var["revision"],
                "file": var["file"],
                "sha256": digest,
                "size_bytes": os.path.getsize(dest),
                "path": dest,
                "mlx_bits": var.get("mlx_bits"),
            }
            print(f"[gguf] {key}: sha256={digest[:16]}... size={os.path.getsize(dest)/2**20:.1f} MiB")

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[done] log written to {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
