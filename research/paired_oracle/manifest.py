"""Build and verify a pinned asset manifest for the paired-oracle study.

Every downloaded asset is recorded with its HF repo id, pinned revision,
resolved file names, byte size and local sha256 so that any future run can
re-derive the exact bytes. Dataset *text* is never committed; only hashes.

Usage::

    python -m research.paired_oracle.manifest build --cache <dir>
    python -m research.paired_oracle.manifest verify --cache <dir> [--strict]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_NAME = "manifest.json"
REVISIONS_NAME = "repo_revisions.json"

# Maps a cache subdirectory to the (repo, revision) it was downloaded from.
# Subdirectories are flat: <cache>/<kind>/<slug>/...
KNOWN_ASSETS: dict[str, dict[str, str]] = {
    "gguf/ggml-org-qwen35-0.8b-bf16": {
        "repo": "ggml-org/Qwen3.5-0.8B-GGUF",
        "revision": "8fea620810c4afa23dd6443f999a48574c1611a3",
        "kind": "gguf",
    },
    "gguf/ggml-org-qwen35-0.8b-q3_k_m": {
        "repo": "local:llama-quantize",  # produced locally, see q3_local record
        "revision": "derived:8fea620810c4afa23dd6443f999a48574c1611a3",
        "kind": "gguf",
    },
    "gguf/yoozlabs-qwen35-0.8b-qat-q4_0": {
        "repo": "YoozLabs/Qwen3.5-0.8B-qat-GGUF",
        "revision": "078b637ae52dae5773e3159de9c1d691ed771837",
        "kind": "gguf",
    },
    "gguf/ggml-org-qwen35-0.8b-q4_0": {
        "repo": "ggml-org/Qwen3.5-0.8B-GGUF",
        "revision": "8fea620810c4afa23dd6443f999a48574c1611a3",
        "kind": "gguf",
    },
    "gguf/unsloth-llama32-1b-bf16": {
        "repo": "unsloth/Llama-3.2-1B-Instruct-GGUF",
        "revision": "b69aef112e9f895e6f98d7ae0949f72ff09aa401",
        "kind": "gguf",
    },
    "mlx/llama32-1b-instruct-bf16": {
        "repo": "mlx-community/Llama-3.2-1B-Instruct-bf16",
        "revision": "863c846a9ac6fad4e49e1743d52984dff262e953",
        "kind": "mlx",
    },
    "mlx/llama32-1b-instruct-4bit": {
        "repo": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "revision": "08231374eeacb049a0eade7922910865b8fce912",
        "kind": "mlx",
    },
    "mlx/qwen35-0.8b-instruct-bf16": {
        "repo": "mlx-community/Qwen3.5-0.8B-MLX-bf16",
        "revision": "7aef04e9adfd926ce0da9da376fe9610c8818a58",
        "kind": "mlx",
    },
    "mlx/qwen35-0.8b-instruct-4bit": {
        "repo": "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        "revision": "5d894f8cc4ef3e6c88537bf3746ed262f549da6a",
        "kind": "mlx",
    },
    "mlx/qwen35-0.8b-base-bf16": {
        "repo": "mlx-community/Qwen3.5-0.8B-bf16",
        "revision": "3067585164dbcc505eec73d554349de6a27571a4",
        "kind": "mlx",
    },
    "mlx/qwen35-0.8b-base-3bit": {
        "repo": "mlx-community/Qwen3.5-0.8B-3bit",
        "revision": "63755e255fea6273fd31ec1e2a27accb81d95c2d",
        "kind": "mlx",
    },
    "mlx/qwen35-0.8b-base-4bit": {
        "repo": "mlx-community/Qwen3.5-0.8B-4bit",
        "revision": "da28692b5f139cb0ec58a356b437486b7dac7462",
        "kind": "mlx",
    },
    "mlx/qwen35-0.8b-base-6bit": {
        "repo": "mlx-community/Qwen3.5-0.8B-6bit",
        "revision": "779b383518183ae3af13b26a5cdc829a26f7c937",
        "kind": "mlx",
    },
    "mlx/qwen35-0.8b-base-mixed_3_4": {
        "repo": "mlx-community/Qwen3.5-0.8B-mixed_3_4",
        "revision": "d913dfcbe130739e734397cd54c70c2f7777abb0",
        "kind": "mlx",
    },
    "mlx/qwen35-0.8b-base-mixed_3_6": {
        "repo": "mlx-community/Qwen3.5-0.8B-mixed_3_6",
        "revision": "974a35e3eaaa7c3c5dd74991662092e5f742cda4",
        "kind": "mlx",
    },
    "mlx/qwen35-0.8b-yooz-qat-4bit": {
        "repo": "YoozLabs/Qwen3.5-0.8B-qat-lean-4bit-mlx",
        "revision": "51d364e8e3a704d4926074833621d06d2527008c",
        "kind": "mlx",
    },
}

_HASH_CHUNK = 1 << 20


@dataclass
class FileRecord:
    path: str
    size: int
    sha256: str


@dataclass
class AssetRecord:
    key: str
    repo: str
    revision: str
    kind: str
    files: list[FileRecord] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _asset_files(asset_dir: Path) -> list[FileRecord]:
    records: list[FileRecord] = []
    for path in sorted(asset_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            records.append(
                FileRecord(
                    path=str(path.relative_to(asset_dir)),
                    size=path.stat().st_size,
                    sha256=sha256_file(path),
                )
            )
    return records


def build(cache: Path) -> dict:
    revisions: dict[str, str] = {}
    rev_path = cache / REVISIONS_NAME
    if rev_path.exists():
        revisions = {
            repo: rec["sha"]
            for repo, rec in json.loads(rev_path.read_text()).items()
        }

    assets: list[AssetRecord] = []
    for key, meta in KNOWN_ASSETS.items():
        asset_dir = cache / key
        if not asset_dir.is_dir():
            continue
        revision = revisions.get(meta["repo"], meta["revision"])
        assets.append(
            AssetRecord(
                key=key,
                repo=meta["repo"],
                revision=revision,
                kind=meta["kind"],
                files=_asset_files(asset_dir),
            )
        )

    manifest = {
        "manifest_version": 1,
        "assets": [asdict(a) for a in assets],
        "total_bytes": sum(a.total_bytes for a in assets),
    }
    out = cache / MANIFEST_NAME
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify(cache: Path, strict: bool) -> int:
    manifest_path = cache / MANIFEST_NAME
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path}; run 'build' first")
        return 2
    manifest = json.loads(manifest_path.read_text())
    failures = 0
    for asset in manifest["assets"]:
        asset_dir = cache / asset["key"]
        if not asset_dir.is_dir():
            print(f"MISSING dir {asset['key']}")
            failures += 1
            continue
        for rec in asset["files"]:
            f = asset_dir / rec["path"]
            if not f.is_file():
                print(f"MISSING file {asset['key']}/{rec['path']}")
                failures += 1
                continue
            if f.stat().st_size != rec["size"]:
                print(f"SIZE mismatch {asset['key']}/{rec['path']}")
                failures += 1
                continue
            if sha256_file(f) != rec["sha256"]:
                print(f"SHA256 mismatch {asset['key']}/{rec['path']}")
                failures += 1
    if failures == 0:
        print(f"OK: {len(manifest['assets'])} assets verified, "
              f"{manifest['total_bytes'] / 1e9:.2f} GB")
    elif strict:
        print(f"FAILED: {failures} mismatches (strict)")
        return 1
    else:
        print(f"WARNING: {failures} mismatches")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--cache", default=os.environ.get("GGUF2MLX_ORACLE_CACHE", ""))
        p.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    if not args.cache:
        parser.error("--cache or GGUF2MLX_ORACLE_CACHE is required")
    cache = Path(args.cache).expanduser().resolve()
    if args.command == "build":
        manifest = build(cache)
        print(f"manifest written: {cache / MANIFEST_NAME} "
              f"({len(manifest['assets'])} assets, "
              f"{manifest['total_bytes'] / 1e9:.2f} GB)")
        return 0
    return verify(cache, args.strict)


if __name__ == "__main__":
    sys.exit(main())
