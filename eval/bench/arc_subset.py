"""ARC-Challenge fixed-subset builder for the capability benchmark.

Builds a deterministic 100-question subset of the ARC-Challenge *validation*
split (zero-shot, multiple choice, letter-only answers) for comparing a
source GGUF against converted MLX candidates.

Data policy (repo hygiene):

* the dataset TEXT is never committed — the subset lives in a temp dir;
* what IS committed is ``arc_config.json``: dataset id, config, split,
  pinned revision, seed, selection method, the prompt template, the sha256
  of the canonical serialized subset (so any re-download is verifiable),
  and license attribution.

Dataset: AllenAI AI2 ARC (allenai/ai2_arc), license CC-BY-SA-4.0.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

DATASET_ID = "allenai/ai2_arc"
CONFIG_NAME = "ARC-Challenge"
SPLIT = "validation"
REVISION = "210d026faf9955653af8916fad021475a3f00453"
SEED = 20260921
N_SUBSET = 100
LICENSE = "CC-BY-SA-4.0"
ATTRIBUTION = (
    "AI2 ARC (Allen Institute for AI), https://allenai.org/data/arc, "
    "license CC-BY-SA-4.0; accessed via the HF dataset allenai/ai2_arc "
    f"at revision {REVISION}"
)

PROMPT_TEMPLATE = (
    "The following is a multiple choice question. "
    "Answer with only the letter of the correct option.\n\n"
    "Question: {question}\n{choices}\nAnswer:"
)

_LABEL_RE = re.compile(r"^\(?([A-E])[).:]?$")


def load_arc_validation() -> list[dict]:
    """Load the pinned validation split (requires network on first call)."""
    from datasets import load_dataset

    ds = load_dataset(
        DATASET_ID, CONFIG_NAME, split=SPLIT, revision=REVISION
    )
    rows = []
    for r in ds:
        labels = list(r["choices"]["label"])
        texts = list(r["choices"]["text"])
        # normalize numeric answer keys (1..5) to letters (A..E)
        key = str(r["answerKey"]).strip()
        if key.isdigit():
            key = "ABCDE"[int(key) - 1]
        key = key.upper()
        if key not in "ABCDE" or not labels or not texts:
            continue
        if len(labels) != len(texts):
            continue
        # normalize choice labels to letters in positional order
        norm_labels = []
        for i, lb in enumerate(labels):
            m = _LABEL_RE.match(str(lb).strip().upper())
            norm_labels.append(m.group(1) if m else "ABCDE"[i])
        rows.append({
            "id": r["id"],
            "question": r["question"],
            "labels": norm_labels,
            "texts": texts,
            "answer_key": key,
        })
    return rows


def select_subset(rows: list[dict], n: int = N_SUBSET, seed: int = SEED) -> list[dict]:
    """Deterministic subset: sort by id, seeded shuffle, take first n."""
    ordered = sorted(rows, key=lambda r: r["id"])
    rng = random.Random(seed)
    indices = list(range(len(ordered)))
    rng.shuffle(indices)
    return [ordered[i] for i in sorted(indices[:n])]


def canonical_subset_bytes(subset: list[dict]) -> bytes:
    """Canonical serialization used for the committed sha256."""
    return json.dumps(
        sorted(subset, key=lambda r: r["id"]),
        ensure_ascii=False, sort_keys=True, indent=1,
    ).encode("utf-8")


def subset_sha256(subset: list[dict]) -> str:
    return hashlib.sha256(canonical_subset_bytes(subset)).hexdigest()


def render_prompt(item: dict) -> str:
    """Rendered zero-shot prompt — the EXACT string both engines receive."""
    lines = [
        f"{lb}. {tx}" for lb, tx in zip(item["labels"], item["texts"])
    ]
    return PROMPT_TEMPLATE.format(
        question=item["question"].strip(), choices="\n".join(lines)
    )


def extract_letter(output: str) -> str | None:
    """Strict letter extraction from a completion.

    Order: leading standalone letter ("C", "C.", "C)", "C:"), an explicit
    "answer/option is B" statement (the letter itself stays uppercase-only —
    a case-insensitive letter class would match prose like "answer is a
    bit"), then the first *standalone* uppercase A-E word.

    An unclosed ``<think>`` section means the model spent its whole budget
    thinking and never emitted a final answer: return None.
    """
    if "</think>" in output:
        output = output.split("</think>", 1)[1]
    elif "<think>" in output:
        return None
    s = output.strip()
    if not s:
        return None
    m = re.match(r"^\(?([A-E])(?=[\s.):]|$)", s)
    if m:
        return m.group(1)
    m = re.search(r"(?:[Aa]nswer|[Oo]ption)\s*(?:is|:)?\s*\(?\b([A-E])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-E])\b", s)
    return m.group(1) if m else None


def build_and_save(out_jsonl: Path) -> dict:
    """Download (pinned), select, save; return the committed config record."""
    rows = load_arc_validation()
    subset = select_subset(rows)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as fh:
        for item in subset:
            fh.write(json.dumps({**item, "prompt": render_prompt(item)},
                                ensure_ascii=False) + "\n")
    return {
        "dataset_id": DATASET_ID,
        "config": CONFIG_NAME,
        "split": SPLIT,
        "revision": REVISION,
        "seed": SEED,
        "n_subset": len(subset),
        "selection": "sorted by id, random.Random(seed).shuffle, first n",
        "prompt_template": PROMPT_TEMPLATE,
        "subset_sha256": subset_sha256(subset),
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "pool_size": len(rows),
    }


def verify_saved(out_jsonl: Path, expected_sha: str) -> None:
    """Re-verify a saved subset file against the committed hash.

    The saved records carry the rendered ``prompt`` for the runners; the
    committed hash covers the dataset items only.
    """
    items = [{k: v for k, v in json.loads(l).items() if k != "prompt"}
             for l in out_jsonl.read_text().splitlines() if l.strip()]
    got = subset_sha256(items)
    if got != expected_sha:
        raise SystemExit(
            f"ARC subset hash mismatch: expected {expected_sha}, got {got}; "
            "the dataset content changed — do NOT score against unverified data"
        )
