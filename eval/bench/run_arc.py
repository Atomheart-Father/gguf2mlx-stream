#!/usr/bin/env python3
"""Run the ARC-Challenge fixed subset against one model side.

Unlike the chat-based capability eval, the ARC benchmark feeds the *same
rendered raw prompt string* to both engines (zero-shot completion, letter
answer) — no chat template on either side. The rendered text is part of the
subset JSONL; its sha256 is recorded in the committed arc_config.json.

Per-question records (JSONL) go to a temp directory and are never committed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arc_subset import extract_letter, verify_saved  # noqa: E402

LLAMA_PERF_RE = {
    "load_s": re.compile(r"load time\s*=\s*([0-9.]+)\s*ms"),
    "eval_s": re.compile(r"eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+) runs"),
}


def is_repetition(text: str) -> bool:
    s = re.sub(r"\s+", " ", text.strip())
    if len(s) < 48:
        return False
    for k in range(8, 49):
        if len(s) >= 2 * k and s[-k:] == s[-2 * k:-k]:
            return True
    for seg in set(re.findall(r"\S{10,}", s)):
        if s.count(seg) >= 5:
            return True
    return False


def run_mlx(model_path: Path, subset_path: Path, out_path: Path,
            max_tokens: int, limit: int | None) -> None:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    items = [json.loads(l) for l in subset_path.read_text().splitlines() if l.strip()]
    model, tokenizer = load(str(model_path))
    sampler = make_sampler(temp=0.0)
    with out_path.open("w") as fh:
        for item in items[:limit]:
            ids = tokenizer.encode(item["prompt"])
            t1 = time.time()
            text = generate(model, tokenizer, prompt=ids,
                            max_tokens=max_tokens, sampler=sampler)
            gen_s = time.time() - t1
            letter = extract_letter(text)
            n_out = len(tokenizer.encode(text))
            fh.write(json.dumps({
                "id": item["id"], "answer_key": item["answer_key"],
                "prompt_sha256": __import__("hashlib").sha256(
                    item["prompt"].encode()).hexdigest()[:16],
                "raw_output": text, "letter": letter,
                "gen_s": round(gen_s, 3), "approx_out_tokens": n_out,
                "truncated": n_out >= max_tokens - 1,
                "repetition": is_repetition(text),
                "extra_text": bool(letter) and len(text.strip()) > len(letter) + 2,
                "max_tokens": max_tokens, "temp": 0.0, "side": "mlx",
            }, ensure_ascii=False) + "\n")
            fh.flush()


def run_llamacpp(gguf: Path, subset_path: Path, out_path: Path,
                 max_tokens: int, ctx: int, bin_name: str,
                 limit: int | None) -> None:
    import hashlib

    items = [json.loads(l) for l in subset_path.read_text().splitlines() if l.strip()]
    with out_path.open("w") as fh:
        for item in items[:limit]:
            cmd = [bin_name, "-m", str(gguf), "--no-display-prompt",
                   "-c", str(ctx), "-n", str(max_tokens),
                   "--temp", "0", "-ngl", "99", "-p", item["prompt"]]
            t1 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  stdin=subprocess.DEVNULL)
            gen_s = time.time() - t1
            runs = None
            for line in proc.stderr.splitlines():
                hit = LLAMA_PERF_RE["eval_s"].search(line)
                if hit:
                    runs = int(hit.group(2))
            raw = proc.stdout.strip()
            letter = extract_letter(raw)
            fh.write(json.dumps({
                "id": item["id"], "answer_key": item["answer_key"],
                "prompt_sha256": hashlib.sha256(
                    item["prompt"].encode()).hexdigest()[:16],
                "raw_output": raw, "letter": letter,
                "gen_s": round(gen_s, 3), "eval_tokens": runs,
                "truncated": runs is not None and runs >= max_tokens - 1,
                "repetition": is_repetition(raw),
                "extra_text": bool(letter) and len(raw) > len(letter) + 2,
                "returncode": proc.returncode,
                "max_tokens": max_tokens, "temp": 0.0, "ctx": ctx,
                "side": "llamacpp",
            }, ensure_ascii=False) + "\n")
            fh.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="side", required=True)

    p = sub.add_parser("mlx")
    p.add_argument("--subset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--expected-sha", default=None,
                   help="verify the subset against the committed hash "
                        "before running")
    p.add_argument("--model", type=Path, required=True)
    p = sub.add_parser("llamacpp")
    p.add_argument("--subset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--expected-sha", default=None,
                   help="verify the subset against the committed hash "
                        "before running")
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--llama-bin", default="llama-completion")
    p.add_argument("--ctx", type=int, default=2048)
    args = ap.parse_args()

    if args.expected_sha:
        verify_saved(args.subset, args.expected_sha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.side == "mlx":
        run_mlx(args.model, args.subset, args.out, args.max_tokens, args.limit)
    else:
        run_llamacpp(args.gguf, args.subset, args.out, args.max_tokens,
                     args.ctx, args.llama_bin, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
