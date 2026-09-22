#!/usr/bin/env python3
"""Run the capability-regression question set against one model side.

Two sides, identical protocol (same chat template semantics, same turns,
temperature 0, same max tokens, thinking enabled on both sides):

* ``mlx``      — a converted MLX-LM model directory (mlx_lm.load + generate).
  The chat template is the one shipped in the converted output.
* ``llamacpp`` — the source GGUF via Homebrew ``llama-completion`` in
  conversation mode (``--jinja`` uses the GGUF's embedded template — the
  same template the conversion was derived from).

No system prompt is used on either side: llama.cpp 0.4.1 ``-sys`` + ``--jinja``
crashes on this Qwen3.5 template, and fairness requires identical turns on
both sides, so both sides send a single user turn.

Per-question records (JSONL) go to the output directory; nothing here is
committed — the scored report is what lands in the repo.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

LLAMA_PERF_RE = {
    "load_s": re.compile(r"load time\s*=\s*([0-9.]+)\s*ms"),
    "prompt_eval_s": re.compile(r"prompt eval time\s*=\s*([0-9.]+)\s*ms"),
    "eval_s": re.compile(r"eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+) runs"),
}


def load_questions(path: Path):
    data = json.loads(path.read_text())
    return data["questions"], data["max_tokens"]


def clean_llama_stdout(s: str) -> str:
    """Strip conversation-UI artifacts (echoed turns, EOF marker)."""
    if "\nassistant\n" in s:
        s = s.split("\nassistant\n", 1)[1]
    s = re.sub(r"^>?\s*EOF by user\s*$", "", s, flags=re.MULTILINE)
    return s.strip()


def run_mlx(model_path: Path, questions: list[dict], max_tokens: int,
            out_path: Path, limit: int | None) -> None:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    t0 = time.time()
    model, tokenizer = load(str(model_path))
    load_s = time.time() - t0
    print(f"[mlx] loaded {model_path.name} in {load_s:.1f}s", flush=True)

    sampler = make_sampler(temp=0.0)
    with out_path.open("w") as fh:
        for q in questions[:limit]:
            msgs = [{"role": "user", "content": q["prompt"]}]
            ids = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True)
            t1 = time.time()
            text = generate(model, tokenizer, prompt=ids,
                            max_tokens=max_tokens, sampler=sampler)
            gen_s = time.time() - t1
            n_out = len(tokenizer.encode(text))
            rec = {
                "id": q["id"], "prompt": q["prompt"], "raw_output": text,
                "gen_s": round(gen_s, 2), "load_s": None,
                "thinking": "template-default",
                "approx_out_tokens": n_out,
                "truncated_suspect": n_out >= max_tokens - 2,
                "max_tokens": max_tokens, "temp": 0.0,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[mlx] {q['id']}: {gen_s:.1f}s  {n_out} tok", flush=True)


def run_llamacpp(gguf: Path, questions: list[dict], max_tokens: int,
                 out_path: Path, bin_name: str, ctx: int,
                 limit: int | None) -> None:
    with out_path.open("w") as fh:
        for q in questions[:limit]:
            cmd = [
                bin_name, "-m", str(gguf), "--jinja", "--no-display-prompt",
                "-c", str(ctx), "-n", str(max_tokens),
                "--temp", "0", "-ngl", "99",
                "-p", q["prompt"],
            ]
            t1 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  stdin=subprocess.DEVNULL, check=False)
            wall_s = time.time() - t1
            perf = {}
            for m in proc.stderr.splitlines():
                for key, rx in LLAMA_PERF_RE.items():
                    hit = rx.search(m)
                    if hit:
                        perf[key] = hit.group(1)
            runs = None
            for m in proc.stderr.splitlines():
                hit = LLAMA_PERF_RE["eval_s"].search(m)
                if hit:
                    runs = int(hit.group(2))
            raw = clean_llama_stdout(proc.stdout)
            rec = {
                "id": q["id"], "prompt": q["prompt"], "raw_output": raw,
                "gen_s": round(wall_s, 2),
                "load_s": round(float(perf["load_s"]) / 1000.0, 2) if "load_s" in perf else None,
                "prompt_eval_s": round(float(perf["prompt_eval_s"]) / 1000.0, 2) if "prompt_eval_s" in perf else None,
                "eval_s": round(float(perf["eval_s"].split()[0]) / 1000.0, 2) if "eval_s" in perf else None,
                "eval_tokens": runs,
                "returncode": proc.returncode,
                "truncated_suspect": runs is not None and runs >= max_tokens - 2,
                "max_tokens": max_tokens, "temp": 0.0, "ctx": ctx,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[llama] {q['id']}: wall {wall_s:.1f}s"
                  f"  (load {rec['load_s']}s, eval {rec['eval_s']}s / {runs} tok)"
                  f"  {len(raw)} chars", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="side", required=True)

    p_mlx = sub.add_parser("mlx")
    p_mlx.add_argument("--questions", type=Path, required=True)
    p_mlx.add_argument("--out", type=Path, required=True,
                       help="JSONL output path (temp dir, not committed)")
    p_mlx.add_argument("--limit", type=int, default=None,
                       help="only run the first N questions (smoke tests)")
    p_mlx.add_argument("--model", type=Path, required=True,
                       help="converted MLX-LM model directory")
    p_mlx.add_argument("--max-tokens", type=int, default=None)

    p_lama = sub.add_parser("llamacpp")
    p_lama.add_argument("--questions", type=Path, required=True)
    p_lama.add_argument("--out", type=Path, required=True,
                        help="JSONL output path (temp dir, not committed)")
    p_lama.add_argument("--limit", type=int, default=None,
                        help="only run the first N questions (smoke tests)")
    p_lama.add_argument("--gguf", type=Path, required=True)
    p_lama.add_argument("--llama-bin", default="llama-completion")
    p_lama.add_argument("--ctx", type=int, default=4096)
    p_lama.add_argument("--max-tokens", type=int, default=None)
    args = ap.parse_args()

    questions, default_max = load_questions(args.questions)
    max_tokens = args.max_tokens or default_max

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.side == "mlx":
        run_mlx(args.model, questions, max_tokens, args.out, args.limit)
    else:
        run_llamacpp(args.gguf, questions, max_tokens, args.out,
                     args.llama_bin, args.ctx, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
