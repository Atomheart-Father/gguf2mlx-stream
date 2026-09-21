"""Fast log-likelihood calibration for quantized model profiles.

Two-level gate, level 1: compare candidate quantized models against a
BF16 reference *without any generation* — immune to think-loop and
EOS anomalies that corrupt generation-based scoring.

Metrics (all computed with the SAME tokenizer, shared by both sides):

* ``nll_ref`` / ``nll_cand`` — teacher-forced next-token NLL over pinned
  corpus windows; ``ce_delta = nll_cand - nll_ref`` (bits/token of
  degradation) and ``kl_ref_to_cand`` (mean KL over next-token
  distributions).
* ``mc_letter_acc`` — for the pinned ARC-Challenge subset, probability
  mass of the *correct letter token* right after the standard prompt.
* ``mc_text_acc`` — argmax over the summed log-likelihood of each
  option *text* continuation (knowledge without instruction-following).

Corpus and subset are pinned files in the oracle cache; sha256 values
are recorded in the report.

Usage::

    python -m research.paired_oracle.calib_ll \
        --reference <bf16_dir> --candidates <dir1> <dir2> ... \
        --corpus <corpus.txt> --arc-subset <arc.jsonl> \
        --report <out.json> [--window 512] [--windows 32]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx

from eval.bench.arc_subset import render_prompt


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(model_dir: str):
    from mlx_lm import load

    model, tokenizer = load(model_dir)
    model.eval()
    return model, tokenizer


def _windows(tokenizer, corpus: str, window: int, count: int) -> list[list[int]]:
    ids = tokenizer.encode(corpus, add_special_tokens=False)
    out = []
    for start in range(0, len(ids) - window, window):
        out.append(ids[start : start + window])
        if len(out) >= count:
            break
    return out


@mx.compile
def _log_softmax(logits: mx.array) -> mx.array:
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def _token_nll_and_kl(ref_model, cand_model, tokens: list[int]) -> tuple[float, float]:
    """Return (nll per token of cand, mean KL(ref||cand)) for one window."""
    x = mx.array([tokens[:-1]])
    y = mx.array(tokens[1:])
    lr = _log_softmax(ref_model(x).astype(mx.float32))
    lc = _log_softmax(cand_model(x).astype(mx.float32))
    nll = -mx.take_along_axis(lc, y[None, :, None], axis=-1).mean()
    kl = (mx.exp(lr) * (lr - lc)).sum(axis=-1).mean()
    return float(nll), float(kl)


def _letter_token_ids(tokenizer, labels: list[str]) -> dict[str, int]:
    """Map option letters to their single leading token id (as ' A', ' B'...)."""
    out = {}
    for label in labels:
        ids = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(ids) >= 1:
            out[label] = ids[0]
    return out


def _continuation_logprob(model, prompt_ids: list[int], cont_ids: list[int]) -> float:
    x = mx.array([prompt_ids + cont_ids])
    logits = model(x).astype(mx.float32)
    logprobs = _log_softmax(logits[0])
    total = 0.0
    n_prompt = len(prompt_ids)
    for i, tok in enumerate(cont_ids):
        total += float(logprobs[n_prompt - 1 + i, tok])
    return total


def arc_metrics(model, tokenizer, subset_path: Path, labels: list[str]) -> dict:
    letter_ids = _letter_token_ids(tokenizer, labels)
    n = 0
    letter_correct = 0
    text_correct = 0
    margins: list[float] = []
    skipped = 0
    with subset_path.open() as f:
        for line in f:
            rec = json.loads(line)
            item_labels = rec["labels"]
            texts = rec["texts"]
            answer = rec["answer_key"]
            if answer not in item_labels or not all(lb in letter_ids for lb in item_labels):
                skipped += 1
                continue
            prompt = render_prompt(rec)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            n += 1
            # letter mode: next-token distribution right after "Answer:"
            x = mx.array([prompt_ids])
            logprobs = _log_softmax(model(x).astype(mx.float32))[0]
            letter_scores = {lb: float(logprobs[-1, letter_ids[lb]]) for lb in item_labels}
            best = max(letter_scores, key=letter_scores.get)
            if best == answer:
                letter_correct += 1
            gold = letter_scores[answer]
            others = [v for k, v in letter_scores.items() if k != answer]
            margins.append(gold - max(others))
            # text mode: summed continuation log-likelihood per option
            text_scores = {
                lb: _continuation_logprob(
                    model, prompt_ids, tokenizer.encode(" " + txt, add_special_tokens=False)
                )
                for lb, txt in zip(item_labels, texts)
            }
            if max(text_scores, key=text_scores.get) == answer:
                text_correct += 1
    return {
        "n_questions": n,
        "n_skipped": skipped,
        "mc_letter_acc": letter_correct / n if n else 0.0,
        "mc_text_acc": text_correct / n if n else 0.0,
        "mean_letter_margin": sum(margins) / len(margins) if margins else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--arc-subset", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--labels", default="A,B,C,D,E")
    args = parser.parse_args(argv)

    corpus_path = Path(args.corpus).expanduser()
    subset_path = Path(args.arc_subset).expanduser()
    corpus = corpus_path.read_text()

    print(f"loading reference {args.reference} ...")
    ref_model, tokenizer = _load(args.reference)
    labels = args.labels.split(",")

    windows = _windows(tokenizer, corpus, args.window, args.windows)
    print(f"corpus: {len(windows)} windows x {args.window} tokens")
    nlls_ref = []
    for w in windows:
        x = mx.array([w[:-1]])
        y = mx.array(w[1:])
        lr = _log_softmax(ref_model(x).astype(mx.float32))
        nlls_ref.append(float(-mx.take_along_axis(lr, y[None, :, None], axis=-1).mean()))
    nll_ref = sum(nlls_ref) / len(nlls_ref)
    print(f"reference NLL: {nll_ref:.4f} nats/token")
    print("reference ARC log-likelihood ...")
    arc_ref = arc_metrics(ref_model, tokenizer, subset_path, labels)
    print(f"reference mc_letter_acc={arc_ref['mc_letter_acc']:.3f} "
          f"mc_text_acc={arc_ref['mc_text_acc']:.3f}")

    results = {
        "reference_dir": args.reference,
        "corpus_sha256": sha256_file(corpus_path),
        "arc_subset_sha256": sha256_file(subset_path),
        "window_tokens": args.window,
        "n_windows": len(windows),
        "reference": {"nll": nll_ref, "arc": arc_ref},
        "candidates": {},
    }
    for cand in args.candidates:
        print(f"loading candidate {cand} ...")
        cand_model, cand_tokenizer = _load(cand)
        if cand_tokenizer.get_vocab() != tokenizer.get_vocab():
            print(f"  WARNING: tokenizer mismatch vs reference for {cand}")
        print("  corpus NLL/KL ...")
        nlls, kls = [], []
        for w in windows:
            nll, kl = _token_nll_and_kl(ref_model, cand_model, w)
            nlls.append(nll)
            kls.append(kl)
        cm = {
            "nll_cand": sum(nlls) / len(nlls),
            "kl_ref_to_cand_mean": sum(kls) / len(kls),
            "ce_delta_vs_ref": sum(nlls) / len(nlls) - nll_ref,
        }
        print("  ARC log-likelihood ...")
        am = arc_metrics(cand_model, tokenizer, subset_path, labels)
        results["candidates"][cand] = {"corpus": cm, "arc": am}
        print(f"  nll={cm['nll_cand']:.4f} ce_delta={cm['ce_delta_vs_ref']:.4f} "
              f"kl={cm['kl_ref_to_cand_mean']:.4f} "
              f"letter={am['mc_letter_acc']:.3f} text={am['mc_text_acc']:.3f}")
        del cand_model

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(results, indent=2) + "\n")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
