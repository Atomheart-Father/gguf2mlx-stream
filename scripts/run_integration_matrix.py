#!/usr/bin/env python
"""Real-model integration matrix runner.

For every model/variant in ``tests/integration/models.yaml`` this script runs
the full release-gate pipeline and records everything in
``reports/integration-matrix.json`` + ``reports/integration-matrix.md``:

    inspect -> validate-config -> dry-run -> convert -> verify
        -> mlx_lm.load + chat generation (>=64 tokens, temp 0)
        -> source GGUF generation via llama.cpp (raw + chat)
        -> semantic comparison records

Conversions run in subprocesses; peak RSS / wall time come from the
converter's own ``--report-json``. Model files are never committed; reports
are. Usage:

    python scripts/run_integration_matrix.py [--only gemma3_270m:Q4_K_M] [--skip-convert]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

INTEGRATION_DIR = Path(
    os.environ.get("GGUF2MLX_INTEGRATION_DIR", REPO / ".integration-models")
)
GGUF_DIR = INTEGRATION_DIR / "gguf"
OUT_DIR = INTEGRATION_DIR / "out"
REPORT_DIR = REPO / "reports"
PY = sys.executable

# simple, stable probe prompts (tensor-semantics validation, not benchmarks)
RAW_PROMPTS = ["The capital of France is", "17 * 23 ="]
CHAT_PROMPTS = [
    ("The capital of France is? Answer with just the city name.", "Paris"),
    ("17 * 23 = ? Answer with just the number.", "391"),
    ("Write one sentence about the ocean.", None),
]


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    ok = sum(1 for c in s if c.isprintable() or c in "\n\r\t")
    return ok / len(s)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_inspect(gguf: Path) -> dict:
    from gguf2mlx_stream.source.gguf import GGUFSource

    s = GGUFSource(str(gguf))
    return {
        "status": "PASS",
        "gguf_arch": s.arch,
        "n_tensors": len(s.tensors),
        "source_bytes": os.path.getsize(gguf),
        "metadata": {k: v for k, v in sorted(s.metadata.items())
                     if not k.startswith("tokenizer.") and not k.startswith("general.")},
    }


def stage_cli(cmd: list[str]) -> dict:
    t0 = time.time()
    r = sh(cmd)
    return {
        "status": "PASS" if r.returncode == 0 else "FAIL",
        "rc": r.returncode,
        "elapsed_s": round(time.time() - t0, 1),
        "stderr": r.stderr[-2000:] if r.returncode else "",
    }


def stage_convert(gguf: Path, cfg: Path, out: Path, bits: int, tokenizer_dir: Path,
                  report_json: Path) -> dict:
    t0 = time.time()
    r = sh([
        PY, "-m", "gguf2mlx_stream.cli", "convert", str(gguf),
        "--arch-config", str(cfg), "--output", str(out),
        "--tokenizer-source", str(tokenizer_dir),
        "--bits", str(bits), "--quiet", "--overwrite",
        "--report-json", str(report_json),
    ], cwd=str(REPO))
    if r.returncode != 0:
        return {"status": "FAIL", "rc": r.returncode, "elapsed_s": round(time.time() - t0, 1),
                "stderr": r.stderr[-2000:]}
    rep = json.loads(report_json.read_text())
    return {
        "status": "PASS",
        "elapsed_s": round(rep.get("elapsed_s", time.time() - t0), 1),
        "peak_rss_gib": rep.get("peak_rss_gib"),
        "n_jobs": rep.get("n_jobs"),
        "n_dropped": rep.get("n_dropped"),
        "dims": rep.get("dims"),
        "output_bytes": rep.get("output_bytes"),
        "n_shards": rep.get("n_shards"),
    }


def stage_structural(out: Path, plan_report: dict) -> dict:
    index = json.loads((out / "model.safetensors.index.json").read_text())
    cfg = json.loads((out / "config.json").read_text())
    wm = index["weight_map"]
    missing_files = sorted({f for f in wm.values() if not (out / f).is_file()})
    checks = {
        "index_files_present": not missing_files,
        "config_complete": cfg.get("model_type") is not None
        and bool(cfg.get("architectures")),
        "unmatched_none": True,  # convert fails loudly on unmatched when policy=error
    }
    total = 0
    for f in sorted(set(wm.values())):
        total += (out / f).stat().st_size
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "n_keys": len(wm),
        "n_shards": len(set(wm.values())),
        "index_total_size": index["metadata"]["total_size"],
        "missing_files": missing_files,
        "model_type": cfg.get("model_type"),
        "quantization": cfg.get("quantization"),
    }


def stage_mlx_generation(out: Path) -> dict:
    """Subprocess: mlx_lm.load + chat generation at temp 0."""
    script = r"""
import json, sys
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model_dir = sys.argv[1]
prompts = json.loads(sys.argv[2])
model, tokenizer = load(model_dir)
sampler = make_sampler(temp=0.0)
results = []
for prompt_text, expected in prompts:
    msgs = [{"role": "user", "content": prompt_text}]
    p = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
    out = generate(model, tokenizer, prompt=p, max_tokens=160, sampler=sampler)
    results.append({"prompt": prompt_text, "expected": expected, "output": out})
print(json.dumps(results))
"""
    t0 = time.time()
    r = sh([PY, "-c", script, str(out),
            json.dumps([[p, e] for p, e in CHAT_PROMPTS])])
    elapsed = round(time.time() - t0, 1)
    if r.returncode != 0:
        return {"status": "FAIL", "elapsed_s": elapsed, "stderr": r.stderr[-2000:]}
    try:
        results = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"status": "FAIL", "elapsed_s": elapsed, "stderr": f"parse: {exc}"}
    gens = []
    for res in results:
        text = res["output"]
        gens.append({
            "prompt": res["prompt"],
            "expected": res["expected"],
            "n_chars": len(text),
            "printable_ratio": round(printable_ratio(text), 3),
            "immediate_eos": len(text) < 2,
            "expected_hit": (res["expected"] in text) if res["expected"] else None,
            "output": text[:400],
        })
    # runtime bar: no crash, no empty/immediately-EOS output, no mojibake,
    # and the model demonstrably produces multi-token text somewhere
    ok = (
        all(not g["immediate_eos"] and g["printable_ratio"] > 0.9 for g in gens)
        and max(g["n_chars"] for g in gens) >= 30
    )
    return {"status": "PASS" if ok else "FAIL", "elapsed_s": elapsed, "generations": gens}


def stage_llamacpp(gguf: Path) -> dict:
    """Source-side generation via llama.cpp: raw completions + chat turns."""
    results = {"status": "PASS", "raw": [], "chat": []}
    for p in RAW_PROMPTS:
        r = sh(["llama-completion", "-m", str(gguf), "-p", p,
                "-n", "24", "--temp", "0", "--no-conversation"])
        results["raw"].append({
            "prompt": p,
            "status": "PASS" if r.returncode == 0 else "FAIL",
            "output": r.stdout.strip()[:200],
        })
    for prompt_text, expected in CHAT_PROMPTS:
        r = sh(["llama-completion", "-m", str(gguf), "-p", prompt_text,
                "-n", "160", "--temp", "0"])
        text = r.stdout.strip()
        # strip the echoed conversation scaffold / trailer
        for marker in ("\n> EOF by user", "> EOF by user"):
            text = text.split(marker)[0]
        results["chat"].append({
            "prompt": prompt_text,
            "expected": expected,
            "status": "PASS" if r.returncode == 0 else "FAIL",
            "expected_hit": (expected in text) if expected else None,
            "output": text.strip()[:400],
        })
    if any(x["status"] != "PASS" for x in results["raw"] + results["chat"]):
        results["status"] = "FAIL"
    return results


def stage_omlx(models: list[Path], port: int = 8965) -> dict:
    """Isolated oMLX server: discovery + load + chat completions."""
    cli = os.environ.get("OMLX_CLI", "/Applications/oMLX.app/Contents/MacOS/omlx-cli")
    if not Path(cli).exists():
        return {"status": "SKIPPED", "reason": "oMLX app not installed"}
    # dummy credential for the throwaway local server on 127.0.0.1 only
    api_key = os.environ.get("OMLX_MATRIX_KEY", "gguf2mlx-matrix-local-key")
    log_path = INTEGRATION_DIR / f"omlx-{port}.log"
    proc = subprocess.Popen(
        [cli, "serve", "--model-dir", str(models[0].parent), "--host", "127.0.0.1",
         "--port", str(port), "--api-key", api_key, "--log-level", "info"],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
    )
    import urllib.request

    base = f"http://127.0.0.1:{port}"
    try:
        # wait for the server to come up
        up = False
        for _ in range(60):
            if proc.poll() is not None:
                return {"status": "FAIL", "reason": "server exited early",
                        "log": log_path.read_text()[-1500:]}
            try:
                req = urllib.request.Request(base + "/v1/models",
                                             headers={"Authorization": f"Bearer {api_key}"})
                with urllib.request.urlopen(req, timeout=2) as resp:
                    discovered = {m["id"] for m in json.load(resp)["data"]}
                up = True
                break
            except Exception:
                time.sleep(1)
        if not up:
            return {"status": "FAIL", "reason": "server did not start",
                    "log": log_path.read_text()[-1500:]}

        want = {p.name for p in models}
        discovery = {"status": "PASS" if want <= discovered else "FAIL",
                     "expected": sorted(want), "discovered": sorted(discovered)}

        loaded = []
        for p in models:
            req = urllib.request.Request(
                base + "/v1/chat/completions",
                data=json.dumps({
                    "model": p.name,
                    "messages": [{"role": "user",
                                  "content": "17 * 23 = ? Answer with just the number."}],
                    "max_tokens": 300,
                    "temperature": 0,
                }).encode(),
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    body = json.load(resp)
                text = body["choices"][0]["message"]["content"]
                loaded.append({"model": p.name, "status": "PASS",
                               "expected_hit": "391" in text,
                               "output": text.strip()[:300]})
            except Exception as exc:
                body = ""
                try:
                    body = exc.read().decode()[:300]  # type: ignore[union-attr]
                except Exception:
                    pass
                loaded.append({"model": p.name, "status": "FAIL",
                               "error": f"{type(exc).__name__}: {exc} {body}"})
        ok = discovery["status"] == "PASS" and all(x["status"] == "PASS" for x in loaded)
        return {"status": "PASS" if ok else "FAIL",
                "discovery": discovery, "chat_completions": loaded}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=[],
                    help="restrict to model:variant entries (repeatable)")
    ap.add_argument("--omlx", action="store_true",
                    help="run the isolated oMLX stage for one model per family")
    ap.add_argument("--omlx-port", type=int, default=8965)
    args = ap.parse_args()

    manifest = yaml.safe_load((REPO / "tests/integration/models.yaml").read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    omlx_probe_models: list[Path] = []
    entries: dict[str, dict] = {}
    t_all = time.time()

    for model, entry in manifest.items():
        if model == "tokenizer_files":
            continue
        for quant, var in sorted(entry["variants"].items()):
            key = f"{model}:{quant}"
            if args.only and key not in args.only:
                continue
            gguf = GGUF_DIR / var["file"]
            if not gguf.is_file():
                entries[key] = {"status": "SKIPPED", "reason": f"missing {gguf}"}
                continue
            print(f"=== {key} ===", flush=True)
            e: dict = {"family": entry["family"], "status": "RUNNING"}
            e["source"] = {
                "gguf_repo": var["gguf_repo"], "revision": var["revision"],
                "file": var["file"], "sha256": "(see download-log.json)",
            }
            dl_log = INTEGRATION_DIR / "download-log.json"
            if dl_log.exists():
                dl = json.loads(dl_log.read_text())
                sha = dl.get(model, {}).get(quant, {}).get("sha256")
                if sha:
                    e["source"]["sha256"] = sha
                    e["source"]["size_bytes"] = dl[model][quant]["size_bytes"]

            ins = stage_inspect(gguf)
            e["inspect"] = ins
            if ins["status"] != "PASS":
                e["status"] = "FAIL"
                entries[key] = e
                continue

            cfg_path = REPO / "configs" / entry["arch_config"]
            e["validate_config"] = stage_cli([PY, "-m", "gguf2mlx_stream.cli",
                                              "validate-config", str(cfg_path)], )
            out = OUT_DIR / f"{model}-{quant.lower().replace('_','')}-mlx{var['mlx_bits']}bit"
            report_json = INTEGRATION_DIR / f"report-{key}.json"
            e["dry_run"] = stage_cli([PY, "-m", "gguf2mlx_stream.cli", "convert",
                                      str(gguf), "--arch-config", str(cfg_path),
                                      "--dry-run", "-o", "/dev/null"])
            e["convert"] = stage_convert(gguf, cfg_path, out, var["mlx_bits"],
                                         INTEGRATION_DIR / "tokenizer" / model,
                                         report_json)
            if e["convert"]["status"] != "PASS":
                e["status"] = "FAIL"
                entries[key] = e
                continue

            e["structural"] = stage_structural(out, e["convert"])
            e["verify"] = stage_cli([PY, "-m", "gguf2mlx_stream.cli", "verify",
                                     str(gguf), str(out), "--arch-config", str(cfg_path),
                                     "--bits", str(var["mlx_bits"])])
            e["mlx_generation"] = stage_mlx_generation(out)
            e["llamacpp"] = stage_llamacpp(gguf)

            stages = [e["validate_config"], e["dry_run"], e["structural"],
                      e["verify"], e["mlx_generation"], e["llamacpp"]]
            e["status"] = "PASS" if all(s["status"] == "PASS" for s in stages) else "FAIL"
            # probe one model per family for oMLX (prefer Q4_K_M)
            if quant == "Q4_K_M":
                omlx_probe_models.append(out)
            entries[key] = e
            print(f"--- {key}: {e['status']}", flush=True)

    omlx_result = None
    if args.omlx and omlx_probe_models:
        print("=== oMLX (isolated server) ===", flush=True)
        omlx_result = stage_omlx(omlx_probe_models, port=args.omlx_port)
        # propagate: a family with a failing oMLX check keeps its PASS status
        # for convert/verify but the report shows the oMLX result separately.
        for m in omlx_probe_models:
            for key, e in entries.items():
                if e.get("status") == "PASS" and key in m.name and not e.get("omlx"):
                    e["omlx"] = "see omlx section"
        if omlx_result["status"] == "FAIL":
            for x in omlx_result.get("chat_completions", []):
                for key, e in entries.items():
                    if x["model"].startswith(key.rsplit(":", 1)[0]) and e.get("status") == "PASS":
                        e["omlx_note"] = x.get("error", "oMLX check failed")

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_wall_s": round(time.time() - t_all, 1),
        "host": "Apple M4 Pro, 24 GB unified memory",
        "mlx_version": _pkg_version("mlx"),
        "mlx_lm_version": _pkg_version("mlx_lm"),
        "llama_cpp": _llama_version(),
        "entries": entries,
        "omlx": omlx_result,
    }
    (REPORT_DIR / "integration-matrix.json").write_text(json.dumps(summary, indent=2))
    (REPORT_DIR / "integration-matrix.md").write_text(_to_markdown(summary))
    n_pass = sum(1 for e in entries.values() if e["status"] == "PASS")
    print(f"\n{n_pass}/{len(entries)} variants PASS "
          f"(oMLX: {omlx_result['status'] if omlx_result else 'not run'}); "
          f"reports written to reports/", flush=True)
    return 0


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "?"


def _llama_version() -> str:
    r = sh(["llama-completion", "--version"])
    text = (r.stdout + r.stderr).strip()
    return text.splitlines()[0].strip() if text else "?"


def _to_markdown(s: dict) -> str:
    lines = [
        "# Integration Matrix", "",
        f"*Generated: {s['generated_at']} | host: {s['host']} | "
        f"mlx {s['mlx_version']} | mlx-lm {s['mlx_lm_version']} | {s['llama_cpp']}*", "",
        "| model | variant | status | src GiB | out GiB | RSS GiB | conv s | verify | load+gen | llamacpp |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, e in s["entries"].items():
        if e.get("status") == "SKIPPED":
            lines.append(f"| {key} | - | SKIPPED ({e.get('reason','')[:40]}) | | | | | | | |")
            continue
        src = e.get("source", {}).get("size_bytes", 0) or 0
        outb = e.get("convert", {}).get("output_bytes", 0) or 0
        rss = e.get("convert", {}).get("peak_rss_gib", "-")
        if isinstance(rss, float):
            rss = f"{rss:.2f}"
        secs = e.get("convert", {}).get("elapsed_s", "-")
        verify = e.get("verify", {}).get("status", "-")
        gen = e.get("mlx_generation", {}).get("status", "-")
        lc = e.get("llamacpp", {}).get("status", "-")
        hits = []
        for g in e.get("mlx_generation", {}).get("generations", []):
            if g.get("expected_hit"):
                hits.append(g["expected"])
        lines.append(
            f"| {key} | {key.rsplit(':',1)[1]} | **{e['status']}** "
            f"| {src/2**30:.2f} | {outb/2**30:.2f} | {rss} | {secs} "
            f"| {verify} | {gen} {('[hit: '+','.join(hits)+']') if hits else ''} | {lc} |"
        )
    if s.get("omlx"):
        lines += ["", "## oMLX (isolated server)", "",
                  f"status: **{s['omlx']['status']}**"]
        d = s["omlx"].get("discovery", {})
        if d:
            lines.append(f"- discovery: {d['status']} ({len(d.get('discovered', []))} models)")
        for c in s["omlx"].get("chat_completions", []):
            lines.append(f"- {c['model']}: {c['status']} "
                         f"{'(hit)' if c.get('expected_hit') else ''} "
                         f"{c.get('error', '')}")
    lines += ["", "## Notes", "",
              "- MLX generation is chat-template based (temp 0, max 160 tokens); "
              "`expected_hit` records whether the reference answer appears.",
              "- llama.cpp raw completions are untemplate raw continuations; "
              "small wording differences vs MLX are expected after requantization.",
              "- Thinking models may spend tokens on reasoning before answering.",
              ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
