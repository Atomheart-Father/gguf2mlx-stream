"""Optional local integration tests (require real source GGUFs).

The original Nyx source GGUFs were deleted after the golden conversion, so
these tests are SKIPped by default. When a compatible GGUF is available
again, point the env vars at it and the full bounded-memory regression runs:

    GGUF2MLX_TEST_Q6_GGUF=/path/Nyx-RP-9B-Instruct-2608-v1.Q6_K.gguf \
    GGUF2MLX_TEST_Q4_GGUF=/path/Nyx-RP-9B-Instruct-2608-v1.Q4_K_M.gguf \
    GGUF2MLX_TEST_SOURCE_DIR=/path/dir-with-config-and-tokenizer \
    GGUF2MLX_TEST_LOAD=1 \          # optional mlx_lm.load + generation
    pytest tests/test_integration_nyx.py -s

Source GGUFs are only READ; outputs go to a fresh temp directory. Nothing
under the model library is ever written.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from gguf2mlx_stream.cli import main as cli_main
from gguf2mlx_stream.config.schema import load_arch_config
from gguf2mlx_stream.planner import plan_conversion
from gguf2mlx_stream.source.gguf import GGUFSource

ROOT = Path(__file__).resolve().parent.parent
QWEN35_YAML = str(ROOT / "configs" / "qwen3_5.yaml")

Q6 = os.environ.get("GGUF2MLX_TEST_Q6_GGUF")
Q4 = os.environ.get("GGUF2MLX_TEST_Q4_GGUF")
SOURCE_DIR = os.environ.get("GGUF2MLX_TEST_SOURCE_DIR")
DO_LOAD_TEST = os.environ.get("GGUF2MLX_TEST_LOAD") == "1"

pytestmark = pytest.mark.integration


def _ref_config() -> dict | None:
    if SOURCE_DIR:
        return json.loads((Path(SOURCE_DIR) / "config.json").read_text())
    return None


def _run_conversion(gguf: str, bits: int, tmp_path: Path, label: str) -> Path:
    out_dir = tmp_path / f"out-{label}"
    args = [
        "convert", gguf,
        "--arch-config", QWEN35_YAML,
        "--output", str(out_dir),
        "--bits", str(bits),
        "--group-size", "64",
        "--max-shard-gb", "4",
        "--report-json", str(tmp_path / f"report-{label}.json"),
    ]
    if SOURCE_DIR:
        args += ["--source-config", str(Path(SOURCE_DIR) / "config.json"),
                 "--tokenizer-source", SOURCE_DIR]
    rc = cli_main(args)
    assert rc == 0, f"conversion failed for {label}"
    return out_dir


def _plausible_output(out_dir: Path, bits: int) -> None:
    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    assert len(wm) == 927, f"expected 927 keys, got {len(wm)}"
    assert not any("layers.32." in k for k in wm), "MTP keys must be dropped"
    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["quantization"]["bits"] == bits
    # size regression reference: ~6.8 GiB (6-bit) / ~4.7 GiB (4-bit)
    size_gib = index["metadata"]["total_size"] / 2**30
    lo, hi = (5.8, 7.8) if bits == 6 else (3.8, 5.8)
    assert lo < size_gib < hi, f"output size {size_gib:.2f} GiB implausible for {bits}-bit"


@pytest.mark.skipif(not Q6, reason="SKIPPED: local source GGUF no longer available "
                                   "(set GGUF2MLX_TEST_Q6_GGUF)")
def test_integration_q6_to_6bit(tmp_path):
    assert Q6
    out_dir = _run_conversion(Q6, 6, tmp_path, "q6")
    _plausible_output(out_dir, 6)
    rc = cli_main(["verify", Q6, str(out_dir), "--arch-config", QWEN35_YAML, "--bits", "6"])
    assert rc == 0, "verify failed for 6-bit"
    report = json.loads((tmp_path / "report-q6.json").read_text())
    print(f"\n[q6] {report}")
    if DO_LOAD_TEST:
        from gguf2mlx_stream.verifier import load_test
        text = load_test(str(out_dir), prompt="The capital of France is", max_tokens=16)
        print(f"[q6] generation: {text!r}")
        assert text.strip(), "generation produced nothing"


@pytest.mark.skipif(not Q4, reason="SKIPPED: local source GGUF no longer available "
                                   "(set GGUF2MLX_TEST_Q4_GGUF)")
def test_integration_q4_to_4bit(tmp_path):
    assert Q4
    out_dir = _run_conversion(Q4, 4, tmp_path, "q4")
    _plausible_output(out_dir, 4)
    rc = cli_main(["verify", Q4, str(out_dir), "--arch-config", QWEN35_YAML, "--bits", "4"])
    assert rc == 0, "verify failed for 4-bit"
    report = json.loads((tmp_path / "report-q4.json").read_text())
    print(f"\n[q4] {report}")
    if DO_LOAD_TEST:
        from gguf2mlx_stream.verifier import load_test
        text = load_test(str(out_dir), prompt="The capital of France is", max_tokens=16)
        print(f"[q4] generation: {text!r}")
        assert text.strip(), "generation produced nothing"


@pytest.mark.skipif(not (Q6 and Path(Q6).is_file()), reason="source GGUF unavailable")
def test_plan_only_on_real_gguf(tmp_path):
    """Cheap planning smoke test whenever any real GGUF is provided."""
    cfg = load_arch_config(QWEN35_YAML)
    source = GGUFSource(Q6)
    plan = plan_conversion(cfg, source, ref_config=_ref_config())
    assert len(plan.jobs) == 137
    assert len(plan.dropped) == 15
    print("\n".join(plan.summary_lines()[:12]))
