"""Packaging tests: built-in configs, wheel contents, clean installability.

Guarantees under test (release P1):

* the five official architecture configs are discoverable as built-ins and
  load/validate without a source checkout;
* the built wheel contains them verbatim under ``gguf2mlx_stream/configs/``
  (byte-identical to the authoritative repository-root ``configs/``);
* the wheel metadata version matches the runtime ``__version__``;
* the CLI resolves ``--arch-config`` by name, by path, and by auto-detecting
  the GGUF architecture.
"""

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

OFFICIAL = ("gemma3", "llama", "qwen3", "qwen3_5", "qwen3_5_moe")


def test_builtin_config_names_and_load():
    from gguf2mlx_stream.builtin import builtin_config_names, load_builtin_config

    assert builtin_config_names() == list(OFFICIAL)
    for name in OFFICIAL:
        cfg = load_builtin_config(name)
        assert cfg.rules, name
        assert cfg.architecture.id == name
        assert cfg.output.model_type, name


def test_resolve_arch_config_by_name_and_path():
    from gguf2mlx_stream.builtin import resolve_arch_config
    from gguf2mlx_stream.errors import ConfigError

    cfg, desc = resolve_arch_config("llama")
    assert cfg.architecture.id == "llama"
    assert desc.startswith("builtin:")

    cfg, desc = resolve_arch_config(str(ROOT / "configs" / "llama.yaml"))
    assert cfg.architecture.id == "llama"
    assert desc.endswith("llama.yaml")

    with pytest.raises(ConfigError, match="neither an existing file"):
        resolve_arch_config("no-such-config")
    with pytest.raises(ConfigError):
        resolve_arch_config(None, None)
    with pytest.raises(ConfigError, match="no built-in"):
        resolve_arch_config(None, "definitely_not_an_arch")


def test_auto_detect_ambiguous_and_unique():
    from gguf2mlx_stream.builtin import resolve_arch_config

    cfg, desc = resolve_arch_config(None, "llama")
    assert cfg.architecture.id == "llama"
    assert "auto-detected" in desc
    # qwen35 is accepted only by the qwen3_5 config
    cfg, desc = resolve_arch_config(None, "qwen35")
    assert cfg.architecture.id == "qwen3_5"


def test_cli_list_configs_and_validate_by_name(capsys):
    from gguf2mlx_stream.cli import main as cli_main

    assert cli_main(["list-configs"]) == 0
    out = capsys.readouterr().out
    for name in OFFICIAL:
        assert name in out

    assert cli_main(["validate-config", "gemma3"]) == 0
    assert "gemma3" in capsys.readouterr().out


def test_wheel_contains_builtin_configs_verbatim(tmp_path):
    if importlib.util.find_spec("hatchling") is None:  # pragma: no cover
        pytest.skip("hatchling not installed in the test environment")
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    r = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "--wheel-dir", str(wheel_dir), str(ROOT)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    wheel = next(wheel_dir.glob("gguf2mlx_stream-*.whl"))

    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        for name in OFFICIAL:
            member = f"gguf2mlx_stream/configs/{name}.yaml"
            assert member in names, f"{member} missing from wheel"
            # byte-identical to the authoritative repository-root configs/
            assert z.read(member) == (ROOT / "configs" / f"{name}.yaml").read_bytes()
        # configs are packaged under the package namespace, not dumped at the
        # wheel root as a stray top-level directory
        assert not any(n.startswith("configs/") for n in names)
        # runtime version and wheel metadata agree
        meta_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        version = next(
            line.split(": ", 1)[1]
            for line in z.read(meta_name).decode().splitlines()
            if line.startswith("Version:")
        )
    from gguf2mlx_stream import __version__
    assert version == __version__ == "0.1.0a1"


def test_convert_without_arch_config_autodetects(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent))
    from conftest import write_minimal_tokenizer  # noqa: E402
    from test_pipeline_synthetic import build_fixture_gguf  # noqa: E402
    from gguf2mlx_stream.cli import main as cli_main

    gguf_path, _, _, _ = build_fixture_gguf(tmp_path)  # arch "qwen35"
    out_dir = tmp_path / "out"
    rc = cli_main([
        "convert", str(gguf_path), "--output", str(out_dir),
        "--tokenizer-source", write_minimal_tokenizer(tmp_path / "tokenizer"),
        "--bits", "4", "--quiet",
    ])
    assert rc == 0, "auto-detected conversion failed"
    assert (out_dir / "config.json").is_file()
