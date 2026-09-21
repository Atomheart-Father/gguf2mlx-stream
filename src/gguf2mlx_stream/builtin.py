"""Built-in architecture configuration discovery.

The official architecture configs (``qwen3_5``, ``qwen3``, ``llama``,
``gemma3``) live in the repository-root ``configs/`` directory, which is the
single authoritative source. At wheel build time hatchling force-includes
that directory into the wheel as ``gguf2mlx_stream/configs/``, so an
installed package can resolve every official config without a source
checkout:

    gguf2mlx-stream convert model.gguf --arch-config llama ...

``--arch-config`` accepts either a config *name* (resolved through this
module) or a path to any YAML file. When it is omitted entirely, the
converter auto-detects the config from the GGUF's ``general.architecture``
if exactly one built-in config accepts it.

During development from a source checkout (editable install) the packaged
copy does not exist; discovery then falls back to the repository-root
``configs/`` directory next to ``src/``.
"""

from __future__ import annotations

import os
from importlib import resources
from typing import Any

import yaml

from .config.schema import ArchConfig, arch_config_from_dict
from .errors import ConfigError


def _builtin_config_dir() -> Any | None:
    """``gguf2mlx_stream/configs`` resource when present (installed wheel)."""
    try:
        root = resources.files("gguf2mlx_stream")
    except ModuleNotFoundError:  # pragma: no cover - package always importable here
        return None
    configs = root / "configs"
    try:
        if configs.is_dir():
            return configs
    except (FileNotFoundError, NotADirectoryError):
        pass
    return None


def _repo_config_dir() -> str | None:
    """Repository-root ``configs/`` for source checkouts (editable installs)."""
    candidate = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "configs",
    )
    if os.path.isdir(candidate):
        return candidate
    return None


def builtin_config_names() -> list[str]:
    """Names of the available built-in architecture configs (sorted)."""
    names: set[str] = set()
    pkg_dir = _builtin_config_dir()
    if pkg_dir is not None:
        try:
            for entry in pkg_dir.iterdir():
                if entry.name.endswith(".yaml") and entry.is_file():
                    names.add(entry.name[: -len(".yaml")])
        except (FileNotFoundError, NotADirectoryError):
            pass
    repo_dir = _repo_config_dir()
    if repo_dir is not None:
        for fn in os.listdir(repo_dir):
            if fn.endswith(".yaml"):
                names.add(fn[: -len(".yaml")])
    return sorted(names)


def load_builtin_config(name: str) -> ArchConfig:
    """Load a built-in config by name (package copy first, repo fallback)."""
    pkg_dir = _builtin_config_dir()
    if pkg_dir is not None:
        entry = pkg_dir / f"{name}.yaml"
        try:
            if entry.is_file():
                raw = yaml.safe_load(entry.read_text(encoding="utf-8"))
                return arch_config_from_dict(raw, path=f"builtin:{name}.yaml")
        except (FileNotFoundError, NotADirectoryError):
            pass
    repo_dir = _repo_config_dir()
    if repo_dir is not None:
        path = os.path.join(repo_dir, f"{name}.yaml")
        if os.path.isfile(path):
            return arch_config_from_dict(_read_yaml(path), path=path)
    raise ConfigError(
        f"unknown built-in architecture config {name!r}; available: "
        f"{builtin_config_names()}"
    )


def _read_yaml(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_arch_config(
    value: str | None, gguf_arch: str | None = None
) -> tuple[ArchConfig, str]:
    """Resolve ``--arch-config`` (path or built-in name) for a conversion.

    Returns ``(config, description)``. When ``value`` is None the config is
    auto-detected from the GGUF's ``general.architecture`` if exactly one
    built-in config accepts it.
    """
    if value is not None:
        if os.path.isfile(value):
            return load_from_file(value), value
        names = builtin_config_names()
        if value in names:
            return load_builtin_config(value), f"builtin:{value}.yaml"
        raise ConfigError(
            f"architecture config {value!r} is neither an existing file nor a "
            f"built-in config name; available built-ins: {names}"
        )
    if gguf_arch is None:
        raise ConfigError(
            "no architecture config given: pass --arch-config (a built-in "
            f"name such as {builtin_config_names()} or a YAML path)"
        )
    matches: list[tuple[str, ArchConfig]] = []
    for name in builtin_config_names():
        cfg = load_builtin_config(name)
        if cfg.architecture.gguf_arch is not None and cfg.architecture.accepts(gguf_arch):
            matches.append((name, cfg))
    if not matches:
        raise ConfigError(
            f"no built-in architecture config accepts GGUF architecture "
            f"{gguf_arch!r}; available built-ins: {builtin_config_names()}"
        )
    if len(matches) > 1:
        raise ConfigError(
            f"GGUF architecture {gguf_arch!r} is accepted by several built-in "
            f"configs ({[n for n, _ in matches]}); pass --arch-config explicitly"
        )
    name, cfg = matches[0]
    return cfg, f"builtin:{name}.yaml (auto-detected from general.architecture={gguf_arch!r})"


def load_from_file(path: str) -> ArchConfig:
    """Load and validate an architecture config from an explicit YAML path."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return arch_config_from_dict(raw, path=path)
