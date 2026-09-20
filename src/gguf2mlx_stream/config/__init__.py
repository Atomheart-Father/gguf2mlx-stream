"""Architecture config loading (YAML -> validated dataclasses)."""

from .schema import ArchConfig, Rule, arch_config_from_dict, load_arch_config

__all__ = ["ArchConfig", "Rule", "arch_config_from_dict", "load_arch_config"]
