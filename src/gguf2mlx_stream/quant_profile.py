"""Opt-in per-tensor quantization profile overlay (``--quant-profile``).

A *quant profile* is a small, declarative JSON document that overrides the
conversion-level quantization for subsets of output rules — the research
counterpart of mlx-lm's ``mixed_3_4``-style mixed quantization. It exists so
experiments can express things like "lm_head and the value/down paths at 4
bits, everything else at 3 bits" without duplicating an architecture config.

This is strictly opt-in: without ``--quant-profile`` conversion behavior is
byte-for-byte identical to before. Profiles are data, not code: the loader
accepts a fixed schema (regex + bit parameters), compiles the regexes, and
rejects anything else. There is no expression evaluation.

Profile document::

    {
      "name": "mixed_3_4_g64",
      "default": {"bits": 3, "group_size": 64},
      "rules": [
        {"match": "lm_head", "bits": 8},
        {"match": "down_proj", "bits": 4, "group_size": 64}
      ]
    }

Semantics:

* ``rules[].match`` is a Python ``re`` pattern searched against each rule's
  **dest template** (``{n}``/``{i}`` placeholders intact). First match wins.
* Matching applies only to rules that quantize; drop/copy rules are ignored.
* A match replaces the rule's ``bits``/``group_size`` with the overlay's
  values (fields left ``null`` in the overlay keep the rule's current value,
  which may itself come from the architecture config).
* ``default`` (optional) replaces the *conversion-level* bits/group size for
  rules no overlay matched. When the user passes an explicit ``--bits N`` and
  the profile also declares a default, that is treated as a conflict and
  rejected; ``--bits auto`` plus a profile default resolves to the profile.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config.schema import ArchConfig, Rule
from .constants import SUPPORTED_BITS, SUPPORTED_GROUP_SIZES
from .errors import ConversionError

_MAX_PATTERN_LEN = 512


@dataclass(frozen=True)
class ProfileOverlay:
    pattern: re.Pattern[str]
    bits: int | None
    group_size: int | None


@dataclass(frozen=True)
class QuantProfile:
    name: str
    default_bits: int | None
    default_group_size: int | None
    overlays: tuple[ProfileOverlay, ...]

    def as_record(self) -> dict:
        """JSON-serializable summary for output config.json provenance."""
        return {
            "name": self.name,
            "default_bits": self.default_bits,
            "default_group_size": self.default_group_size,
            "rules": [
                {"match": o.pattern.pattern, "bits": o.bits, "group_size": o.group_size}
                for o in self.overlays
            ],
        }


@dataclass(frozen=True)
class OverlayApplication:
    rule_display_name: str
    dest_template: str
    bits: int | None
    group_size: int | None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConversionError(f"quant profile: {message}")


def _parse_quant_spec(raw: object, where: str) -> tuple[int | None, int | None]:
    _require(isinstance(raw, dict), f"{where} must be an object")
    assert isinstance(raw, dict)
    unknown = set(raw) - {"bits", "group_size"}
    _require(not unknown, f"{where} has unknown keys: {sorted(unknown)}")
    bits = raw.get("bits")
    group = raw.get("group_size")
    _require(bits is not None or group is not None,
             f"{where} must set at least one of bits/group_size")
    _require(bits is None or (isinstance(bits, int) and not isinstance(bits, bool) and bits in SUPPORTED_BITS),
             f"{where}.bits must be one of {list(SUPPORTED_BITS)}")
    _require(group is None or (isinstance(group, int) and not isinstance(group, bool) and group in SUPPORTED_GROUP_SIZES),
             f"{where}.group_size must be one of {list(SUPPORTED_GROUP_SIZES)}")
    return bits, group


def load_quant_profile(path: str | Path) -> QuantProfile:
    """Load and validate a profile JSON document."""
    try:
        doc = json.loads(Path(path).read_text())
    except OSError as exc:
        raise ConversionError(f"quant profile: cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConversionError(f"quant profile: {path} is not valid JSON: {exc}") from exc
    _require(isinstance(doc, dict), f"{path} must contain a JSON object")
    assert isinstance(doc, dict)
    unknown = set(doc) - {"name", "default", "rules"}
    _require(not unknown, f"{path} has unknown keys: {sorted(unknown)}")

    name = doc.get("name")
    _require(isinstance(name, str) and name.strip(), f"{path}: 'name' must be a non-empty string")
    _require(len(name) <= 128, f"{path}: 'name' too long")

    default_bits: int | None = None
    default_group: int | None = None
    if "default" in doc:
        default_bits, default_group = _parse_quant_spec(doc["default"], "default")

    rules_raw = doc.get("rules", [])
    _require(isinstance(rules_raw, list), f"{path}: 'rules' must be a list")
    overlays: list[ProfileOverlay] = []
    for idx, entry in enumerate(rules_raw):
        where = f"rules[{idx}]"
        _require(isinstance(entry, dict), f"{path}: {where} must be an object")
        assert isinstance(entry, dict)
        unknown = set(entry) - {"match", "bits", "group_size"}
        _require(not unknown, f"{path}: {where} has unknown keys: {sorted(unknown)}")
        pattern_raw = entry.get("match")
        _require(isinstance(pattern_raw, str) and pattern_raw.strip(),
                 f"{path}: {where}.match must be a non-empty string")
        assert isinstance(pattern_raw, str)
        _require(len(pattern_raw) <= _MAX_PATTERN_LEN,
                 f"{path}: {where}.match exceeds {_MAX_PATTERN_LEN} characters")
        try:
            pattern = re.compile(pattern_raw)
        except re.error as exc:
            raise ConversionError(f"quant profile: {path}: {where}.match: {exc}") from exc
        bits, group = _parse_quant_spec(
            {"bits": entry.get("bits"), "group_size": entry.get("group_size")}, where
        )
        overlays.append(ProfileOverlay(pattern=pattern, bits=bits, group_size=group))

    return QuantProfile(
        name=name,
        default_bits=default_bits,
        default_group_size=default_group,
        overlays=tuple(overlays),
    )


def apply_quant_profile(
    config: ArchConfig, profile: QuantProfile
) -> tuple[ArchConfig, list[OverlayApplication], tuple[str, ...]]:
    """Return a new ArchConfig with overlay-matched rules' quant params replaced.

    Also returns the list of applications (for logging/records) and the
    overlay patterns that matched no quantized rule (likely typos).
    """
    new_rules: list[Rule] = []
    applications: list[OverlayApplication] = []
    matched_patterns: set[str] = set()

    for rule in config.rules:
        if not (rule.quantize and not rule.drop):
            new_rules.append(rule)
            continue
        replacement: Rule | None = None
        for overlay in profile.overlays:
            if overlay.pattern.search(rule.dest or ""):
                replacement = dataclasses.replace(
                    rule,
                    bits=overlay.bits if overlay.bits is not None else rule.bits,
                    group_size=(
                        overlay.group_size
                        if overlay.group_size is not None
                        else rule.group_size
                    ),
                )
                applications.append(
                    OverlayApplication(
                        rule_display_name=rule.display_name,
                        dest_template=rule.dest or "",
                        bits=replacement.bits,
                        group_size=replacement.group_size,
                    )
                )
                matched_patterns.add(overlay.pattern.pattern)
                break
        new_rules.append(replacement if replacement is not None else rule)

    unmatched = tuple(
        o.pattern.pattern
        for o in profile.overlays
        if o.pattern.pattern not in matched_patterns
    )
    return (
        dataclasses.replace(config, rules=tuple(new_rules)),
        applications,
        unmatched,
    )


def resolve_default_bits(args_bits: str | int | None, profile: QuantProfile | None) -> str | int | None:
    """Resolve the effective conversion-level bits under ``--quant-profile``.

    * explicit ``--bits N`` + profile default -> conflict error
    * ``--bits auto`` (or None) + profile default -> the profile default
    * otherwise -> the original value unchanged
    """
    if profile is None or profile.default_bits is None:
        return args_bits
    if isinstance(args_bits, int):
        raise ConversionError(
            f"--bits {args_bits} conflicts with quant profile '{profile.name}' "
            f"default bits {profile.default_bits}; the profile default would be "
            "ignored — pass only one"
        )
    return profile.default_bits
