from __future__ import annotations

from pathlib import Path

import yaml


def load_safety_negative_terms(configs_dir: str) -> list[str]:
    """Base negative_terms merged into every StyleContract, never left to the LLM (§2.2)."""
    path = Path(configs_dir) / "safety" / "safety_rules.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [str(term) for term in data.get("base_negative_terms", [])]


def load_style_preset_yaml(configs_dir: str, preset_name: str) -> str:
    """Raw YAML text of a style preset, fed into the Creative Director prompt as-is."""
    path = Path(configs_dir) / "style_presets" / f"{preset_name}.yaml"
    return path.read_text(encoding="utf-8")
