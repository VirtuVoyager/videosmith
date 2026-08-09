from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RubricCriterion:
    name: str
    description: str
    weight: float
    min_score: float
    conditional: bool = False  # only scored/counted when the project has a lesson (§6)


@dataclass(frozen=True)
class Rubric:
    criteria: list[RubricCriterion]
    pass_threshold: float
    max_attempts_before_human_review: int
    audio_wer_retry_threshold: float
    audio_max_attempts_before_human_review: int


def load_rubric(configs_dir: str) -> Rubric:
    path = Path(configs_dir) / "rubrics" / "critic_rubric.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    criteria = [
        RubricCriterion(
            name=c["name"],
            description=c["description"],
            weight=float(c["weight"]),
            min_score=float(c["min_score"]),
            conditional=bool(c.get("conditional", False)),
        )
        for c in data["criteria"]
    ]
    return Rubric(
        criteria=criteria,
        pass_threshold=float(data["pass_threshold"]),
        max_attempts_before_human_review=int(data["max_attempts_before_human_review"]),
        audio_wer_retry_threshold=float(data["audio_wer_retry_threshold"]),
        audio_max_attempts_before_human_review=int(data["audio_max_attempts_before_human_review"]),
    )


def load_safety_negative_terms(configs_dir: str) -> list[str]:
    """Base negative_terms merged into every StyleContract, never left to the LLM (§2.2)."""
    path = Path(configs_dir) / "safety" / "safety_rules.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [str(term) for term in data.get("base_negative_terms", [])]


def load_style_preset_yaml(configs_dir: str, preset_name: str) -> str:
    """Raw YAML text of a style preset, fed into the Creative Director prompt as-is."""
    path = Path(configs_dir) / "style_presets" / f"{preset_name}.yaml"
    return path.read_text(encoding="utf-8")
