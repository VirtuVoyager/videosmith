from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from storysmith.errors import LLMStructuredOutputError
from storysmith.models import CostEntry, ProjectStatus, SceneManifest, VideoProject
from storysmith.settings import Settings
from storysmith.util import prompts

if TYPE_CHECKING:
    from storysmith.pipeline import PortBundle

_SYSTEM_PROMPT = (
    "You produce structured content contracts for StorySmith, an autonomous "
    "kids' shorts platform. Respond only via the emit tool."
)

_MIN_SCENES = 4
_MAX_SCENES = 7
_MIN_DURATION_S = 30.0
_MAX_DURATION_S = 60.0
_DURATION_TOLERANCE_S = 2.0
_MIN_STYLE_WORD_LEN = 3


def _validation_violations(manifest: SceneManifest, art_style: str) -> list[str]:
    """Code-level post-validation from §2.2 -- never left to the LLM to self-check."""
    violations: list[str] = []

    scene_count = len(manifest.scenes)
    if not (_MIN_SCENES <= scene_count <= _MAX_SCENES):
        violations.append(
            f"scene count is {scene_count}, must be between {_MIN_SCENES} and {_MAX_SCENES}"
        )

    total = sum(scene.duration_s for scene in manifest.scenes)
    lo, hi = _MIN_DURATION_S - _DURATION_TOLERANCE_S, _MAX_DURATION_S + _DURATION_TOLERANCE_S
    if not (lo <= total <= hi):
        violations.append(
            f"total duration is {total:.1f}s, must be within "
            f"[{_MIN_DURATION_S}, {_MAX_DURATION_S}]s (tolerance {_DURATION_TOLERANCE_S}s)"
        )

    style_words = [
        word.strip(",.")
        for word in art_style.lower().split()
        if len(word.strip(",.")) > _MIN_STYLE_WORD_LEN
    ]
    for scene in manifest.scenes:
        prompt_lower = scene.video_prompt.lower()
        if style_words and not any(word in prompt_lower for word in style_words):
            violations.append(
                f"scene {scene.index} video_prompt doesn't restate the art style: "
                f"{scene.video_prompt!r}"
            )

    return violations


def _violation_note(violations: list[str]) -> str:
    if not violations:
        return ""
    bullets = "\n".join(f"- {v}" for v in violations)
    return (
        "## Correction needed\n"
        f"Your previous SceneManifest violated:\n{bullets}\n"
        "Fix these issues and resend the full SceneManifest."
    )


async def run(state: VideoProject, *, ports: PortBundle, settings: Settings) -> dict[str, Any]:
    assert state.style is not None
    style = state.style

    async def _ask(violation_note: str) -> tuple[SceneManifest, float]:
        user_prompt = prompts.load(
            "director",
            brief=state.brief,
            mode=state.mode.value,
            style_json=style.model_dump_json(),
            violation_note=violation_note,
        )
        obj, cost = await ports.llm.complete_structured(
            system=_SYSTEM_PROMPT, user=user_prompt, schema=SceneManifest, model_tier="standard"
        )
        assert isinstance(obj, SceneManifest)
        return obj, cost

    manifest, cost = await _ask("")
    cost_entries = [
        CostEntry(at=datetime.now(UTC), item="director:manifest", provider="llm", cost_usd=cost)
    ]

    violations = _validation_violations(manifest, style.art_style)
    if violations:
        manifest, cost2 = await _ask(_violation_note(violations))
        cost_entries.append(
            CostEntry(
                at=datetime.now(UTC),
                item="director:manifest_corrective",
                provider="llm",
                cost_usd=cost2,
            )
        )
        violations = _validation_violations(manifest, style.art_style)
        if violations:
            raise LLMStructuredOutputError(
                "Director's SceneManifest still violates constraints after the corrective "
                "round: " + "; ".join(violations)
            )

    return {
        "manifest": manifest,
        "status": ProjectStatus.SCRIPTED,
        "cost_ledger": cost_entries,
    }
