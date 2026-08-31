from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from storysmith.errors import LLMStructuredOutputError
from storysmith.models import (
    CostEntry,
    ProjectStatus,
    Scene,
    SceneGenMode,
    SceneManifest,
    StyleContract,
    VideoProject,
)
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

# Amendment 01 §3: crude noun-ish token overlap check between an i2v scene's
# scene_image_prompt (composition, fixed) and video_prompt (motion only) --
# not real NLP/POS tagging, just a stopword-filtered word-overlap heuristic
# that's good enough to catch a video_prompt that redundantly re-describes
# layout the still already fixed.
_MOTION_PROMPT_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "with",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "as",
    "by",
    "from",
    "into",
    "over",
    "under",
    "static",
    "camera",
    "remains",
    "stays",
    "stay",
    "fixed",
    "nothing",
    "else",
    "moves",
    "move",
    "moving",
    "then",
    "while",
    "only",
    "does",
    "not",
    "no",
}
_LAYOUT_OVERLAP_THRESHOLD = 0.5


def _significant_words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z]+", text.lower())
        if len(word) > _MIN_STYLE_WORD_LEN and word not in _MOTION_PROMPT_STOPWORDS
    }


def _scene_violations(scene: Scene, style_words: list[str], character_names: set[str]) -> list[str]:
    violations: list[str] = []

    if scene.dialogue:
        unknown_speakers = {turn.speaker for turn in scene.dialogue} - character_names
        if unknown_speakers:
            violations.append(
                f"scene {scene.index} dialogue has speaker(s) {sorted(unknown_speakers)} "
                f"not in the cast ({sorted(character_names)}) -- every speaker must exactly "
                "match a character name"
            )

    if scene.gen_mode == SceneGenMode.I2V:
        if not scene.scene_image_prompt or not scene.scene_image_prompt.strip():
            violations.append(
                f"scene {scene.index} has gen_mode=i2v but no scene_image_prompt "
                "-- every i2v scene needs a complete static composition prompt"
            )
        else:
            # §2.2's art-style-restatement rule exists because a t2v video
            # model gets nothing but video_prompt as context. An i2v scene's
            # model also sees the composed still (which scene_image_prompt
            # already restates the art style into) -- amendment §4
            # deliberately tells video_prompt to describe motion only, not
            # style/layout, so checking style words against video_prompt
            # alone would fight that instruction. Check the combined text
            # instead: the style still has to appear *somewhere* in the two
            # prompts together, just not necessarily in video_prompt itself.
            combined_lower = f"{scene.scene_image_prompt} {scene.video_prompt}".lower()
            if style_words and not any(word in combined_lower for word in style_words):
                violations.append(
                    f"scene {scene.index}'s scene_image_prompt/video_prompt together "
                    f"don't restate the art style: {scene.scene_image_prompt!r} / "
                    f"{scene.video_prompt!r}"
                )
            image_words = _significant_words(scene.scene_image_prompt)
            motion_words = _significant_words(scene.video_prompt)
            overlap = image_words & motion_words
            if image_words and len(overlap) / len(image_words) > _LAYOUT_OVERLAP_THRESHOLD:
                violations.append(
                    f"scene {scene.index} video_prompt repeats scene-composition words "
                    f"already fixed by scene_image_prompt ({sorted(overlap)}) -- "
                    "video_prompt must describe motion only, not layout"
                )
    else:
        # t2v: video_prompt is the scene's only prompt, so §2.2's original
        # rule applies unchanged -- it must restate the art style itself.
        prompt_lower = scene.video_prompt.lower()
        if style_words and not any(word in prompt_lower for word in style_words):
            violations.append(
                f"scene {scene.index} video_prompt doesn't restate the art style: "
                f"{scene.video_prompt!r}"
            )
    return violations


def _validation_violations(manifest: SceneManifest, style: StyleContract) -> list[str]:
    """Code-level post-validation from §2.2 (+ Amendment 01 §3, Amendment 02) --
    never left to the LLM to self-check."""
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
        for word in style.art_style.lower().split()
        if len(word.strip(",.")) > _MIN_STYLE_WORD_LEN
    ]
    character_names = {c.name for c in style.characters}
    for scene in manifest.scenes:
        violations.extend(_scene_violations(scene, style_words, character_names))

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
            default_gen_mode=settings.default_scene_gen_mode,
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

    violations = _validation_violations(manifest, style)
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
        violations = _validation_violations(manifest, style)
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
