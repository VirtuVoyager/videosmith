from __future__ import annotations

import asyncio
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from storysmith.models import (
    AssetKind,
    AssetRef,
    CostEntry,
    Mode,
    ProjectStatus,
    QAReport,
    QAVerdict,
    VideoProject,
)
from storysmith.settings import Settings
from storysmith.util import prompts
from storysmith.util.assets import latest_audio_master, latest_narration_assets
from storysmith.util.configs import Rubric, load_rubric, load_safety_negative_terms
from storysmith.util.ffmpeg import build_frame_extract_cmd, probe_duration, run_ffmpeg
from storysmith.util.text import word_error_rate

if TYPE_CHECKING:
    from storysmith.pipeline import PortBundle

_SYSTEM_PROMPT = (
    "You produce structured content contracts for StorySmith, an autonomous "
    "kids' shorts platform. Respond only via the emit tool."
)

_KEYFRAME_POSITIONS = (0.10, 0.50, 0.90)
_AUDIO_RETRY_KEY = -1  # sentinel in retry_counts for the audio track (not a scene index)


def _latest_scene_videos(state: VideoProject) -> list[tuple[int, AssetRef]]:
    latest: dict[int, AssetRef] = {}
    for asset in state.assets:
        if asset.kind != AssetKind.SCENE_VIDEO or asset.scene_index is None:
            continue
        current = latest.get(asset.scene_index)
        if current is None or asset.attempt > current.attempt:
            latest[asset.scene_index] = asset
    return sorted(latest.items())


def _rubric_text(rubric: Rubric, *, has_lesson: bool) -> str:
    lines = [
        f"- **{c.name}** (weight {c.weight}): {c.description}"
        for c in rubric.criteria
        if not (c.conditional and not has_lesson)
    ]
    return "\n".join(lines)


def _weighted_score(rubric: Rubric, scores: dict[str, float], *, has_lesson: bool) -> float:
    total = 0.0
    total_weight = 0.0
    for criterion in rubric.criteria:
        if criterion.conditional and not has_lesson:
            continue
        total += scores.get(criterion.name, 0.0) * criterion.weight
        total_weight += criterion.weight
    return total / total_weight if total_weight else 0.0


async def _extract_keyframes(ports: PortBundle, asset: AssetRef) -> list[bytes]:
    video_bytes = await ports.storage.get(uri=asset.uri)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "scene.mp4"
        video_path.write_bytes(video_bytes)
        duration = await asyncio.to_thread(probe_duration, video_path)
        frames = []
        for i, position in enumerate(_KEYFRAME_POSITIONS):
            frame_path = tmp_path / f"frame_{i}.jpg"
            await asyncio.to_thread(
                run_ffmpeg, build_frame_extract_cmd(video_path, duration * position, frame_path)
            )
            frames.append(frame_path.read_bytes())
        return frames


async def _score_scene(
    idx: int,
    asset: AssetRef,
    *,
    state: VideoProject,
    ports: PortBundle,
    rubric: Rubric,
    has_lesson: bool,
) -> tuple[QAReport, CostEntry]:
    assert state.style is not None
    keyframes = await _extract_keyframes(ports, asset)
    char_ref_bytes = None
    if state.style.characters and state.style.characters[0].image_uri:
        char_ref_bytes = await ports.storage.get(uri=state.style.characters[0].image_uri)
    images = [*keyframes, *([char_ref_bytes] if char_ref_bytes else [])]

    # SPEC-GAP: lesson_note is always empty -- VideoProject has no field
    # carrying a theme's `lesson` text yet (configs/themes.yaml civic/hygiene
    # themes have one, but WP2's auto-mode theme sampling was deferred
    # pending a persistence layer, so nothing threads it into state here).
    # has_lesson is likewise always False until that lands.
    user_prompt = prompts.load(
        "critic",
        scene_index=idx,
        style_json=state.style.model_dump_json(),
        rubric_text=_rubric_text(rubric, has_lesson=has_lesson),
        lesson_note="",
    )
    obj, cost = await ports.llm.complete_structured(
        system=_SYSTEM_PROMPT,
        user=user_prompt,
        schema=QAReport,
        model_tier="vision",
        images=images,
    )
    assert isinstance(obj, QAReport)
    cost_entry = CostEntry(
        at=datetime.now(UTC), item=f"critic:scene{idx}", provider="llm", cost_usd=cost
    )
    return obj, cost_entry


async def _score_audio(
    state: VideoProject, *, ports: PortBundle
) -> tuple[str, str, list[CostEntry]]:
    """Returns (transcript, expected_text, cost_entries)."""
    assert state.manifest is not None
    cost_entries: list[CostEntry] = []

    if state.mode == Mode.RHYME:
        master = latest_audio_master(state.assets)
        assert master is not None
        audio_bytes = await ports.storage.get(uri=master.uri)
        words, cost = await ports.transcribe.transcribe(audio=audio_bytes)
        cost_entries.append(
            CostEntry(
                at=datetime.now(UTC), item="critic:audio", provider="transcribe", cost_usd=cost
            )
        )
        transcript = " ".join(str(w["word"]) for w in words)
        expected = state.manifest.lyrics or ""
        return transcript, expected, cost_entries

    narration_assets = latest_narration_assets(state.assets)
    scenes_by_index = {s.index: s for s in state.manifest.scenes}
    transcript_parts: list[str] = []
    expected_parts: list[str] = []
    for asset in narration_assets:
        audio_bytes = await ports.storage.get(uri=asset.uri)
        words, cost = await ports.transcribe.transcribe(audio=audio_bytes)
        cost_entries.append(
            CostEntry(
                at=datetime.now(UTC),
                item=f"critic:audio:scene{asset.scene_index}",
                provider="transcribe",
                cost_usd=cost,
            )
        )
        transcript_parts.append(" ".join(str(w["word"]) for w in words))
        if asset.scene_index is not None:
            expected_parts.append(scenes_by_index[asset.scene_index].narration)
    return " ".join(transcript_parts), " ".join(expected_parts), cost_entries


async def run(state: VideoProject, *, ports: PortBundle, settings: Settings) -> dict[str, Any]:
    rubric = load_rubric(settings.configs_dir)
    base_safety_terms = load_safety_negative_terms(settings.configs_dir)
    has_lesson = False  # see SPEC-GAP note in _score_scene

    retry_counts = dict(state.retry_counts)
    reports: list[QAReport] = []
    cost_entries: list[CostEntry] = []

    for idx, asset in _latest_scene_videos(state):
        raw_report, cost = await _score_scene(
            idx, asset, state=state, ports=ports, rubric=rubric, has_lesson=has_lesson
        )
        weighted = _weighted_score(rubric, raw_report.scores, has_lesson=has_lesson)
        if raw_report.safety_flags:
            verdict = QAVerdict.HUMAN_REVIEW  # never auto-retry safety issues (§6)
        elif weighted >= rubric.pass_threshold:
            verdict = QAVerdict.PASS
        else:
            retry_counts[idx] = retry_counts.get(idx, 0) + 1
            verdict = QAVerdict.RETRY
            if retry_counts[idx] >= rubric.max_attempts_before_human_review:
                verdict = QAVerdict.HUMAN_REVIEW
        reports.append(raw_report.model_copy(update={"scene_index": idx, "verdict": verdict}))
        cost_entries.append(cost)

    transcript, expected, audio_costs = await _score_audio(state, ports=ports)
    cost_entries.extend(audio_costs)
    wer = word_error_rate(expected, transcript) if expected else 0.0
    transcript_lower = transcript.lower()
    audio_safety_flags = [term for term in base_safety_terms if term.lower() in transcript_lower]

    if audio_safety_flags:
        audio_verdict = QAVerdict.HUMAN_REVIEW
    elif wer <= rubric.audio_wer_retry_threshold:
        audio_verdict = QAVerdict.PASS
    else:
        retry_counts[_AUDIO_RETRY_KEY] = retry_counts.get(_AUDIO_RETRY_KEY, 0) + 1
        audio_verdict = QAVerdict.RETRY
        if retry_counts[_AUDIO_RETRY_KEY] >= rubric.audio_max_attempts_before_human_review:
            audio_verdict = QAVerdict.HUMAN_REVIEW

    reports.append(
        QAReport(
            scene_index=None,
            verdict=audio_verdict,
            scores={"audio_accuracy": max(0.0, 1.0 - wer)},
            safety_flags=audio_safety_flags,
            critique=(
                ""
                if audio_verdict == QAVerdict.PASS
                else f"Audio word error rate {wer:.2f} exceeds threshold "
                f"{rubric.audio_wer_retry_threshold}; regenerate narration/lyrics."
            ),
        )
    )

    return {
        "qa_reports": reports,
        "retry_counts": retry_counts,
        "cost_ledger": cost_entries,
        "status": ProjectStatus.QA,
    }
