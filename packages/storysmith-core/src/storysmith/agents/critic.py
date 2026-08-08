from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from storysmith.models import (
    AssetKind,
    AssetRef,
    CostEntry,
    ProjectStatus,
    QAReport,
    QAVerdict,
    VideoProject,
)
from storysmith.settings import Settings

if TYPE_CHECKING:
    from storysmith.pipeline import PortBundle

_MAX_ATTEMPTS_BEFORE_HUMAN_REVIEW = 3

# SPEC-GAP: real rubric-driven prompt (configs/rubrics/critic_rubric.yaml,
# keyframe extraction, vision LLM call per §6) is WP6 scope. This placeholder
# still calls the LLM port (so cost_ledger/stub wiring is exercised) but
# doesn't do real keyframe/vision analysis.
_SYSTEM_PROMPT = "You are the Critic. Score the scene against the style contract and rubric."


def _latest_scene_videos(state: VideoProject) -> list[tuple[int, AssetRef]]:
    latest: dict[int, AssetRef] = {}
    for asset in state.assets:
        if asset.kind != AssetKind.SCENE_VIDEO or asset.scene_index is None:
            continue
        current = latest.get(asset.scene_index)
        if current is None or asset.attempt > current.attempt:
            latest[asset.scene_index] = asset
    return sorted(latest.items())


async def run(state: VideoProject, *, ports: PortBundle, settings: Settings) -> dict[str, Any]:
    reports: list[QAReport] = []
    retry_counts = dict(state.retry_counts)
    cost_entries: list[CostEntry] = []

    for idx, _asset in _latest_scene_videos(state):
        obj, cost = await ports.llm.complete_structured(
            system=_SYSTEM_PROMPT,
            user=f"scene_index={idx}",
            schema=QAReport,
            model_tier="vision",
        )
        assert isinstance(obj, QAReport)
        report = obj.model_copy(update={"scene_index": idx})
        if report.verdict == QAVerdict.RETRY:
            retry_counts[idx] = retry_counts.get(idx, 0) + 1
            if retry_counts[idx] >= _MAX_ATTEMPTS_BEFORE_HUMAN_REVIEW:
                report = report.model_copy(update={"verdict": QAVerdict.HUMAN_REVIEW})
        reports.append(report)
        cost_entries.append(
            CostEntry(
                at=datetime.now(UTC), item=f"critic:scene{idx}", provider="llm", cost_usd=cost
            )
        )

    audio_obj, audio_cost = await ports.llm.complete_structured(
        system=_SYSTEM_PROMPT,
        user="audio_master",
        schema=QAReport,
        model_tier="standard",
    )
    assert isinstance(audio_obj, QAReport)
    reports.append(audio_obj.model_copy(update={"scene_index": None}))
    cost_entries.append(
        CostEntry(at=datetime.now(UTC), item="critic:audio", provider="llm", cost_usd=audio_cost)
    )

    return {
        "qa_reports": reports,
        "retry_counts": retry_counts,
        "cost_ledger": cost_entries,
        "status": ProjectStatus.QA,
    }
