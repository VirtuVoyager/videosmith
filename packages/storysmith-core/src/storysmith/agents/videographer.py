from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from storysmith.models import (
    AssetKind,
    AssetRef,
    CostEntry,
    ProjectStatus,
    QAVerdict,
    Scene,
    VideoProject,
)
from storysmith.settings import Settings
from storysmith.util.hashing import sha256_hex

if TYPE_CHECKING:
    from storysmith.pipeline import PortBundle

_MAX_CONCURRENT_GENERATIONS = 3


def _scenes_to_generate(state: VideoProject) -> list[Scene]:
    assert state.manifest is not None
    already_generated = {
        a.scene_index
        for a in state.assets
        if a.kind == AssetKind.SCENE_VIDEO and a.scene_index is not None
    }
    if not already_generated:
        return list(state.manifest.scenes)
    retry_indices = {
        r.scene_index
        for r in state.qa_reports
        if r.scene_index is not None and r.verdict == QAVerdict.RETRY
    }
    return [s for s in state.manifest.scenes if s.index in retry_indices]


def _latest_critique(state: VideoProject, scene_index: int) -> str | None:
    for report in reversed(state.qa_reports):
        if report.scene_index == scene_index and report.critique:
            return report.critique
    return None


async def _generate_one(
    scene: Scene,
    *,
    state: VideoProject,
    ports: PortBundle,
    semaphore: asyncio.Semaphore,
) -> tuple[AssetRef | None, CostEntry | None]:
    assert state.style is not None
    attempt = state.retry_counts.get(scene.index, 0) + 1
    prompt = scene.video_prompt
    critique = _latest_critique(state, scene.index)
    if critique:
        prompt = f"{prompt}\nAVOID THE FOLLOWING ISSUES: {critique}"

    content_hash = sha256_hex(
        "video-model",  # SPEC-GAP: real model id string comes from settings in WP3
        prompt,
        str(scene.duration_s),
    )
    if any(a.content_hash == content_hash for a in state.assets):
        return None, None  # idempotent skip: identical request already produced this asset

    async with semaphore:
        video_bytes, cost = await ports.video_gen.generate(
            prompt=prompt,
            duration_s=scene.duration_s,
            aspect_ratio=state.style.aspect_ratio,
            reference_image=None,
        )
    uri = await ports.storage.put(
        key=f"{state.project_id}/scene_{scene.index}/attempt_{attempt}.mp4",
        data=video_bytes,
        content_type="video/mp4",
    )
    asset = AssetRef(
        kind=AssetKind.SCENE_VIDEO,
        scene_index=scene.index,
        attempt=attempt,
        uri=uri,
        content_hash=content_hash,
        cost_usd=cost,
    )
    cost_entry = CostEntry(
        at=datetime.now(UTC),
        item=f"video:scene{scene.index}:attempt{attempt}",
        provider="video_gen",
        cost_usd=cost,
    )
    return asset, cost_entry


async def run(state: VideoProject, *, ports: PortBundle, settings: Settings) -> dict[str, Any]:
    scenes = _scenes_to_generate(state)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_GENERATIONS)
    results = await asyncio.gather(
        *(_generate_one(scene, state=state, ports=ports, semaphore=semaphore) for scene in scenes)
    )
    assets = [asset for asset, _ in results if asset is not None]
    cost_entries = [entry for _, entry in results if entry is not None]
    return {
        "assets": assets,
        "cost_ledger": cost_entries,
        "status": ProjectStatus.GENERATING,
    }
