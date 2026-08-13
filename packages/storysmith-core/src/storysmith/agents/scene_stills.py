from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from storysmith.models import (
    AssetKind,
    AssetRef,
    CostEntry,
    FailureLayer,
    QAVerdict,
    Scene,
    SceneGenMode,
    VideoProject,
)
from storysmith.settings import Settings
from storysmith.util.hashing import sha256_hex

if TYPE_CHECKING:
    from storysmith.pipeline import PortBundle

_MAX_CONCURRENT_GENERATIONS = 3


def _scenes_needing_stills(state: VideoProject) -> list[Scene]:
    """Amendment 01 §5: mirrors videographer._scenes_to_generate -- all i2v
    scenes on first pass, only composition-flagged retries afterward."""
    assert state.manifest is not None
    i2v_scenes = [s for s in state.manifest.scenes if s.gen_mode == SceneGenMode.I2V]
    already_generated = {a.scene_index for a in state.assets if a.kind == AssetKind.SCENE_STILL}
    if not already_generated:
        return i2v_scenes
    composition_retry_indices = {
        r.scene_index
        for r in state.qa_reports
        if r.scene_index is not None
        and r.verdict == QAVerdict.RETRY
        and r.failure_layer == FailureLayer.COMPOSITION
    }
    return [s for s in i2v_scenes if s.index in composition_retry_indices]


def _latest_composition_critique(state: VideoProject, scene_index: int) -> str | None:
    for report in reversed(state.qa_reports):
        if (
            report.scene_index == scene_index
            and report.failure_layer == FailureLayer.COMPOSITION
            and report.critique
        ):
            return report.critique
    return None


async def _generate_one(
    scene: Scene,
    *,
    state: VideoProject,
    ports: PortBundle,
    settings: Settings,
    semaphore: asyncio.Semaphore,
) -> tuple[AssetRef | None, CostEntry | None]:
    assert state.style is not None and scene.scene_image_prompt is not None
    prior_attempts = sum(
        1 for a in state.assets if a.kind == AssetKind.SCENE_STILL and a.scene_index == scene.index
    )
    attempt = prior_attempts + 1

    # ImageGenPort is text-only for the adapter currently wired in (see
    # settings.scene_image_model's SPEC-GAP) -- so character identity is
    # folded into the prompt text rather than passed as a conditioning image.
    prompt = f"{state.style.art_style}, {scene.scene_image_prompt}"
    critique = _latest_composition_critique(state, scene.index)
    if critique:
        prompt = f"{prompt}\nAVOID THE FOLLOWING ISSUES: {critique}"

    content_hash = sha256_hex(settings.image_model, prompt)
    if any(a.content_hash == content_hash for a in state.assets):
        return None, None  # idempotent skip: identical request already produced this still

    async with semaphore:
        image_bytes, cost = await ports.image_gen.generate(
            prompt=prompt, aspect_ratio=state.style.aspect_ratio
        )
    uri = await ports.storage.put(
        key=f"{state.project_id}/scene_{scene.index}/still_attempt_{attempt}.png",
        data=image_bytes,
        content_type="image/png",
    )
    asset = AssetRef(
        kind=AssetKind.SCENE_STILL,
        scene_index=scene.index,
        attempt=attempt,
        uri=uri,
        content_hash=content_hash,
        cost_usd=cost,
    )
    cost_entry = CostEntry(
        at=datetime.now(UTC),
        item=f"scene_still:scene{scene.index}:attempt{attempt}",
        provider="image_gen",
        cost_usd=cost,
    )
    return asset, cost_entry


async def run(state: VideoProject, *, ports: PortBundle, settings: Settings) -> dict[str, Any]:
    scenes = _scenes_needing_stills(state)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_GENERATIONS)
    results = await asyncio.gather(
        *(
            _generate_one(scene, state=state, ports=ports, settings=settings, semaphore=semaphore)
            for scene in scenes
        )
    )
    assets = [asset for asset, _ in results if asset is not None]
    cost_entries = [entry for _, entry in results if entry is not None]
    # owns_status=False in graph/build.py (runs parallel to music_director,
    # same as videographer used to before this amendment) -- no "status" key.
    return {"assets": assets, "cost_ledger": cost_entries}
