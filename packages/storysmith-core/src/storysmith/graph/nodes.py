from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from storysmith.agents import creative_director as creative_director_agent
from storysmith.agents import critic as critic_agent
from storysmith.agents import director as director_agent
from storysmith.agents import editor as editor_agent
from storysmith.agents import music_director as music_director_agent
from storysmith.agents import publisher as publisher_agent
from storysmith.agents import scene_stills as scene_stills_agent
from storysmith.agents import videographer as videographer_agent
from storysmith.models import (
    AssetKind,
    AssetRef,
    CostEntry,
    ProjectStatus,
    QAVerdict,
    VideoProject,
)
from storysmith.settings import Settings
from storysmith.util.hashing import sha256_bytes

if TYPE_CHECKING:
    from storysmith.pipeline import PortBundle

# char_refs and review_gate have no dedicated agents/*.py file — §0.1's file
# tree only lists creative_director/director/videographer/music_director/
# critic/editor/publisher under agents/, so this thin-node-delegates-to-agent
# pattern doesn't apply to those two; their (small) logic lives here directly.


async def creative_director(
    state: VideoProject, *, ports: PortBundle, settings: Settings
) -> dict[str, Any]:
    return await creative_director_agent.run(state, ports=ports, settings=settings)


async def director(state: VideoProject, *, ports: PortBundle, settings: Settings) -> dict[str, Any]:
    return await director_agent.run(state, ports=ports, settings=settings)


async def char_refs(
    state: VideoProject, *, ports: PortBundle, settings: Settings
) -> dict[str, Any]:
    assert state.style is not None
    style = state.style

    async def _generate_one(index: int) -> tuple[AssetRef, CostEntry, str]:
        character = style.characters[index]
        prompt = (
            f"{character.description}, {style.art_style}, character reference sheet, "
            "full body, neutral pose, plain background"
        )
        image_bytes, cost = await ports.image_gen.generate(
            prompt=prompt, aspect_ratio=style.aspect_ratio
        )
        uri = await ports.storage.put(
            key=f"{state.project_id}/char_{character.name}.png",
            data=image_bytes,
            content_type="image/png",
        )
        asset = AssetRef(
            kind=AssetKind.CHAR_IMAGE,
            uri=uri,
            content_hash=sha256_bytes(image_bytes),
            cost_usd=cost,
        )
        cost_entry = CostEntry(
            at=datetime.now(UTC),
            item=f"char_refs:{character.name}",
            provider="image_gen",
            cost_usd=cost,
        )
        return asset, cost_entry, uri

    results = await asyncio.gather(*(_generate_one(i) for i in range(len(style.characters))))
    updated_characters = [
        char.model_copy(update={"image_uri": uri})
        for char, (_, _, uri) in zip(style.characters, results, strict=True)
    ]
    updated_style = style.model_copy(update={"characters": updated_characters})
    return {
        "style": updated_style,
        "assets": [asset for asset, _, _ in results],
        "cost_ledger": [entry for _, entry, _ in results],
    }


async def scene_stills(
    state: VideoProject, *, ports: PortBundle, settings: Settings
) -> dict[str, Any]:
    return await scene_stills_agent.run(state, ports=ports, settings=settings)


async def videographer(
    state: VideoProject, *, ports: PortBundle, settings: Settings
) -> dict[str, Any]:
    return await videographer_agent.run(state, ports=ports, settings=settings)


async def music_director(
    state: VideoProject, *, ports: PortBundle, settings: Settings
) -> dict[str, Any]:
    return await music_director_agent.run(state, ports=ports, settings=settings)


async def critic(state: VideoProject, *, ports: PortBundle, settings: Settings) -> dict[str, Any]:
    return await critic_agent.run(state, ports=ports, settings=settings)


async def editor(state: VideoProject, *, ports: PortBundle, settings: Settings) -> dict[str, Any]:
    return await editor_agent.run(state, ports=ports, settings=settings)


def _latest_final_video(assets: list[AssetRef]) -> AssetRef | None:
    candidates = [a for a in assets if a.kind == AssetKind.FINAL_VIDEO]
    return max(candidates, key=lambda a: a.attempt, default=None)


async def review_gate(
    state: VideoProject, *, ports: PortBundle, settings: Settings
) -> dict[str, Any]:
    # §7: notify with title, cost, QA summary, presigned video URL (when a
    # final cut exists -- critic can route straight here on HUMAN_REVIEW
    # before editor ever runs, so there may only be scene-level assets), and
    # an approve/reject deep link into the UI console (which holds the
    # bearer token and calls the API -- a bare Telegram link can't).
    title = state.manifest.title if state.manifest else state.brief
    escalations = [r for r in state.qa_reports if r.verdict == QAVerdict.HUMAN_REVIEW]
    if escalations:
        flagged = ", ".join(
            "audio" if r.scene_index is None else f"scene {r.scene_index}" for r in escalations
        )
        headline = f'"{title}" needs human review -- flagged: {flagged}'
    else:
        headline = f'"{title}" passed QA, ready for review'

    final_video = _latest_final_video(state.assets)
    video_line = ""
    if final_video is not None:
        presigned = await ports.storage.presign(uri=final_video.uri)
        video_line = f"\nVideo: {presigned}"

    text = f"{headline}\nCost so far: ${state.total_cost:.2f}{video_line}"
    review_link = f"{settings.console_base_url}/p/{state.project_id}"
    await ports.notify.send(text=text, link=review_link)
    return {"status": ProjectStatus.REVIEW}


async def publisher(
    state: VideoProject, *, ports: PortBundle, settings: Settings
) -> dict[str, Any]:
    return await publisher_agent.run(state, ports=ports, settings=settings)
