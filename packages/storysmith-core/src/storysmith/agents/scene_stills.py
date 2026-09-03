from __future__ import annotations

import asyncio
import io
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from PIL import Image

from storysmith.errors import ContentRejectedError
from storysmith.models import (
    AssetKind,
    AssetRef,
    CostEntry,
    FailureLayer,
    QAVerdict,
    Scene,
    SceneGenMode,
    StyleContract,
    VideoProject,
)
from storysmith.settings import Settings
from storysmith.util.hashing import sha256_bytes, sha256_hex

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


def _stitch_horizontally(images: list[bytes]) -> bytes:
    """Composites multiple character reference sheets into one image, side
    by side -- a documented workaround for single-image-conditioned editors
    (flux-kontext-pro included) that take exactly one `input_image`: stitch
    several references into one "reference grid" instead of trying to pass
    several images in one call. Each crop is resized to the shortest
    image's height first so the grid isn't lopsided."""
    frames = [Image.open(io.BytesIO(data)).convert("RGB") for data in images]
    target_height = min(frame.height for frame in frames)
    resized = [
        frame.resize((round(frame.width * target_height / frame.height), target_height))
        for frame in frames
    ]
    canvas = Image.new("RGB", (sum(frame.width for frame in resized), target_height), "white")
    x = 0
    for frame in resized:
        canvas.paste(frame, (x, 0))
        x += frame.width
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


async def _build_reference_image(style: StyleContract, ports: PortBundle) -> bytes | None:
    """Fetches every character's frozen reference sheet and composites them
    into one image to condition scene generation on (Amendment 03) -- None
    (falling back to today's pure text-to-image path) if no character has a
    frozen reference yet."""
    images = [
        await ports.storage.get(uri=character.image_uri)
        for character in style.characters
        if character.image_uri
    ]
    if not images:
        return None
    if len(images) == 1:
        return images[0]
    return _stitch_horizontally(images)


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
    reference_image: bytes | None,
) -> tuple[AssetRef | None, CostEntry | None]:
    assert state.style is not None and scene.scene_image_prompt is not None
    prior_attempts = sum(
        1 for a in state.assets if a.kind == AssetKind.SCENE_STILL and a.scene_index == scene.index
    )
    attempt = prior_attempts + 1

    # Character identity is still folded into the prompt text too (Director
    # already writes full descriptions in here, see director.py's
    # restatement check) -- redundant with the reference image below, but
    # cheap insurance and still the *only* signal when no reference exists
    # yet (a fresh show mid-episode, before char_refs has run).
    prompt = f"{state.style.art_style}, {scene.scene_image_prompt}"
    critique = _latest_composition_critique(state, scene.index)
    if critique:
        prompt = f"{prompt}\nAVOID THE FOLLOWING ISSUES: {critique}"

    model_id = settings.scene_image_model if reference_image is not None else settings.image_model
    ref_hash = sha256_bytes(reference_image) if reference_image is not None else ""
    content_hash = sha256_hex(model_id, prompt, ref_hash)
    if any(a.content_hash == content_hash for a in state.assets):
        return None, None  # idempotent skip: identical request already produced this still

    try:
        async with semaphore:
            image_bytes, cost = await ports.image_gen.generate(
                prompt=prompt,
                aspect_ratio=state.style.aspect_ratio,
                reference_image=reference_image,
            )
    except ContentRejectedError as exc:
        # NOT retried (§3.1) -- bubbles to Critic as an auto-fail instead of
        # crashing the whole run. No bytes were ever generated, so there's
        # nothing to store; the poison marker (no real uri) tells Critic to
        # emit HUMAN_REVIEW for this scene without trying to fetch/score it.
        return (
            AssetRef(
                kind=AssetKind.SCENE_STILL,
                scene_index=scene.index,
                attempt=attempt,
                uri="",
                content_hash=content_hash,
                cost_usd=0.0,
                meta={"content_rejected": "true", "rejection_reason": str(exc)[:500]},
            ),
            None,
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
    assert state.style is not None
    scenes = _scenes_needing_stills(state)
    # Built once per run (not per scene): the same frozen cast conditions
    # every scene this pass, and fetching/compositing it is real work
    # (storage round-trips + image processing) not worth repeating per scene.
    reference_image = await _build_reference_image(state.style, ports)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_GENERATIONS)
    results = await asyncio.gather(
        *(
            _generate_one(
                scene,
                state=state,
                ports=ports,
                settings=settings,
                semaphore=semaphore,
                reference_image=reference_image,
            )
            for scene in scenes
        )
    )
    assets = [asset for asset, _ in results if asset is not None]
    cost_entries = [entry for _, entry in results if entry is not None]
    # owns_status=False in graph/build.py (runs parallel to music_director,
    # same as videographer used to before this amendment) -- no "status" key.
    return {"assets": assets, "cost_ledger": cost_entries}
