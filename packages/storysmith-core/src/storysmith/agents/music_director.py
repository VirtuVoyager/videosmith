from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from storysmith.models import AssetKind, AssetRef, CostEntry, Mode, VideoProject
from storysmith.settings import Settings
from storysmith.util.hashing import sha256_bytes

if TYPE_CHECKING:
    from storysmith.pipeline import PortBundle

_NARRATION_GAP_S = 0.3
# Sentinel key in retry_counts for the audio track (mirrors critic.py's
# _AUDIO_RETRY_KEY) -- not a scene index, since audio has none.
_AUDIO_RETRY_KEY = -1


async def _run_rhyme(state: VideoProject, *, ports: PortBundle) -> dict[str, Any]:
    assert state.manifest is not None
    attempt = state.retry_counts.get(_AUDIO_RETRY_KEY, 0) + 1
    mood = state.style.mood if state.style is not None else ""
    audio, cost = await ports.music_gen.generate(
        mode=Mode.RHYME,
        lyrics=state.manifest.lyrics,
        description=mood,
        duration_s=state.manifest.total_duration_s,
    )
    uri = await ports.storage.put(
        key=f"{state.project_id}/audio_master/attempt_{attempt}.mp3",
        data=audio,
        content_type="audio/mpeg",
    )
    asset = AssetRef(
        kind=AssetKind.AUDIO_MASTER,
        attempt=attempt,
        uri=uri,
        content_hash=sha256_bytes(audio),
        cost_usd=cost,
    )
    cost_entry = CostEntry(
        at=datetime.now(UTC), item="music:rhyme_master", provider="music_gen", cost_usd=cost
    )
    return {"assets": [asset], "cost_ledger": [cost_entry]}


async def _run_topical(
    state: VideoProject, *, ports: PortBundle, settings: Settings
) -> dict[str, Any]:
    assert state.manifest is not None
    manifest = state.manifest
    attempt = state.retry_counts.get(_AUDIO_RETRY_KEY, 0) + 1
    mood = state.style.mood if state.style is not None else "cheerful"

    bed, bed_cost = await ports.music_gen.generate(
        mode=Mode.TOPICAL,
        lyrics=None,
        description=f"instrumental bed, {mood}",
        duration_s=manifest.total_duration_s,
    )
    bed_uri = await ports.storage.put(
        key=f"{state.project_id}/audio_bed/attempt_{attempt}.mp3",
        data=bed,
        content_type="audio/mpeg",
    )
    assets = [
        AssetRef(
            kind=AssetKind.AUDIO_MASTER,
            attempt=attempt,
            uri=bed_uri,
            content_hash=sha256_bytes(bed),
            cost_usd=bed_cost,
            meta={"role": "bed"},
        )
    ]
    cost_entries = [
        CostEntry(
            at=datetime.now(UTC), item="music:topical_bed", provider="music_gen", cost_usd=bed_cost
        )
    ]

    timing_map: list[dict[str, float | int]] = []
    offset = 0.0
    for scene in manifest.scenes:
        if not scene.narration:
            offset += scene.duration_s + _NARRATION_GAP_S
            continue
        speech, tts_cost = await ports.tts.speak(text=scene.narration, voice=settings.tts_voice)
        n_uri = await ports.storage.put(
            key=f"{state.project_id}/narration_{scene.index}/attempt_{attempt}.mp3",
            data=speech,
            content_type="audio/mpeg",
        )
        assets.append(
            AssetRef(
                kind=AssetKind.AUDIO_MASTER,
                scene_index=scene.index,
                attempt=attempt,
                uri=n_uri,
                content_hash=sha256_bytes(speech),
                cost_usd=tts_cost,
                meta={"role": "narration"},
            )
        )
        cost_entries.append(
            CostEntry(
                at=datetime.now(UTC),
                item=f"tts:scene{scene.index}",
                provider="tts",
                cost_usd=tts_cost,
            )
        )
        timing_map.append({"scene_index": scene.index, "start_s": offset})
        offset += scene.duration_s + _NARRATION_GAP_S

    timing_bytes = json.dumps(timing_map).encode("utf-8")
    timing_uri = await ports.storage.put(
        key=f"{state.project_id}/timing_map.json",
        data=timing_bytes,
        content_type="application/json",
    )
    assets[0] = assets[0].model_copy(
        update={"meta": {**assets[0].meta, "timing_map_uri": timing_uri}}
    )

    return {"assets": assets, "cost_ledger": cost_entries}


async def run(state: VideoProject, *, ports: PortBundle, settings: Settings) -> dict[str, Any]:
    # NOTE: status is intentionally omitted from the return value — this node
    # runs in parallel with videographer (both branch off char_refs and join
    # at critic); only videographer writes "status" so the two branches never
    # race on the same state key in one LangGraph superstep.
    if state.mode == Mode.RHYME:
        return await _run_rhyme(state, ports=ports)
    return await _run_topical(state, ports=ports, settings=settings)
