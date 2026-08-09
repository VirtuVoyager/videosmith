from __future__ import annotations

import json

import pytest
from storysmith.agents import music_director
from storysmith.models import AssetKind, Mode, MusicCue, Scene, SceneManifest, VideoProject
from storysmith.pipeline import PortBundle
from storysmith.settings import Settings
from storysmith_adapters.stubs import (
    StubImageGen,
    StubLLM,
    StubMusicGen,
    StubNotify,
    StubPublish,
    StubStorage,
    StubTranscribe,
    StubTTS,
    StubVideoGen,
)

pytestmark = pytest.mark.wp4


def _ports(storage: StubStorage) -> PortBundle:
    return PortBundle(
        llm=StubLLM(),
        image_gen=StubImageGen(),
        video_gen=StubVideoGen(),
        music_gen=StubMusicGen(),
        tts=StubTTS(),
        transcribe=StubTranscribe(),
        storage=storage,
        publish=StubPublish(),
        notify=StubNotify(),
    )


async def test_rhyme_mode_produces_one_audio_master(settings_test: Settings) -> None:
    manifest = SceneManifest(
        title="t",
        description="d",
        tags=[],
        total_duration_s=40.0,
        lyrics="one two three four five",
        music_cues=[MusicCue(description="intro", start_s=0, end_s=4)],
        scenes=[Scene(index=0, duration_s=8, video_prompt="p", narration="one two")],
    )
    state = VideoProject(
        project_id="wp4-rhyme", mode=Mode.RHYME, brief="counting ducks", manifest=manifest
    )

    result = await music_director.run(state, ports=_ports(StubStorage()), settings=settings_test)

    audio_assets = [a for a in result["assets"] if a.kind == AssetKind.AUDIO_MASTER]
    assert len(audio_assets) == 1
    assert audio_assets[0].scene_index is None
    assert len(result["cost_ledger"]) == 1
    assert "status" not in result  # parallel with videographer -- must not race on state["status"]


async def test_topical_mode_produces_narration_assets_bed_and_timing_map(
    settings_test: Settings,
) -> None:
    scenes = [
        Scene(index=i, duration_s=5, video_prompt="p", narration=f"line {i}") for i in range(3)
    ]
    scenes.append(Scene(index=3, duration_s=5, video_prompt="p", narration=""))  # silent scene
    manifest = SceneManifest(
        title="t",
        description="d",
        tags=[],
        total_duration_s=20.0,
        lyrics=None,
        music_cues=[MusicCue(description="bed", start_s=0, end_s=20)],
        scenes=scenes,
    )
    state = VideoProject(
        project_id="wp4-topical", mode=Mode.TOPICAL, brief="sharing toys", manifest=manifest
    )
    storage = StubStorage()

    result = await music_director.run(state, ports=_ports(storage), settings=settings_test)

    audio_assets = [a for a in result["assets"] if a.kind == AssetKind.AUDIO_MASTER]
    bed_assets = [a for a in audio_assets if a.meta.get("role") == "bed"]
    narration_assets = [a for a in audio_assets if a.meta.get("role") == "narration"]

    assert len(bed_assets) == 1
    assert len(narration_assets) == 3  # one per scene with non-empty narration
    assert {a.scene_index for a in narration_assets} == {0, 1, 2}

    timing_map_uri = bed_assets[0].meta["timing_map_uri"]
    timing_map = json.loads(await storage.get(uri=timing_map_uri))
    assert {entry["scene_index"] for entry in timing_map} == {0, 1, 2}
    assert "status" not in result
