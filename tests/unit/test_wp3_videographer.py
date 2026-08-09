from __future__ import annotations

import asyncio
from typing import Any

import pytest
from storysmith.agents import videographer
from storysmith.models import (
    AssetKind,
    AssetRef,
    CharacterRef,
    Mode,
    QAReport,
    QAVerdict,
    Scene,
    SceneManifest,
    StyleContract,
    VideoProject,
)
from storysmith.pipeline import PortBundle
from storysmith.settings import Settings
from storysmith.util.hashing import sha256_hex
from storysmith_adapters.stubs import (
    StubImageGen,
    StubLLM,
    StubMusicGen,
    StubNotify,
    StubPublish,
    StubStorage,
    StubTranscribe,
    StubTTS,
)

pytestmark = pytest.mark.wp3


def _style(*, with_character: bool = False) -> StyleContract:
    characters = [CharacterRef(name="Ducky", description="a yellow duck")] if with_character else []
    return StyleContract(
        art_style="soft 2D cutout animation",
        palette=["#FFFFFF"],
        mood="cheerful",
        tempo_bpm=100,
        characters=characters,
        pacing_rules="short cuts",
        negative_terms=[],
    )


def _ports(video_gen: Any) -> PortBundle:
    return PortBundle(
        llm=StubLLM(),
        image_gen=StubImageGen(),
        video_gen=video_gen,
        music_gen=StubMusicGen(),
        tts=StubTTS(),
        transcribe=StubTranscribe(),
        storage=StubStorage(),
        publish=StubPublish(),
        notify=StubNotify(),
    )


class _CountingVideoGen:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, **kwargs: Any) -> tuple[bytes, float]:
        self.calls += 1
        return b"NEW_VIDEO", 0.01


async def test_idempotent_skip_on_same_hash(settings_test: Settings) -> None:
    scene = Scene(index=0, duration_s=8, video_prompt="soft 2D duck swims", narration="")
    manifest = SceneManifest(
        title="t",
        description="d",
        tags=[],
        total_duration_s=8,
        lyrics=None,
        music_cues=[],
        scenes=[scene],
    )
    style = _style(with_character=False)

    # Same formula videographer.py uses: no reference image -> t2v model, empty ref hash.
    expected_hash = sha256_hex(
        settings_test.video_model_t2v, scene.video_prompt, str(scene.duration_s), ""
    )
    existing_asset = AssetRef(
        kind=AssetKind.SCENE_VIDEO,
        scene_index=0,
        attempt=1,
        uri="stub://existing.mp4",
        content_hash=expected_hash,
    )
    # verdict=RETRY (so _scenes_to_generate selects scene 0 again) but with an
    # empty critique, so the regenerated prompt is byte-identical to before --
    # the exact case the content-hash idempotency check exists to catch.
    qa_report = QAReport(
        scene_index=0, verdict=QAVerdict.RETRY, scores={}, safety_flags=[], critique=""
    )

    state = VideoProject(
        project_id="wp3-idem",
        mode=Mode.RHYME,
        brief="counting ducks",
        style=style,
        manifest=manifest,
        assets=[existing_asset],
        qa_reports=[qa_report],
        retry_counts={0: 1},
    )

    video_gen = _CountingVideoGen()
    result = await videographer.run(state, ports=_ports(video_gen), settings=settings_test)

    assert video_gen.calls == 0
    assert result["assets"] == []


class _TrackingVideoGen:
    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0

    async def generate(self, **kwargs: Any) -> tuple[bytes, float]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        return b"V", 0.01


async def test_semaphore_bounds_concurrency(settings_test: Settings) -> None:
    scenes = [
        Scene(index=i, duration_s=5, video_prompt=f"soft 2D duck scene {i}", narration="")
        for i in range(5)
    ]
    manifest = SceneManifest(
        title="t",
        description="d",
        tags=[],
        total_duration_s=25,
        lyrics=None,
        music_cues=[],
        scenes=scenes,
    )
    state = VideoProject(
        project_id="wp3-sem",
        mode=Mode.RHYME,
        brief="counting ducks",
        style=_style(with_character=False),
        manifest=manifest,
    )

    video_gen = _TrackingVideoGen()
    await videographer.run(state, ports=_ports(video_gen), settings=settings_test)

    assert video_gen.max_in_flight <= 3
