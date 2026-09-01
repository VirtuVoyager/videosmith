from __future__ import annotations

import asyncio
from typing import Any

import pytest
from storysmith.agents import videographer
from storysmith.errors import ContentRejectedError
from storysmith.models import (
    AssetKind,
    AssetRef,
    CharacterRef,
    Mode,
    QAReport,
    QAVerdict,
    Scene,
    SceneGenMode,
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
    # gen_mode=t2v explicitly: this test is about a scene with no reference
    # image (Amendment 01 defaults gen_mode to i2v, which would otherwise
    # look up a scene still that doesn't exist here and still fall back to
    # None -- explicit t2v keeps the "no reference image" intent unambiguous).
    scene = Scene(
        index=0,
        duration_s=8,
        video_prompt="soft 2D duck swims",
        narration="",
        gen_mode=SceneGenMode.T2V,
    )
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


class _RejectingVideoGen:
    async def generate(self, **kwargs: Any) -> tuple[bytes, float]:
        raise ContentRejectedError("provider rejected prompt: unit-test")


async def test_regression_content_rejected_produces_poison_asset_not_crash(
    settings_test: Settings,
) -> None:
    """CLAUDE.md §3.1: a content-policy rejection must bubble to Critic as
    an auto-fail, not crash the whole pipeline run -- confirmed live (flux-
    schnell false-flagged a completely benign scene as NSFW), and nothing
    caught it before this fix: the exception propagated all the way up
    through the graph and silently killed the background run task."""
    scene = Scene(
        index=0,
        duration_s=5,
        video_prompt="soft 2D duck swims",
        narration="",
        gen_mode=SceneGenMode.T2V,
    )
    manifest = SceneManifest(
        title="t", description="d", tags=[], total_duration_s=5, music_cues=[], scenes=[scene]
    )
    state = VideoProject(
        project_id="wp3-rejected",
        mode=Mode.RHYME,
        brief="counting ducks",
        style=_style(),
        manifest=manifest,
    )

    result = await videographer.run(
        state, ports=_ports(_RejectingVideoGen()), settings=settings_test
    )

    assert result["assets"][0].meta["content_rejected"] == "true"
    assert "unit-test" in result["assets"][0].meta["rejection_reason"]
    assert result["assets"][0].uri == ""


async def test_regression_rejected_still_skips_video_generation_entirely(
    settings_test: Settings,
) -> None:
    """A scene whose start-frame still was already content-rejected
    (scene_stills.py) shouldn't silently fall back to a t2v-style call with
    no reference -- that would mask the rejection instead of surfacing it."""
    scene = Scene(
        index=0,
        duration_s=5,
        video_prompt="soft 2D duck swims",
        narration="",
        gen_mode=SceneGenMode.I2V,
    )
    manifest = SceneManifest(
        title="t", description="d", tags=[], total_duration_s=5, music_cues=[], scenes=[scene]
    )
    rejected_still = AssetRef(
        kind=AssetKind.SCENE_STILL,
        scene_index=0,
        attempt=1,
        uri="",
        content_hash="h",
        meta={"content_rejected": "true", "rejection_reason": "flagged as NSFW"},
    )
    state = VideoProject(
        project_id="wp3-still-rejected",
        mode=Mode.RHYME,
        brief="counting ducks",
        style=_style(),
        manifest=manifest,
        assets=[rejected_still],
    )
    video_gen = _CountingVideoGen()

    result = await videographer.run(state, ports=_ports(video_gen), settings=settings_test)

    assert video_gen.calls == 0  # never even attempted -- propagated immediately
    new_video_asset = next(a for a in result["assets"] if a.kind == AssetKind.SCENE_VIDEO)
    assert new_video_asset.meta["content_rejected"] == "true"
    assert "NSFW" in new_video_asset.meta["rejection_reason"]


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
