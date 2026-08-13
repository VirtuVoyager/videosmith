from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel
from storysmith.agents import director, scene_stills, videographer
from storysmith.graph.build import _critic_router
from storysmith.models import (
    AssetKind,
    AssetRef,
    CharacterRef,
    FailureLayer,
    Mode,
    MusicCue,
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
from storysmith_adapters.stubs import (
    StubLLM,
    StubMusicGen,
    StubNotify,
    StubPublish,
    StubStorage,
    StubTranscribe,
    StubTTS,
    StubVideoGen,
)

pytestmark = pytest.mark.amendment01


def _style() -> StyleContract:
    return StyleContract(
        art_style="soft 2D cutout animation",
        palette=["#FFFFFF"],
        mood="cheerful",
        tempo_bpm=100,
        characters=[CharacterRef(name="Ducky", description="a yellow duck")],
        pacing_rules="short cuts",
        negative_terms=[],
    )


def _manifest(scenes: list[Scene]) -> SceneManifest:
    return SceneManifest(
        title="t",
        description="d",
        tags=[],
        total_duration_s=sum(s.duration_s for s in scenes),
        lyrics="l",
        music_cues=[MusicCue(description="x", start_s=0, end_s=1)],
        scenes=scenes,
    )


class _ScriptedLLM:
    def __init__(self, responses: list[BaseModel]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete_structured(self, **kwargs: Any) -> tuple[BaseModel, float]:
        obj = self._responses[self.calls]
        self.calls += 1
        return obj, 0.001


class _CountingImageGen:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    async def generate(self, *, prompt: str, aspect_ratio: str) -> tuple[bytes, float]:
        self.calls += 1
        self.prompts.append(prompt)
        return b"STILL_BYTES", 0.003


class _CapturingVideoGen:
    def __init__(self) -> None:
        self.last_reference_image: bytes | None = None

    async def generate(self, **kwargs: Any) -> tuple[bytes, float]:
        self.last_reference_image = kwargs["reference_image"]
        return b"VIDEO", 0.01


def _ports(*, llm: Any = None, image_gen: Any = None, video_gen: Any = None) -> PortBundle:
    from storysmith_adapters.stubs import StubImageGen

    return PortBundle(
        llm=llm or StubLLM(),
        image_gen=image_gen or StubImageGen(),
        video_gen=video_gen or StubVideoGen(),
        music_gen=StubMusicGen(),
        tts=StubTTS(),
        transcribe=StubTranscribe(),
        storage=StubStorage(),
        publish=StubPublish(),
        notify=StubNotify(),
    )


def _filler_scenes(count: int, *, start_index: int) -> list[Scene]:
    """Valid t2v filler scenes -- lets a test isolate one scene's violation
    without also tripping the (unrelated) scene-count/duration checks."""
    return [
        Scene(
            index=start_index + i,
            duration_s=8,
            video_prompt=f"soft drifting clouds in the sky, scene {start_index + i}",
            narration="",
            gen_mode=SceneGenMode.T2V,
        )
        for i in range(count)
    ]


async def test_i2v_scene_requires_image_prompt(settings_test: Settings) -> None:
    bad_scene = Scene(
        index=0,
        duration_s=8,
        video_prompt="soft style, the duck bobs its head, camera remains static",
        narration="hi",
        gen_mode=SceneGenMode.I2V,
        scene_image_prompt=None,  # violation: i2v with no still prompt
    )
    good_scene = bad_scene.model_copy(
        update={"scene_image_prompt": "soft 2D cutout animation duck centered, plain background"}
    )
    filler = _filler_scenes(4, start_index=1)
    bad = _manifest([bad_scene, *filler])
    good = _manifest([good_scene, *filler])
    llm = _ScriptedLLM([bad, good])
    state = VideoProject(project_id="a01-1", mode=Mode.RHYME, brief="b", style=_style())

    result = await director.run(state, ports=_ports(llm=llm), settings=settings_test)

    assert llm.calls == 2
    assert result["manifest"] == good


async def test_t2v_scene_skips_still_generation(settings_test: Settings) -> None:
    manifest = _manifest(
        [
            Scene(
                index=i,
                duration_s=5,
                video_prompt="sky drifting clouds",
                narration="",
                gen_mode=SceneGenMode.T2V,
            )
            for i in range(4)
        ]
    )
    state = VideoProject(
        project_id="a01-2", mode=Mode.RHYME, brief="b", style=_style(), manifest=manifest
    )
    image_gen = _CountingImageGen()

    result = await scene_stills.run(
        state, ports=_ports(image_gen=image_gen), settings=settings_test
    )

    assert image_gen.calls == 0
    assert result["assets"] == []


async def test_videographer_uses_scene_still_not_char_ref_for_i2v(settings_test: Settings) -> None:
    scene = Scene(
        index=0,
        duration_s=8,
        video_prompt="soft 2D cutout animation duck bobs its head",
        narration="",
        gen_mode=SceneGenMode.I2V,
        scene_image_prompt="soft 2D cutout animation duck centered, plain background",
    )
    manifest = _manifest([scene])
    storage = StubStorage()
    char_uri = await storage.put(key="char.png", data=b"CHAR_REF_BYTES", content_type="image/png")
    still_uri = await storage.put(
        key="still.png", data=b"SCENE_STILL_BYTES", content_type="image/png"
    )
    style = _style().model_copy(
        update={
            "characters": [
                CharacterRef(name="Ducky", description="a yellow duck", image_uri=char_uri)
            ]
        }
    )
    still_asset = AssetRef(
        kind=AssetKind.SCENE_STILL, scene_index=0, attempt=1, uri=still_uri, content_hash="h1"
    )
    state = VideoProject(
        project_id="a01-3",
        mode=Mode.RHYME,
        brief="b",
        style=style,
        manifest=manifest,
        assets=[still_asset],
    )
    video_gen = _CapturingVideoGen()
    ports = _ports(video_gen=video_gen)
    ports.storage = storage  # reuse the storage instance that actually has the bytes

    await videographer.run(state, ports=ports, settings=settings_test)

    assert video_gen.last_reference_image == b"SCENE_STILL_BYTES"


def _report(scene_index: int, verdict: QAVerdict, failure_layer: FailureLayer) -> QAReport:
    return QAReport(
        scene_index=scene_index,
        verdict=verdict,
        scores={},
        safety_flags=[],
        critique="",
        failure_layer=failure_layer,
    )


def _state(reports: list[QAReport]) -> VideoProject:
    return VideoProject(project_id="a01-4", mode=Mode.RHYME, brief="b", qa_reports=reports)


def test_critic_composition_failure_routes_to_scene_stills() -> None:
    state = _state([_report(0, QAVerdict.RETRY, FailureLayer.COMPOSITION)])
    assert _critic_router(state) == ["scene_stills"]


def test_critic_motion_failure_routes_straight_to_videographer() -> None:
    state = _state([_report(0, QAVerdict.RETRY, FailureLayer.MOTION)])
    assert _critic_router(state) == ["videographer"]


def test_critic_other_failure_routes_straight_to_videographer() -> None:
    # backward-compat default for pre-Amendment-01 QA reports
    state = _state([_report(0, QAVerdict.RETRY, FailureLayer.OTHER)])
    assert _critic_router(state) == ["videographer"]


async def test_video_prompt_motion_only_flags_layout_overlap(settings_test: Settings) -> None:
    image_prompt = "soft 2D cutout animation, Ducky the yellow duck centered, plain background"
    # video_prompt redundantly repeats the still's layout words instead of
    # describing only motion -- this must trigger the corrective round.
    bad_scene = Scene(
        index=0,
        duration_s=8,
        video_prompt="soft 2D cutout animation, Ducky the yellow duck centered, plain background",
        narration="hi",
        gen_mode=SceneGenMode.I2V,
        scene_image_prompt=image_prompt,
    )
    good_scene = bad_scene.model_copy(
        update={"video_prompt": "soft style, the duck bobs its head, camera remains static"}
    )
    filler = _filler_scenes(4, start_index=1)
    bad = _manifest([bad_scene, *filler])
    good = _manifest([good_scene, *filler])
    llm = _ScriptedLLM([bad, good])
    state = VideoProject(project_id="a01-5", mode=Mode.RHYME, brief="b", style=_style())

    result = await director.run(state, ports=_ports(llm=llm), settings=settings_test)

    assert llm.calls == 2
    assert result["manifest"] == good
