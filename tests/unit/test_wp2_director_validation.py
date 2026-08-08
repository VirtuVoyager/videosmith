from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel
from storysmith.agents import director
from storysmith.errors import LLMStructuredOutputError
from storysmith.models import (
    CharacterRef,
    Mode,
    MusicCue,
    Scene,
    SceneManifest,
    StyleContract,
    VideoProject,
)
from storysmith.pipeline import PortBundle
from storysmith.settings import Settings
from storysmith_adapters.stubs import (
    StubImageGen,
    StubMusicGen,
    StubNotify,
    StubPublish,
    StubStorage,
    StubTranscribe,
    StubTTS,
    StubVideoGen,
)

pytestmark = pytest.mark.wp2


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


def _manifest(*, scene_count: int, duration_s: float, video_prompt: str) -> SceneManifest:
    scenes = [
        Scene(index=i, duration_s=duration_s, video_prompt=video_prompt, narration="hi")
        for i in range(scene_count)
    ]
    return SceneManifest(
        title="t",
        description="d",
        tags=[],
        total_duration_s=duration_s * scene_count,
        lyrics="l",
        music_cues=[MusicCue(description="x", start_s=0, end_s=1)],
        scenes=scenes,
    )


def _bad_manifest() -> SceneManifest:
    # 2 scenes (below the 4-7 requirement), 10s total (below the 30-60s window)
    return _manifest(scene_count=2, duration_s=5, video_prompt="a duck swims")


def _good_manifest() -> SceneManifest:
    return _manifest(scene_count=5, duration_s=8, video_prompt="soft 2D cutout animation duck")


class _ScriptedLLM:
    def __init__(self, responses: list[BaseModel]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete_structured(self, **kwargs: Any) -> tuple[BaseModel, float]:
        obj = self._responses[self.calls]
        self.calls += 1
        return obj, 0.001


def _ports(llm: _ScriptedLLM) -> PortBundle:
    return PortBundle(
        llm=llm,
        image_gen=StubImageGen(),
        video_gen=StubVideoGen(),
        music_gen=StubMusicGen(),
        tts=StubTTS(),
        transcribe=StubTranscribe(),
        storage=StubStorage(),
        publish=StubPublish(),
        notify=StubNotify(),
    )


def _state() -> VideoProject:
    return VideoProject(
        project_id="wp2-director", mode=Mode.RHYME, brief="counting ducks", style=_style()
    )


async def test_director_validation_rules_triggers_corrective_round(settings_test: Settings) -> None:
    llm = _ScriptedLLM([_bad_manifest(), _good_manifest()])

    result = await director.run(_state(), ports=_ports(llm), settings=settings_test)

    assert llm.calls == 2
    assert result["manifest"] == _good_manifest()
    assert len(result["cost_ledger"]) == 2


async def test_director_validation_rules_hard_fails_after_corrective_round(
    settings_test: Settings,
) -> None:
    llm = _ScriptedLLM([_bad_manifest(), _bad_manifest()])

    with pytest.raises(LLMStructuredOutputError):
        await director.run(_state(), ports=_ports(llm), settings=settings_test)

    assert llm.calls == 2
