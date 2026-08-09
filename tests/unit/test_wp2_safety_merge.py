from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel
from storysmith.agents import creative_director
from storysmith.models import CharacterRef, Mode, StyleContract, VideoProject
from storysmith.pipeline import PortBundle
from storysmith.settings import Settings
from storysmith.util.configs import load_safety_negative_terms
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


def _style_without_safety_terms() -> StyleContract:
    return StyleContract(
        art_style="soft 2D cutout animation",
        palette=["#FFFFFF"],
        mood="cheerful",
        tempo_bpm=100,
        characters=[CharacterRef(name="Ducky", description="a yellow duck")],
        pacing_rules="short cuts",
        negative_terms=["brief-specific term"],
    )


class _FixedLLM:
    def __init__(self, response: BaseModel) -> None:
        self._response = response

    async def complete_structured(self, **kwargs: Any) -> tuple[BaseModel, float]:
        return self._response, 0.001


def _ports(llm: _FixedLLM) -> PortBundle:
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


async def test_negative_terms_merged_from_safety_yaml(settings_test: Settings) -> None:
    llm = _FixedLLM(_style_without_safety_terms())
    state = VideoProject(project_id="wp2-safety", mode=Mode.RHYME, brief="counting ducks")

    result = await creative_director.run(state, ports=_ports(llm), settings=settings_test)

    base_terms = load_safety_negative_terms(settings_test.configs_dir)
    assert base_terms  # the yaml actually has entries
    merged = result["style"].negative_terms
    assert set(base_terms) <= set(merged)
    assert "brief-specific term" in merged
