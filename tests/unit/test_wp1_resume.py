from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel
from storysmith.models import AssetKind, Mode, ProjectStatus
from storysmith.pipeline import Pipeline, PortBundle
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

pytestmark = pytest.mark.wp1


class _CountingLLM(StubLLM):
    def __init__(self) -> None:
        self.calls = 0

    async def complete_structured(self, **kwargs: Any) -> tuple[BaseModel, float]:
        self.calls += 1
        return await super().complete_structured(**kwargs)


class _CountingImageGen(StubImageGen):
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, **kwargs: Any) -> tuple[bytes, float]:
        self.calls += 1
        return await super().generate(**kwargs)


class _FlakyVideoGen(StubVideoGen):
    """Raises once for the scene whose prompt contains fail_marker, then succeeds."""

    def __init__(self, fail_marker: str) -> None:
        self.calls = 0
        self._fail_marker = fail_marker
        self._failed_once = False

    async def generate(self, **kwargs: Any) -> tuple[bytes, float]:
        self.calls += 1
        if self._fail_marker in kwargs["prompt"] and not self._failed_once:
            self._failed_once = True
            raise RuntimeError("injected transient failure")
        return await super().generate(**kwargs)


async def test_resume_from_checkpoint(settings_test: Settings) -> None:
    llm = _CountingLLM()
    image_gen = _CountingImageGen()
    video_gen = _FlakyVideoGen(fail_marker="scene 2")
    ports = PortBundle(
        llm=llm,
        image_gen=image_gen,
        video_gen=video_gen,
        music_gen=StubMusicGen(),
        tts=StubTTS(),
        transcribe=StubTranscribe(),
        storage=StubStorage(),
        publish=StubPublish(),
        notify=StubNotify(),
    )
    pipeline = Pipeline(settings=settings_test, ports=ports)

    with pytest.raises(RuntimeError):
        await pipeline.run(brief="counting ducks", mode=Mode.RHYME, project_id="wp1-resume")

    # creative_director + director each call the LLM once; char_refs calls
    # image_gen once (one character), then scene_stills calls it once more
    # per i2v scene (5, all gen_mode defaults to i2v -- Amendment 01). These
    # nodes ran to completion and were checkpointed before videographer's
    # internal failure.
    llm_calls_before_resume = llm.calls
    image_calls_before_resume = image_gen.calls
    assert llm_calls_before_resume == 2
    assert image_calls_before_resume == 6

    result = await pipeline.run(brief="counting ducks", mode=Mode.RHYME, project_id="wp1-resume")

    # The expensive, already-completed early stages are not re-invoked on
    # resume -- LangGraph resumes from the last successful checkpoint, so
    # creative_director/director/char_refs never run a second time.
    assert image_gen.calls == image_calls_before_resume
    assert llm.calls > llm_calls_before_resume  # critic calls happen after resuming

    assert result.status == ProjectStatus.REVIEW
    scene_videos = [a for a in result.assets if a.kind == AssetKind.SCENE_VIDEO]
    assert len(scene_videos) == 5
