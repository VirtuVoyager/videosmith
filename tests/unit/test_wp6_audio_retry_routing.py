from __future__ import annotations

from typing import Any

import pytest
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

pytestmark = pytest.mark.wp6


class _CountingVideoGen:
    def __init__(self) -> None:
        self._inner = StubVideoGen()
        self.calls = 0

    async def generate(self, **kwargs: Any) -> tuple[bytes, float]:
        self.calls += 1
        return await self._inner.generate(**kwargs)


class _FlakyMusicGen:
    """Returns mismatched lyrics on the first call (fails WER), correct
    lyrics on every call after (passes) -- so exactly one audio RETRY cycle
    happens, and we can observe whether it wrongly re-triggers video_gen."""

    def __init__(self) -> None:
        self._inner = StubMusicGen()
        self.calls = 0

    async def generate(self, **kwargs: Any) -> tuple[bytes, float]:
        self.calls += 1
        if self.calls == 1:
            kwargs = {**kwargs, "lyrics": "completely mismatched wrong lyrics text"}
        return await self._inner.generate(**kwargs)


async def test_audio_only_retry_does_not_regenerate_scenes(settings_test: Settings) -> None:
    video_gen = _CountingVideoGen()
    music_gen = _FlakyMusicGen()
    ports = PortBundle(
        llm=StubLLM(),
        image_gen=StubImageGen(),
        video_gen=video_gen,
        music_gen=music_gen,
        tts=StubTTS(),
        transcribe=StubTranscribe(),
        storage=StubStorage(),
        publish=StubPublish(),
        notify=StubNotify(),
    )
    pipeline = Pipeline(settings=settings_test, ports=ports)

    result = await pipeline.run(
        brief="counting ducks", mode=Mode.RHYME, project_id="wp6-audio-retry"
    )

    assert music_gen.calls == 2  # one failed attempt, one corrective retry
    assert video_gen.calls == 5  # exactly once per scene -- never re-triggered by the audio retry
    assert result.status == ProjectStatus.REVIEW
    assert result.retry_counts.get(-1) == 1
    # no scene ever needed a retry, so no scene keys should appear
    assert all(key == -1 for key in result.retry_counts)
    kinds = {a.kind for a in result.assets}
    assert AssetKind.FINAL_VIDEO in kinds  # audio eventually passed, editor still ran
