from __future__ import annotations

from typing import Any

import pytest
from storysmith.agents import videographer
from storysmith.models import (
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

pytestmark = pytest.mark.wp6


class _CapturingVideoGen:
    def __init__(self) -> None:
        self.last_prompt: str | None = None

    async def generate(self, **kwargs: Any) -> tuple[bytes, float]:
        self.last_prompt = kwargs["prompt"]
        return b"VIDEO", 0.01


async def test_critique_appended_into_regenerated_videographer_prompt(
    settings_test: Settings,
) -> None:
    # §3.2: on retry, videographer must append the critic's critique to the
    # original video_prompt so the regeneration actually addresses it.
    scene = Scene(index=0, duration_s=5, video_prompt="a duck swims", narration="")
    manifest = SceneManifest(
        title="t",
        description="d",
        tags=[],
        total_duration_s=5,
        lyrics=None,
        music_cues=[],
        scenes=[scene],
    )
    style = StyleContract(
        art_style="soft 2D",
        palette=["#fff"],
        mood="cheerful",
        tempo_bpm=100,
        characters=[],
        pacing_rules="short cuts",
        negative_terms=[],
    )
    qa_report = QAReport(
        scene_index=0,
        verdict=QAVerdict.RETRY,
        scores={},
        safety_flags=[],
        critique="make the duck bigger and centered",
    )
    state = VideoProject(
        project_id="wp6-critique",
        mode=Mode.RHYME,
        brief="counting ducks",
        style=style,
        manifest=manifest,
        qa_reports=[qa_report],
        retry_counts={0: 1},
    )
    video_gen = _CapturingVideoGen()
    ports = PortBundle(
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

    await videographer.run(state, ports=ports, settings=settings_test)

    assert video_gen.last_prompt is not None
    assert "make the duck bigger and centered" in video_gen.last_prompt
    assert "a duck swims" in video_gen.last_prompt
