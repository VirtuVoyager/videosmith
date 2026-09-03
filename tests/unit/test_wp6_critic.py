from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel
from storysmith.agents import critic
from storysmith.models import (
    AssetKind,
    AssetRef,
    CharacterRef,
    DialogueLine,
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
    StubMusicGen,
    StubNotify,
    StubPublish,
    StubStorage,
    StubTranscribe,
    StubTTS,
    StubVideoGen,
)

pytestmark = pytest.mark.wp6

_PASSING_SCORES = {
    "style_adherence": 0.9,
    "character_consistency": 0.9,
    "visual_artifacts": 0.9,
    "kid_appeal": 0.9,
    "safety": 0.9,
}
_FAILING_SCORES = {
    "style_adherence": 0.2,
    "character_consistency": 0.2,
    "visual_artifacts": 0.2,
    "kid_appeal": 0.2,
    "safety": 0.2,
}


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


def _manifest() -> SceneManifest:
    scene = Scene(index=0, duration_s=3, video_prompt="soft 2D duck", narration="")
    return SceneManifest(
        title="t",
        description="d",
        tags=[],
        total_duration_s=3.0,
        lyrics="one two three",
        music_cues=[],
        scenes=[scene],
    )


async def test_regression_score_audio_uses_dialogue_not_empty_narration(
    settings_test: Settings,
) -> None:
    """A real live run (topical, dialogue-driven show) hit this: a dialogue
    scene's `narration` is "" by design (Amendment 02 -- the spoken content
    lives in `dialogue` instead), but _score_audio's `expected` text was
    always built from `narration` alone. Comparing a real transcript against
    an empty expected string scores total mismatch, guaranteeing every
    dialogue scene gets RETRY'd into HUMAN_REVIEW regardless of actual audio
    quality -- confirmed live via review_gate's "flagged: audio" notification
    on an otherwise-clean run."""
    storage = StubStorage()
    scene = Scene(
        index=0,
        duration_s=5,
        video_prompt="p",
        narration="",
        dialogue=[
            DialogueLine(speaker="Bob", line="hello there"),
            DialogueLine(speaker="Miko", line="hi bob"),
        ],
    )
    manifest = SceneManifest(
        title="t", description="d", tags=[], total_duration_s=5.0, music_cues=[], scenes=[scene]
    )
    audio_bytes, _ = await StubTTS().speak(text="hello there hi bob", voice="am_adam")
    uri = await storage.put(key="narration0.mp3", data=audio_bytes, content_type="audio/mpeg")
    state = VideoProject(
        project_id="p1",
        mode=Mode.TOPICAL,
        brief="b",
        manifest=manifest,
        assets=[
            AssetRef(
                kind=AssetKind.AUDIO_MASTER,
                scene_index=0,
                attempt=1,
                uri=uri,
                content_hash="h",
                meta={"role": "narration"},
            )
        ],
    )
    ports = PortBundle(
        llm=_ScriptedLLM(_qa_report(scores=_PASSING_SCORES)),
        image_gen=StubImageGen(),
        video_gen=StubVideoGen(),
        music_gen=StubMusicGen(),
        tts=StubTTS(),
        transcribe=StubTranscribe(),
        storage=storage,
        publish=StubPublish(),
        notify=StubNotify(),
    )

    transcript, expected, _ = await critic._score_audio(state, ports=ports)

    assert expected == "hello there hi bob"
    assert transcript == expected


def _qa_report(
    *, scores: dict[str, float], safety_flags: list[str] | None = None, critique: str = ""
) -> QAReport:
    return QAReport(
        scene_index=None,
        verdict=QAVerdict.PASS,  # overwritten by critic.run() regardless
        scores=scores,
        safety_flags=safety_flags or [],
        critique=critique,
    )


class _ScriptedLLM:
    def __init__(self, response: QAReport) -> None:
        self._response = response
        self.calls = 0

    async def complete_structured(self, **kwargs: Any) -> tuple[BaseModel, float]:
        self.calls += 1
        return self._response, 0.001


async def _seeded_state(
    storage: StubStorage, *, retry_counts: dict[int, int] | None = None
) -> VideoProject:
    manifest = _manifest()
    style = _style()

    video_bytes, _ = await StubVideoGen().generate(
        prompt="p", duration_s=3, aspect_ratio="9:16", reference_image=None
    )
    scene_uri = await storage.put(
        key="wp6/scene_0/attempt_1.mp4", data=video_bytes, content_type="video/mp4"
    )
    scene_asset = AssetRef(
        kind=AssetKind.SCENE_VIDEO, scene_index=0, attempt=1, uri=scene_uri, content_hash="h1"
    )

    # Real WER match (StubMusicGen embeds `lyrics` for round-trip via
    # StubTranscribe) so these tests isolate the scene-verdict logic without
    # audio also failing and adding noise.
    audio_bytes, _ = await StubMusicGen().generate(
        mode=Mode.RHYME, lyrics=manifest.lyrics, description="x", duration_s=3.0
    )
    audio_uri = await storage.put(
        key="wp6/audio_master.mp3", data=audio_bytes, content_type="audio/mpeg"
    )
    audio_asset = AssetRef(kind=AssetKind.AUDIO_MASTER, attempt=1, uri=audio_uri, content_hash="h2")

    return VideoProject(
        project_id="wp6",
        mode=Mode.RHYME,
        brief="counting ducks",
        style=style,
        manifest=manifest,
        assets=[scene_asset, audio_asset],
        retry_counts=retry_counts or {},
    )


def _ports(llm: _ScriptedLLM, storage: StubStorage) -> PortBundle:
    from storysmith_adapters.stubs import StubTranscribe

    return PortBundle(
        llm=llm,
        image_gen=StubImageGen(),
        video_gen=StubVideoGen(),
        music_gen=StubMusicGen(),
        tts=StubTTS(),
        transcribe=StubTranscribe(),
        storage=storage,
        publish=StubPublish(),
        notify=StubNotify(),
    )


def _scene_report(reports: list[QAReport]) -> QAReport:
    return next(r for r in reports if r.scene_index == 0)


def _audio_report(reports: list[QAReport]) -> QAReport:
    return next(r for r in reports if r.scene_index is None)


async def test_regression_content_rejected_scene_skips_llm_and_goes_to_human_review(
    settings_test: Settings,
) -> None:
    """scene_stills.py/videographer.py mark a content-policy-rejected scene
    with a poison AssetRef (no real uri/bytes) rather than crashing the run
    (CLAUDE.md §3.1). Critic must recognize that marker and route straight
    to HUMAN_REVIEW without trying to fetch keyframes from an asset that was
    never actually generated -- confirmed live this crashed the whole run
    before this fix (KeyError on storage.get for a URI nothing ever put)."""
    storage = StubStorage()
    state = await _seeded_state(storage)
    rejected_asset = AssetRef(
        kind=AssetKind.SCENE_VIDEO,
        scene_index=0,
        attempt=1,
        uri="",
        content_hash="rejected",
        meta={"content_rejected": "true", "rejection_reason": "flagged as NSFW"},
    )
    state = state.model_copy(
        update={"assets": [a for a in state.assets if a.scene_index != 0] + [rejected_asset]}
    )
    llm = _ScriptedLLM(_qa_report(scores=_PASSING_SCORES))

    result = await critic.run(state, ports=_ports(llm, storage), settings=settings_test)

    scene_report = _scene_report(result["qa_reports"])
    assert scene_report.verdict == QAVerdict.HUMAN_REVIEW
    assert "content_policy_rejection" in scene_report.safety_flags
    assert "NSFW" in scene_report.critique
    assert llm.calls == 0  # never attempted to score a video that doesn't exist


async def test_scene_passes_above_threshold(settings_test: Settings) -> None:
    storage = StubStorage()
    state = await _seeded_state(storage)
    llm = _ScriptedLLM(_qa_report(scores=_PASSING_SCORES))

    result = await critic.run(state, ports=_ports(llm, storage), settings=settings_test)

    assert _scene_report(result["qa_reports"]).verdict == QAVerdict.PASS
    assert _audio_report(result["qa_reports"]).verdict == QAVerdict.PASS
    assert 0 not in result["retry_counts"]


async def test_scene_retries_below_threshold(settings_test: Settings) -> None:
    storage = StubStorage()
    state = await _seeded_state(storage)
    llm = _ScriptedLLM(_qa_report(scores=_FAILING_SCORES, critique="fix the duck's colors"))

    result = await critic.run(state, ports=_ports(llm, storage), settings=settings_test)

    report = _scene_report(result["qa_reports"])
    assert report.verdict == QAVerdict.RETRY
    assert report.critique == "fix the duck's colors"
    assert result["retry_counts"][0] == 1


async def test_scene_escalates_to_human_review_after_retry_ceiling(settings_test: Settings) -> None:
    storage = StubStorage()
    # already 2 prior attempts -- this failing round should hit the ceiling (3)
    state = await _seeded_state(storage, retry_counts={0: 2})
    llm = _ScriptedLLM(_qa_report(scores=_FAILING_SCORES))

    result = await critic.run(state, ports=_ports(llm, storage), settings=settings_test)

    report = _scene_report(result["qa_reports"])
    assert report.verdict == QAVerdict.HUMAN_REVIEW
    assert result["retry_counts"][0] == 3


async def test_safety_flag_forces_human_review_regardless_of_score(settings_test: Settings) -> None:
    storage = StubStorage()
    state = await _seeded_state(storage)
    llm = _ScriptedLLM(_qa_report(scores=_PASSING_SCORES, safety_flags=["scary_imagery"]))

    result = await critic.run(state, ports=_ports(llm, storage), settings=settings_test)

    report = _scene_report(result["qa_reports"])
    assert report.verdict == QAVerdict.HUMAN_REVIEW
    # safety issues never count against (or need) the retry ceiling
    assert 0 not in result["retry_counts"]


async def test_safety_flag_bypasses_retry_ceiling_on_first_attempt(settings_test: Settings) -> None:
    storage = StubStorage()
    state = await _seeded_state(storage)  # attempt 1, would otherwise just RETRY
    llm = _ScriptedLLM(_qa_report(scores=_FAILING_SCORES, safety_flags=["violence"]))

    result = await critic.run(state, ports=_ports(llm, storage), settings=settings_test)

    assert _scene_report(result["qa_reports"]).verdict == QAVerdict.HUMAN_REVIEW


class _RaisingLLM:
    """Simulates a provider outage (e.g. Replicate out of credit, HTTP 429)
    happening mid-QA-call."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    async def complete_structured(self, **kwargs: Any) -> tuple[BaseModel, float]:
        self.calls += 1
        raise self._exc


async def test_regression_scene_qa_provider_error_yields_inconclusive_not_crash(
    settings_test: Settings,
) -> None:
    """Confirmed live: a real run ran out of Replicate credit mid-QA and the
    unhandled exception crashed the whole pipeline even though every scene
    had already been generated (real money already spent) -- total loss, no
    final_video. The vision-LLM call failing must degrade to an INCONCLUSIVE
    verdict for that scene instead of propagating and losing everything
    already paid for."""
    storage = StubStorage()
    state = await _seeded_state(storage)
    llm = _RaisingLLM(RuntimeError("HTTP 429 from Replicate: rate limited, low credit"))

    result = await critic.run(state, ports=_ports(llm, storage), settings=settings_test)

    scene_report = _scene_report(result["qa_reports"])
    assert scene_report.verdict == QAVerdict.INCONCLUSIVE
    assert "429" in scene_report.critique
    # audio QA still ran independently -- one scene's outage doesn't block it
    assert _audio_report(result["qa_reports"]).verdict == QAVerdict.PASS


async def test_regression_audio_qa_provider_error_yields_inconclusive_not_crash(
    settings_test: Settings,
) -> None:
    """Same failure mode as above, but the outage hits transcription during
    audio QA instead of the per-scene vision call."""
    storage = StubStorage()
    state = await _seeded_state(storage)
    llm = _ScriptedLLM(_qa_report(scores=_PASSING_SCORES))

    class _RaisingTranscribe:
        async def transcribe(self, **kwargs: Any) -> tuple[list[dict[str, Any]], float]:
            raise RuntimeError("HTTP 429 from Replicate: rate limited, low credit")

    ports = PortBundle(
        llm=llm,
        image_gen=StubImageGen(),
        video_gen=StubVideoGen(),
        music_gen=StubMusicGen(),
        tts=StubTTS(),
        transcribe=_RaisingTranscribe(),
        storage=storage,
        publish=StubPublish(),
        notify=StubNotify(),
    )

    result = await critic.run(state, ports=ports, settings=settings_test)

    # scene QA still ran independently -- audio's outage doesn't block it
    assert _scene_report(result["qa_reports"]).verdict == QAVerdict.PASS
    audio_report = _audio_report(result["qa_reports"])
    assert audio_report.verdict == QAVerdict.INCONCLUSIVE
    assert "429" in audio_report.critique
