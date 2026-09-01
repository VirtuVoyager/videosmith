from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from storysmith import db
from storysmith.models import Mode, ProjectStatus, QAReport
from storysmith.pipeline import Pipeline
from storysmith.settings import Settings
from storysmith_adapters.stubs_recorded import (
    RecordedFixtures,
    RecordedImageGen,
    RecordedLLM,
    RecordedMusicGen,
    RecordedTTS,
    RecordedVideoGen,
    recorded_port_bundle,
)

_FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "recorded" / "crocky_roachy"

pytestmark = [
    pytest.mark.amendment02,
    pytest.mark.skipif(not _FIXTURE_ROOT.exists(), reason="recorded fixtures not present locally"),
]


def test_recorded_fixtures_loads_real_style_and_manifest() -> None:
    fixtures = RecordedFixtures(_FIXTURE_ROOT)

    assert fixtures.style is not None
    assert {c.name for c in fixtures.style.characters} == {"Crocky", "Roachy"}
    assert fixtures.manifest is not None
    assert len(fixtures.manifest.scenes) == 5
    assert len(fixtures.qa_reports) == 6  # 5 per-scene + 1 aggregate audio verdict


async def test_recorded_llm_replays_style_then_manifest_by_schema() -> None:
    from storysmith.models import SceneManifest, StyleContract

    fixtures = RecordedFixtures(_FIXTURE_ROOT)
    llm = RecordedLLM(fixtures)

    style, cost1 = await llm.complete_structured(system="s", user="u", schema=StyleContract)
    manifest, cost2 = await llm.complete_structured(system="s", user="u", schema=SceneManifest)

    assert style == fixtures.style
    assert manifest == fixtures.manifest
    assert cost1 == cost2 == 0.0


async def test_recorded_llm_replays_five_real_qa_reports_then_falls_back_to_stub() -> None:
    fixtures = RecordedFixtures(_FIXTURE_ROOT)
    llm = RecordedLLM(fixtures)

    real_reports = [
        (await llm.complete_structured(system="s", user="u", schema=QAReport))[0] for _ in range(5)
    ]
    assert real_reports == [r for r in fixtures.qa_reports if r.scene_index is not None]

    # The 6th real QAReport (scene_index=None, the aggregate audio verdict)
    # was never an LLM call at all -- computed from WER in code, see
    # RecordedFixtures's docstring -- so a 6th complete_structured call here
    # falls back to StubLLM's canned always-pass response.
    fallback_report, _ = await llm.complete_structured(system="s", user="u", schema=QAReport)
    assert isinstance(fallback_report, QAReport)
    assert fallback_report not in fixtures.qa_reports


async def test_recorded_image_and_video_replay_real_bytes_in_scene_order() -> None:
    fixtures = RecordedFixtures(_FIXTURE_ROOT)
    expected_stills = [
        (_FIXTURE_ROOT / "scene_stills" / f"scene_{i}.png").read_bytes() for i in range(5)
    ]
    expected_videos = [
        (_FIXTURE_ROOT / "scene_videos" / f"scene_{i}.mp4").read_bytes() for i in range(5)
    ]

    image_gen = RecordedImageGen(fixtures)
    video_gen = RecordedVideoGen(fixtures)

    for i in range(5):
        img_bytes, _ = await image_gen.generate(prompt="p", aspect_ratio="9:16")
        assert img_bytes == expected_stills[i]

        vid_bytes, _ = await video_gen.generate(
            prompt="p", duration_s=5.0, aspect_ratio="9:16", reference_image=None
        )
        assert vid_bytes == expected_videos[i]


async def test_recorded_tts_replays_real_clip_only_on_each_scenes_last_line() -> None:
    """music_director calls TTSPort.speak() once per dialogue line -- a
    multi-line scene's earlier calls never had their raw per-line audio
    persisted (only the ffmpeg-concatenated final clip was stored), so only
    the last call per scene can replay the real recorded content; earlier
    calls get a cheap synthetic placeholder instead (see RecordedTTS)."""
    fixtures = RecordedFixtures(_FIXTURE_ROOT)
    tts = RecordedTTS(fixtures)
    calls_per_scene = [
        len(s.dialogue) if s.dialogue else 1
        for s in fixtures.manifest.scenes  # type: ignore[union-attr]
    ]

    for scene_index, n_calls in enumerate(calls_per_scene):
        results = [await tts.speak(text="x", voice="am_adam") for _ in range(n_calls)]
        *placeholders, (last_bytes, _) = results
        expected = (_FIXTURE_ROOT / "narration" / f"scene_{scene_index}.mp3").read_bytes()
        assert last_bytes == expected
        assert all(bytes_ != expected for bytes_, _ in placeholders)


async def test_recorded_music_gen_replays_audio_bed() -> None:
    fixtures = RecordedFixtures(_FIXTURE_ROOT)
    music_gen = RecordedMusicGen(fixtures)

    audio_bytes, _ = await music_gen.generate(
        mode=Mode.TOPICAL, lyrics=None, description="bed", duration_s=30.0
    )

    assert audio_bytes == (_FIXTURE_ROOT / "audio_bed.mp3").read_bytes()


async def test_recorded_sequence_raises_clearly_when_exhausted() -> None:
    fixtures = RecordedFixtures(_FIXTURE_ROOT)
    image_gen = RecordedImageGen(fixtures)
    for _ in range(5):
        await image_gen.generate(prompt="p", aspect_ratio="9:16")

    with pytest.raises(IndexError, match="scene still image"):
        await image_gen.generate(prompt="p", aspect_ratio="9:16")


async def test_full_pipeline_run_against_recorded_fixtures(pg_required: Settings) -> None:
    """The real payoff: a full graph run, zero network calls, using real
    style/script/media throughout (only Critic's verdicts are the StubLLM
    fallback's synthetic always-pass) -- proves recorded_port_bundle() is
    actually usable for offline dev/testing, not just unit-testable in
    isolation."""
    fixtures = RecordedFixtures(_FIXTURE_ROOT)
    assert fixtures.style is not None
    # Unique per test invocation -- a fixed id resumes from whatever
    # checkpoint a *prior* run of this same test already left in the real
    # Postgres this pg_required fixture points at, whose asset URIs a fresh
    # in-memory StubStorage here never actually populated (KeyError on
    # storage.get). Same root cause as the project's other flaky-Postgres-
    # test follow-up; unique ids sidestep it here too.
    show_id = f"crocky-and-roachy-recorded-test-{uuid.uuid4()}"
    project_id = f"recorded-fixture-run-{uuid.uuid4()}"
    await db.save_show(
        pg_required.db_url, show_id=show_id, name="Recorded Test", style=fixtures.style
    )

    pipeline = Pipeline(settings=pg_required, ports=recorded_port_bundle(fixtures=fixtures))
    result = await pipeline.run(
        brief="a topic", mode=Mode.TOPICAL, project_id=project_id, show_id=show_id
    )

    assert result.status == ProjectStatus.REVIEW
    assert result.manifest == fixtures.manifest
    # Every recorded port replays at STUB_COST_USD=0.0 -- the only nonzero
    # cost is Critic's QAReport calls, which fall back to StubLLM (real
    # QA verdicts aren't recorded, see RecordedLLM's docstring).
    assert result.total_cost < 0.01
