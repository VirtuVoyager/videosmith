from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel
from storysmith.models import (
    AssetKind,
    AssetRef,
    CharacterRef,
    CostEntry,
    Mode,
    MusicCue,
    ProjectStatus,
    QAReport,
    QAVerdict,
    Scene,
    SceneManifest,
    StyleContract,
    VideoProject,
)

pytestmark = pytest.mark.wp1


def _assert_roundtrips(model: BaseModel) -> None:
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model


def test_models_roundtrip() -> None:
    character = CharacterRef(name="Ducky", description="a yellow duck", image_uri="file:///x.png")
    music_cue = MusicCue(description="intro", start_s=0, end_s=2)
    style = StyleContract(
        art_style="soft 2D cutout animation",
        palette=["#FFAA00"],
        mood="cheerful",
        tempo_bpm=100,
        characters=[character],
        pacing_rules="keep cuts short",
        negative_terms=["scary faces"],
    )
    scene = Scene(index=0, duration_s=5, video_prompt="p", narration="n")
    manifest = SceneManifest(
        title="t",
        description="d",
        tags=["a"],
        total_duration_s=40.0,
        lyrics="l",
        music_cues=[music_cue],
        scenes=[scene],
    )
    asset = AssetRef(
        kind=AssetKind.SCENE_VIDEO, scene_index=0, uri="file:///x.mp4", content_hash="abc"
    )
    qa = QAReport(
        scene_index=0, verdict=QAVerdict.PASS, scores={"a": 0.9}, safety_flags=[], critique=""
    )
    cost = CostEntry(at=datetime.now(UTC), item="x", provider="p", cost_usd=0.1)
    project = VideoProject(
        project_id="p1",
        mode=Mode.RHYME,
        brief="b",
        status=ProjectStatus.REVIEW,
        style=style,
        manifest=manifest,
        assets=[asset],
        qa_reports=[qa],
        retry_counts={0: 1},
        cost_ledger=[cost],
        budget_cap_usd=12.0,
    )

    for model in (character, music_cue, style, scene, manifest, asset, qa, cost, project):
        _assert_roundtrips(model)
