from __future__ import annotations

import pytest
from storysmith.models import AssetKind, Mode, ProjectStatus
from storysmith.pipeline import Pipeline
from storysmith.settings import Settings

pytestmark = pytest.mark.wp5


async def _run_and_assert_final_assets(pipeline: Pipeline, *, mode: Mode, project_id: str) -> None:
    result = await pipeline.run(brief="counting ducks", mode=mode, project_id=project_id)

    assert result.status == ProjectStatus.REVIEW
    kinds = {a.kind for a in result.assets}
    assert AssetKind.FINAL_VIDEO in kinds
    assert AssetKind.THUMBNAIL in kinds

    final = next(a for a in result.assets if a.kind == AssetKind.FINAL_VIDEO)
    thumb = next(a for a in result.assets if a.kind == AssetKind.THUMBNAIL)
    final_bytes = await pipeline.ports.storage.get(uri=final.uri)
    thumb_bytes = await pipeline.ports.storage.get(uri=thumb.uri)
    assert len(final_bytes) > 0
    assert len(thumb_bytes) > 0


async def test_editor_end_to_end_rhyme_runs_real_ffmpeg(settings_test: Settings) -> None:
    if settings_test.skip_ffmpeg:
        pytest.skip("SS_SKIP_FFMPEG=1")
    pipeline = Pipeline.with_stubs(settings_test)
    await _run_and_assert_final_assets(pipeline, mode=Mode.RHYME, project_id="wp5-e2e-rhyme")


async def test_editor_end_to_end_topical_runs_real_ffmpeg(settings_test: Settings) -> None:
    if settings_test.skip_ffmpeg:
        pytest.skip("SS_SKIP_FFMPEG=1")
    pipeline = Pipeline.with_stubs(settings_test)
    await _run_and_assert_final_assets(pipeline, mode=Mode.TOPICAL, project_id="wp5-e2e-topical")
