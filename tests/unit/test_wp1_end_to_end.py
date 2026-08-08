from __future__ import annotations

import pytest
from storysmith.models import AssetKind, Mode, ProjectStatus
from storysmith.pipeline import Pipeline

pytestmark = pytest.mark.wp1


async def test_graph_end_to_end_with_stubs(pipeline_stubbed: Pipeline) -> None:
    result = await pipeline_stubbed.run(
        brief="counting to five with ducks", mode=Mode.RHYME, project_id="wp1-e2e"
    )

    assert result.status == ProjectStatus.REVIEW
    kinds = {a.kind for a in result.assets}
    assert AssetKind.FINAL_VIDEO in kinds
    assert AssetKind.THUMBNAIL in kinds
    assert result.cost_ledger
    assert result.total_cost > 0


async def test_graph_end_to_end_topical_mode(pipeline_stubbed: Pipeline) -> None:
    result = await pipeline_stubbed.run(
        brief="sharing toys with friends", mode=Mode.TOPICAL, project_id="wp1-e2e-topical"
    )

    assert result.status == ProjectStatus.REVIEW
    kinds = {a.kind for a in result.assets}
    assert AssetKind.FINAL_VIDEO in kinds
    assert AssetKind.THUMBNAIL in kinds
    assert result.cost_ledger
