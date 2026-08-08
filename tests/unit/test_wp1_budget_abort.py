from __future__ import annotations

import pytest
from storysmith.models import AssetKind, Mode, ProjectStatus
from storysmith.pipeline import Pipeline
from storysmith.settings import Settings
from storysmith_adapters.stubs import stub_port_bundle

pytestmark = pytest.mark.wp1


async def test_budget_abort(settings_test: Settings) -> None:
    settings_test = settings_test.model_copy(update={"budget_cap_usd": 0.002})
    pipeline = Pipeline(settings=settings_test, ports=stub_port_bundle())

    result = await pipeline.run(brief="counting ducks", mode=Mode.RHYME, project_id="wp1-budget")

    assert result.status == ProjectStatus.BUDGET_ABORT
    # aborted before videographer ran: no scene videos were generated
    assert not any(a.kind == AssetKind.SCENE_VIDEO for a in result.assets)
