from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from storysmith import db
from storysmith.errors import BudgetExceededError
from storysmith.models import AssetKind, CostEntry, Mode, ProjectStatus, VideoProject
from storysmith.pipeline import Pipeline, PortBundle, build_run_summary
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

pytestmark = pytest.mark.wp8


@pytest.fixture
async def cost_entries_cleanup(pg_required: Settings) -> AsyncIterator[list[str]]:
    """sum_cost_for_day (the live daily-budget-cap guard) sums every
    cost_entries row for the day regardless of project_id -- a test writing
    entries directly under a shared dev/CI Postgres must delete them
    afterward or it silently inflates the real app's "spent today" total.
    Confirmed to actually happen: this file's synthetic $999/$1.5/$2.25
    rows once summed past $4000 across repeated test runs, incorrectly
    blocking a real live run. Tests append whichever project_id(s) they
    wrote cost entries under (directly via db.record_cost_entries, or
    indirectly by running a real Pipeline) to the yielded list; cleanup
    runs even if the test fails."""
    project_ids: list[str] = []
    try:
        yield project_ids
    finally:
        for project_id in project_ids:
            await db.delete_cost_entries_for_project(pg_required.db_url, project_id=project_id)


class _CountingLLM(StubLLM):
    def __init__(self) -> None:
        self.calls = 0

    async def complete_structured(self, **kwargs: Any) -> tuple[Any, float]:
        self.calls += 1
        return await super().complete_structured(**kwargs)


class _FlakyVideoGen(StubVideoGen):
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


def _ports(video_gen: Any, llm: Any, storage: Any) -> PortBundle:
    return PortBundle(
        llm=llm,
        image_gen=StubImageGen(),
        video_gen=video_gen,
        music_gen=StubMusicGen(),
        tts=StubTTS(),
        transcribe=StubTranscribe(),
        storage=storage,
        publish=StubPublish(),
        notify=StubNotify(),
    )


async def test_cost_ledger_writes_rows(
    pg_required: Settings, cost_entries_cleanup: list[str]
) -> None:
    project_id = f"wp8-ledger-test-{uuid.uuid4()}"
    cost_entries_cleanup.append(project_id)
    entries = [
        CostEntry(at=datetime.now(UTC), item="unit-test:a", provider="llm", cost_usd=1.5),
        CostEntry(at=datetime.now(UTC), item="unit-test:b", provider="video_gen", cost_usd=2.25),
    ]
    before = await db.sum_cost_for_day(pg_required.db_url, day=datetime.now(UTC).date())
    await db.record_cost_entries(pg_required.db_url, project_id=project_id, entries=entries)

    spent = await db.sum_cost_for_day(pg_required.db_url, day=datetime.now(UTC).date())
    assert spent >= before + 3.75 - 1e-9


async def test_resume_across_fresh_pipeline_instance(
    pg_required: Settings, cost_entries_cleanup: list[str]
) -> None:
    """Proves cross-process resumability: a second, entirely new Pipeline +
    ports (nothing shared except settings.db_url and the storage backend,
    standing in for real external storage like S3) resumes the same
    project_id from Postgres without re-running the already-checkpointed
    early stages."""
    # uuid, not id(pg_required): a Python object id is a memory address,
    # not guaranteed unique across separate runs/processes against the same
    # real Postgres (this project_id doubles as the LangGraph checkpoint
    # thread_id) -- confirmed collisions in practice.
    project_id = f"wp8-resume-{uuid.uuid4()}"
    cost_entries_cleanup.append(project_id)
    shared_storage = StubStorage()

    video_gen_a = _FlakyVideoGen(fail_marker="scene 2")
    llm_a = _CountingLLM()
    pipeline_a = Pipeline(settings=pg_required, ports=_ports(video_gen_a, llm_a, shared_storage))
    with pytest.raises(RuntimeError):
        await pipeline_a.run(brief="counting ducks", mode=Mode.RHYME, project_id=project_id)
    assert llm_a.calls == 2  # creative_director + director only

    video_gen_b = _FlakyVideoGen(fail_marker="never matches")
    llm_b = _CountingLLM()
    pipeline_b = Pipeline(settings=pg_required, ports=_ports(video_gen_b, llm_b, shared_storage))
    result = await pipeline_b.run(brief="counting ducks", mode=Mode.RHYME, project_id=project_id)

    # critic legitimately calls the LLM once per scene after resuming (5
    # scenes) -- what must NOT happen is creative_director/director re-running
    # (2 more calls), which would show up as llm_b.calls > 5.
    assert llm_b.calls == 5
    assert result.status == ProjectStatus.REVIEW
    scene_videos = [a for a in result.assets if a.kind == AssetKind.SCENE_VIDEO]
    assert len(scene_videos) == 5


async def test_daily_budget_cap_blocks_new_run(
    pg_required: Settings, cost_entries_cleanup: list[str]
) -> None:
    cap_test_project_id = f"wp8-cap-test-{uuid.uuid4()}"
    cost_entries_cleanup.append(cap_test_project_id)
    await db.record_cost_entries(
        pg_required.db_url,
        project_id=cap_test_project_id,
        entries=[CostEntry(at=datetime.now(UTC), item="x", provider="llm", cost_usd=999.0)],
    )
    capped_settings = pg_required.model_copy(update={"daily_budget_cap_usd": 1.0})
    pipeline = Pipeline(
        settings=capped_settings, ports=_ports(StubVideoGen(), StubLLM(), StubStorage())
    )

    with pytest.raises(BudgetExceededError):
        await pipeline.run(
            brief="counting ducks", mode=Mode.RHYME, project_id=f"wp8-cap-run-{uuid.uuid4()}"
        )


def test_daily_budget_cap_disabled_by_default(settings_test: Settings) -> None:
    assert settings_test.daily_budget_cap_usd == 0.0


def test_run_summary_formatter_golden() -> None:
    project = VideoProject(
        project_id="p1",
        mode=Mode.RHYME,
        brief="test",
        status=ProjectStatus.REVIEW,
        retry_counts={0: 2, 1: 0, 3: 1},
        cost_ledger=[
            CostEntry(at=datetime.now(UTC), item="a", provider="llm", cost_usd=0.1),
            CostEntry(at=datetime.now(UTC), item="b", provider="llm", cost_usd=0.2),
            CostEntry(at=datetime.now(UTC), item="c", provider="video_gen", cost_usd=0.5),
        ],
    )
    summary = build_run_summary(project, wall_time_s=12.345)
    assert summary == {
        "status": ProjectStatus.REVIEW,
        "total_cost_usd": 0.8,
        "cost_by_provider": {"llm": 0.3, "video_gen": 0.5},
        "retries_per_scene": {0: 2, 3: 1},
        "wall_time_s": 12.35,
    }


def test_run_summary_formatter_empty_ledger() -> None:
    project = VideoProject(project_id="p2", mode=Mode.TOPICAL, brief="test")
    summary = build_run_summary(project, wall_time_s=0.0)
    assert summary["total_cost_usd"] == 0.0
    assert summary["cost_by_provider"] == {}
    assert summary["retries_per_scene"] == {}
