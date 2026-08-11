from __future__ import annotations

from datetime import UTC, date, datetime
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


def _postgres_reachable(db_url: str) -> bool:
    import asyncio

    async def _check() -> bool:
        try:
            await db.ensure_schema(db_url)
        except Exception:
            return False
        return True

    return asyncio.run(_check())


@pytest.fixture
def pg_required(settings_test_pg: Settings) -> Settings:
    if not _postgres_reachable(settings_test_pg.db_url):
        pytest.skip("no reachable Postgres at SS_DB_URL / localhost:5432 -- see docker-compose.yml")
    return settings_test_pg


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


async def test_cost_ledger_writes_rows(pg_required: Settings) -> None:
    entries = [
        CostEntry(at=datetime.now(UTC), item="unit-test:a", provider="llm", cost_usd=1.5),
        CostEntry(at=datetime.now(UTC), item="unit-test:b", provider="video_gen", cost_usd=2.25),
    ]
    await db.record_cost_entries(pg_required.db_url, project_id="wp8-ledger-test", entries=entries)

    spent = await db.sum_cost_for_day(pg_required.db_url, day=date.today())
    assert spent >= 3.75 - 1e-9


async def test_resume_across_fresh_pipeline_instance(pg_required: Settings) -> None:
    """Proves cross-process resumability: a second, entirely new Pipeline +
    ports (nothing shared except settings.db_url and the storage backend,
    standing in for real external storage like S3) resumes the same
    project_id from Postgres without re-running the already-checkpointed
    early stages."""
    project_id = f"wp8-resume-{id(pg_required)}"
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


async def test_daily_budget_cap_blocks_new_run(pg_required: Settings) -> None:
    await db.record_cost_entries(
        pg_required.db_url,
        project_id="wp8-cap-test",
        entries=[CostEntry(at=datetime.now(UTC), item="x", provider="llm", cost_usd=999.0)],
    )
    capped_settings = pg_required.model_copy(update={"daily_budget_cap_usd": 1.0})
    pipeline = Pipeline(
        settings=capped_settings, ports=_ports(StubVideoGen(), StubLLM(), StubStorage())
    )

    with pytest.raises(BudgetExceededError):
        await pipeline.run(brief="counting ducks", mode=Mode.RHYME, project_id="wp8-cap-run")


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
