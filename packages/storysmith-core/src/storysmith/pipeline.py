from __future__ import annotations

import time
import uuid
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver

from storysmith import db
from storysmith.errors import BudgetExceededError
from storysmith.graph.build import build_graph, memory_checkpointer, postgres_checkpointer_context
from storysmith.models import Mode, VideoProject
from storysmith.ports import (
    ImageGenPort,
    LLMPort,
    MusicGenPort,
    NotifyPort,
    PublishPort,
    StoragePort,
    TranscribePort,
    TTSPort,
    VideoGenPort,
)
from storysmith.settings import Settings

_log = structlog.get_logger()


@dataclass
class PortBundle:
    llm: LLMPort
    image_gen: ImageGenPort
    video_gen: VideoGenPort
    music_gen: MusicGenPort
    tts: TTSPort
    transcribe: TranscribePort
    storage: StoragePort
    publish: PublishPort
    notify: NotifyPort


class Pipeline:
    def __init__(self, settings: Settings, ports: PortBundle) -> None:
        self.settings = settings
        self.ports = ports
        # settings.db_url empty -> one MemorySaver reused for this Pipeline's
        # lifetime, so a resumed run() call (e.g. after a node-level failure)
        # can still see checkpoints from an earlier run() on the same
        # instance -- see memory_checkpointer()'s docstring.
        self._memory_checkpointer: BaseCheckpointSaver[str] | None = (
            None if settings.db_url else memory_checkpointer()
        )

    @classmethod
    def with_stubs(cls, settings: Settings) -> Pipeline:
        # Local import: storysmith-core must not statically depend on
        # storysmith-adapters (adapters depends on core, not the reverse).
        # This is a dev/test convenience wired in only when actually called.
        from storysmith_adapters.stubs import stub_port_bundle

        return cls(settings=settings, ports=stub_port_bundle())

    async def run(self, brief: str, mode: Mode, project_id: str | None = None) -> VideoProject:
        project_id = project_id or str(uuid.uuid4())
        config = RunnableConfig(configurable={"thread_id": project_id})

        if self.settings.db_url and self.settings.daily_budget_cap_usd > 0:
            spent_today = await db.sum_cost_for_day(self.settings.db_url, day=date.today())
            if spent_today >= self.settings.daily_budget_cap_usd:
                raise BudgetExceededError(
                    f"today's spend ${spent_today:.2f} already at/over daily cap "
                    f"${self.settings.daily_budget_cap_usd:.2f}"
                )

        log = _log.bind(project_id=project_id)
        started = time.monotonic()

        checkpointer_ctx: AbstractAsyncContextManager[BaseCheckpointSaver[str]]
        if self.settings.db_url:
            # AsyncPostgresSaver's connection pool is only valid inside this
            # context, so the graph is rebuilt fresh per run() -- see
            # postgres_checkpointer_context's docstring for why.
            checkpointer_ctx = postgres_checkpointer_context(self.settings)
        else:
            assert self._memory_checkpointer is not None
            checkpointer_ctx = nullcontext(self._memory_checkpointer)

        async with checkpointer_ctx as checkpointer:
            if self.settings.db_url:
                await db.ensure_schema(self.settings.db_url)
            graph = build_graph(settings=self.settings, ports=self.ports, checkpointer=checkpointer)

            try:
                snapshot = await graph.aget_state(config)
                if snapshot.values:
                    log.info("run_resuming")
                    result = await graph.ainvoke(None, config=config)
                else:
                    log.info("run_starting", mode=mode, brief=brief)
                    initial = VideoProject(
                        project_id=project_id,
                        mode=mode,
                        brief=brief,
                        budget_cap_usd=self.settings.budget_cap_usd,
                    )
                    result = await graph.ainvoke(initial, config=config)
            except Exception:
                log.exception("run_failed", wall_time_s=round(time.monotonic() - started, 2))
                raise

        project = VideoProject.model_validate(cast(dict[str, Any], result))
        summary = build_run_summary(project, wall_time_s=time.monotonic() - started)
        log.info("run_summary", **summary)
        return project


def build_run_summary(project: VideoProject, *, wall_time_s: float) -> dict[str, Any]:
    """Pure formatter for the §8 end-of-run summary (total cost, per-provider
    breakdown, retries per scene, wall time) -- split out from run() so it has
    a golden-test-able shape independent of structlog/log capture."""
    cost_by_provider: dict[str, float] = {}
    for entry in project.cost_ledger:
        cost_by_provider[entry.provider] = (
            cost_by_provider.get(entry.provider, 0.0) + entry.cost_usd
        )
    retries_per_scene = {
        idx: attempts for idx, attempts in project.retry_counts.items() if attempts > 0
    }
    return {
        "status": project.status,
        "total_cost_usd": round(project.total_cost, 4),
        "cost_by_provider": {k: round(v, 4) for k, v in cost_by_provider.items()},
        "retries_per_scene": retries_per_scene,
        "wall_time_s": round(wall_time_s, 2),
    }
