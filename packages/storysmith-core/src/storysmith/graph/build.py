from __future__ import annotations

import functools
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import opik
import structlog
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from storysmith import db
from storysmith.graph import nodes
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
from storysmith.settings import Settings

# Every non-builtin type that can end up inside a VideoProject checkpoint.
# Without this, JsonPlusSerializer falls back to permissive msgpack decoding
# for these types and warns "will be blocked in a future version" on every
# resume; LANGGRAPH_STRICT_MSGPACK=true would then break resumption outright.
_CHECKPOINT_ALLOWED_TYPES = [
    Mode,
    ProjectStatus,
    CharacterRef,
    StyleContract,
    MusicCue,
    Scene,
    SceneManifest,
    AssetKind,
    AssetRef,
    QAVerdict,
    QAReport,
    CostEntry,
]

if TYPE_CHECKING:
    from storysmith.pipeline import PortBundle

NodeFn = Callable[[VideoProject], Awaitable[dict[str, Any]]]
GuardedNodeFn = Callable[[VideoProject], Awaitable[dict[str, Any] | Command[Any]]]

_log = structlog.get_logger()


def _instrumented(fn: NodeFn, *, node_name: str, settings: Settings) -> NodeFn:
    """Diagnosability + durable cost ledger, applied at the same choke point
    every node already passes through. structlog gives per-node start/error/
    success events (project_id bound); when settings.db_url is set, any
    CostEntry a node's return dict adds to cost_ledger is also written to the
    cost_entries Postgres table (§8) -- unlike VideoProject.cost_ledger, that
    table survives past this run/process, which is what lets a daily budget
    cap (or any other cross-run query) actually work. When settings.opik_enabled,
    the same choke point gets an opik.track span per node.

    SPEC-GAP: §8 says "decorate every agent function and adapter call" --
    doing that individually across agents/*.py and adapters/*.py is a lot of
    call sites for one span per node already gives. Node-level tracing (one
    span per graph node, which is also the retry/cost-ledger granularity
    everywhere else in this codebase) is the simplest option that satisfies
    the stated goal ("know exactly what went wrong and where"); finer-grained
    per-adapter-call spans can be added later if node-level traces prove too
    coarse in practice.
    """
    traced: NodeFn
    if settings.opik_enabled:
        traced = opik.track(name=node_name, project_name="storysmith")(fn)
    else:
        traced = fn

    async def wrapper(state: VideoProject) -> dict[str, Any]:
        log = _log.bind(project_id=state.project_id, node=node_name)
        log.info("node_start")
        try:
            result: dict[str, Any] = await traced(state)
        except Exception:
            log.exception("node_failed")
            raise
        new_entries = result.get("cost_ledger")
        if new_entries and settings.db_url:
            await db.record_cost_entries(
                settings.db_url, project_id=state.project_id, entries=new_entries
            )
        log.info("node_done", new_cost_usd=sum(e.cost_usd for e in new_entries or []))
        return result

    return wrapper


def _budget_guarded(fn: NodeFn, *, owns_status: bool = True) -> GuardedNodeFn:
    """§1.4 budget guard: applied to every node, checked before it runs.

    owns_status=False for music_director: it runs in parallel with
    videographer (both fan out from char_refs and join at critic), so if both
    guards trip in the same superstep only one may write "status" or
    LangGraph raises InvalidUpdateError on the concurrent write — the same
    reason music_director's normal return already omits "status".
    """

    async def wrapper(state: VideoProject) -> dict[str, Any] | Command[Any]:
        if state.total_cost > state.budget_cap_usd:
            update = {"status": ProjectStatus.BUDGET_ABORT} if owns_status else {}
            return Command(goto=END, update=update)
        return await fn(state)

    return wrapper


def _critic_router(state: VideoProject) -> list[str]:
    # Any HUMAN_REVIEW (scene or audio) takes priority over retrying --
    # further automatic retries are pointless once a human needs to look.
    if any(r.verdict == QAVerdict.HUMAN_REVIEW for r in state.qa_reports):
        return ["review_gate"]

    destinations = []
    if any(r.verdict == QAVerdict.RETRY and r.scene_index is not None for r in state.qa_reports):
        destinations.append("videographer")
    if any(r.verdict == QAVerdict.RETRY and r.scene_index is None for r in state.qa_reports):
        # scene_index=None + RETRY is the audio report (§6) -- route back to
        # music_director, not videographer, so a bad narration/lyrics take
        # doesn't waste a scene-regeneration cycle (and vice versa).
        destinations.append("music_director")
    return destinations or ["editor"]


def _serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_ALLOWED_TYPES)


def memory_checkpointer() -> BaseCheckpointSaver[str]:
    """settings.db_url empty -> in-process-only resumability: a MemorySaver
    survives across multiple Pipeline.run() calls on the same instance (e.g.
    a crashed node's own retry), but not across process restarts. Callers
    must build exactly one of these per Pipeline and reuse it -- a fresh
    MemorySaver per run() would have no prior checkpoint to resume from.
    """
    return MemorySaver(serde=_serde())


@asynccontextmanager
async def postgres_checkpointer_context(
    settings: Settings,
) -> AsyncIterator[BaseCheckpointSaver[str]]:
    """Real cross-process resumability (§8): a crashed/killed run loses
    everything under MemorySaver, so it has to regenerate every costly asset
    from scratch. With settings.db_url set, checkpoints go to Postgres
    instead — a fresh Pipeline in a fresh process can resume the same
    thread_id (project_id) exactly where a prior run left off.

    AsyncPostgresSaver owns a connection pool with its own async-context-
    manager lifecycle, so (unlike MemorySaver) it can't be built once in
    Pipeline.__init__ and reused — it has to be constructed fresh, inside
    this context, for the duration of one run().
    """
    conn_string = db.to_psycopg_dsn(settings.db_url)
    async with AsyncPostgresSaver.from_conn_string(conn_string) as saver:
        saver.serde = _serde()
        await saver.setup()
        yield saver


def build_graph(
    *, settings: Settings, ports: PortBundle, checkpointer: BaseCheckpointSaver[str]
) -> CompiledStateGraph[VideoProject, Any, Any]:
    graph: StateGraph[VideoProject, Any, Any] = StateGraph(VideoProject)

    def bind(fn: Callable[..., Awaitable[dict[str, Any]]], *, owns_status: bool = True) -> Any:
        # cast to Any: mypy can't unify Callable[[VideoProject], Awaitable[dict |
        # Command]] against add_node's generic _Node[NodeInputT] overloads (it
        # infers NodeInputT=Never and rejects every overload) even though
        # Command is a runtime-valid node return type LangGraph itself supports.
        bound = functools.partial(fn, ports=ports, settings=settings)
        node_name = getattr(fn, "__name__", "node")
        instrumented = _instrumented(bound, node_name=node_name, settings=settings)
        return _budget_guarded(instrumented, owns_status=owns_status)

    graph.add_node("creative_director", bind(nodes.creative_director))
    graph.add_node("director", bind(nodes.director))
    graph.add_node("char_refs", bind(nodes.char_refs))
    graph.add_node("videographer", bind(nodes.videographer))
    graph.add_node("music_director", bind(nodes.music_director, owns_status=False))
    graph.add_node("critic", bind(nodes.critic))
    graph.add_node("editor", bind(nodes.editor))
    graph.add_node("review_gate", bind(nodes.review_gate))
    graph.add_node("publisher", bind(nodes.publisher))

    graph.add_edge(START, "creative_director")
    graph.add_edge("creative_director", "director")
    graph.add_edge("director", "char_refs")
    graph.add_edge("char_refs", "videographer")
    graph.add_edge("char_refs", "music_director")
    graph.add_edge("videographer", "critic")
    graph.add_edge("music_director", "critic")
    graph.add_conditional_edges(
        "critic",
        _critic_router,
        ["videographer", "music_director", "editor", "review_gate"],
    )
    graph.add_edge("editor", "review_gate")
    graph.add_edge("review_gate", "publisher")
    graph.add_edge("publisher", END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=["publisher"])
