from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

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


def _critic_router(state: VideoProject) -> str:
    if any(r.verdict == QAVerdict.HUMAN_REVIEW for r in state.qa_reports):
        return "human_review"
    if any(r.verdict == QAVerdict.RETRY for r in state.qa_reports):
        return "retry"
    return "pass"


def _checkpointer(settings: Settings) -> BaseCheckpointSaver[str]:
    # SPEC-GAP: AsyncPostgresSaver needs an opened connection context (async
    # context manager over a pool, plus its checkpoint-table migration) beyond
    # a bare constructor call — wiring that is WP8 scope. WP1 always uses
    # MemorySaver so the graph is runnable/testable without a live Postgres
    # instance; settings.db_url is accepted now so WP8 has a hook point.
    del settings
    serde = JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_ALLOWED_TYPES)
    return MemorySaver(serde=serde)


def build_graph(
    *, settings: Settings, ports: PortBundle
) -> CompiledStateGraph[VideoProject, Any, Any]:
    graph: StateGraph[VideoProject, Any, Any] = StateGraph(VideoProject)

    def bind(fn: Callable[..., Awaitable[dict[str, Any]]], *, owns_status: bool = True) -> Any:
        # cast to Any: mypy can't unify Callable[[VideoProject], Awaitable[dict |
        # Command]] against add_node's generic _Node[NodeInputT] overloads (it
        # infers NodeInputT=Never and rejects every overload) even though
        # Command is a runtime-valid node return type LangGraph itself supports.
        bound = functools.partial(fn, ports=ports, settings=settings)
        return _budget_guarded(bound, owns_status=owns_status)

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
        {"retry": "videographer", "pass": "editor", "human_review": "review_gate"},
    )
    graph.add_edge("editor", "review_gate")
    graph.add_edge("review_gate", "publisher")
    graph.add_edge("publisher", END)

    return graph.compile(checkpointer=_checkpointer(settings), interrupt_before=["publisher"])
