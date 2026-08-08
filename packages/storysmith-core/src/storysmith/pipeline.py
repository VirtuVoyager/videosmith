from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from langchain_core.runnables import RunnableConfig

from storysmith.graph.build import build_graph
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

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


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
        self.graph: CompiledStateGraph[VideoProject, Any, Any] = build_graph(
            settings=settings, ports=ports
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

        snapshot = await self.graph.aget_state(config)
        if snapshot.values:
            result = await self.graph.ainvoke(None, config=config)
        else:
            initial = VideoProject(
                project_id=project_id,
                mode=mode,
                brief=brief,
                budget_cap_usd=self.settings.budget_cap_usd,
            )
            result = await self.graph.ainvoke(initial, config=config)
        return VideoProject.model_validate(cast(dict[str, Any], result))
