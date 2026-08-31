"""FastAPI review console API (§7): project listing, approve/reject, run trigger.

Auth: every route except /healthz requires `Authorization: Bearer <SS_API_BEARER_TOKEN>`.
"""

from __future__ import annotations

import asyncio
import mimetypes
import uuid
from datetime import UTC, datetime
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from storysmith import db
from storysmith.graph.build import build_graph, postgres_checkpointer_context
from storysmith.models import (
    CharacterRef,
    CostEntry,
    Mode,
    ProjectStatus,
    StyleContract,
    VideoProject,
)
from storysmith.pipeline import Pipeline, PortBundle
from storysmith.settings import Settings
from storysmith.util.character_prompts import CHAR_REF_ASPECT_RATIO, build_char_ref_prompt
from storysmith.util.configs import load_safety_negative_terms

app = FastAPI(title="StorySmith Review Console API")
# apps/ui runs on a different origin (console_base_url, default
# localhost:3000) than this API -- without CORS the browser blocks every
# fetch from the review console with no server-side error to debug.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[Settings().console_base_url],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_ports(settings: Settings = Depends(get_settings)) -> PortBundle:
    # Real adapters constructed lazily via FastAPI's dependency injection
    # (not at import/module-load time) so /healthz still works without e.g.
    # SS_ANTHROPIC_API_KEY set, and so tests can override this dependency
    # with a stub PortBundle instead of touching real providers.
    from storysmith_adapters.factory import build_port_bundle

    return build_port_bundle(settings)


def require_bearer_token(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.api_bearer_token:
        # Failing closed on a missing server-side token, rather than silently
        # accepting every request, is the safer default for an unconfigured
        # deployment.
        raise HTTPException(status_code=500, detail="SS_API_BEARER_TOKEN is not configured")
    if authorization != f"Bearer {settings.api_bearer_token}":
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")


def _require_db_url(settings: Settings) -> str:
    if not settings.db_url:
        raise HTTPException(status_code=500, detail="SS_DB_URL is not configured")
    return settings.db_url


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


class ProjectSummary(BaseModel):
    project_id: str
    status: str
    mode: str
    brief: str
    title: str | None
    total_cost_usd: float
    updated_at: str
    show_id: str | None


@app.get(
    "/projects",
    response_model=list[ProjectSummary],
    dependencies=[Depends(require_bearer_token)],
)
async def list_projects(settings: Settings = Depends(get_settings)) -> list[ProjectSummary]:
    rows = await db.list_projects(_require_db_url(settings))
    return [
        ProjectSummary(
            project_id=row.project_id,
            status=row.status,
            mode=row.mode,
            brief=row.brief,
            title=row.title,
            total_cost_usd=row.total_cost_usd,
            updated_at=row.updated_at.isoformat(),
            show_id=row.show_id,
        )
        for row in rows
    ]


class AssetOut(BaseModel):
    kind: str
    scene_index: int | None
    attempt: int
    presigned_url: str


class QAReportOut(BaseModel):
    scene_index: int | None
    verdict: str
    scores: dict[str, float]
    safety_flags: list[str]
    critique: str


class ProjectDetail(BaseModel):
    project_id: str
    mode: str
    brief: str
    status: str
    title: str | None
    total_cost_usd: float
    qa_reports: list[QAReportOut]
    assets: list[AssetOut]
    published_url: str | None
    show_id: str | None


async def _load_project_state(
    project_id: str, settings: Settings, ports: PortBundle
) -> VideoProject:
    config = RunnableConfig(configurable={"thread_id": project_id})
    async with postgres_checkpointer_context(settings) as checkpointer:
        graph = build_graph(settings=settings, ports=ports, checkpointer=checkpointer)
        snapshot = await graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="project not found")
    return VideoProject.model_validate(snapshot.values)


async def _project_detail(project_id: str, settings: Settings, ports: PortBundle) -> ProjectDetail:
    project = await _load_project_state(project_id, settings, ports)
    assets = [
        AssetOut(
            kind=asset.kind.value,
            scene_index=asset.scene_index,
            attempt=asset.attempt,
            presigned_url=await ports.storage.presign(uri=asset.uri),
        )
        for asset in project.assets
    ]
    return ProjectDetail(
        project_id=project.project_id,
        mode=project.mode.value,
        brief=project.brief,
        status=project.status.value,
        title=project.manifest.title if project.manifest else None,
        total_cost_usd=round(project.total_cost, 4),
        qa_reports=[
            QAReportOut(
                scene_index=r.scene_index,
                verdict=r.verdict.value,
                scores=r.scores,
                safety_flags=r.safety_flags,
                critique=r.critique,
            )
            for r in project.qa_reports
        ],
        assets=assets,
        published_url=project.published_url,
        show_id=project.show_id,
    )


@app.get(
    "/projects/{project_id}",
    response_model=ProjectDetail,
    dependencies=[Depends(require_bearer_token)],
)
async def get_project(
    project_id: str,
    settings: Settings = Depends(get_settings),
    ports: PortBundle = Depends(get_ports),
) -> ProjectDetail:
    return await _project_detail(project_id, settings, ports)


@app.post(
    "/projects/{project_id}/approve",
    response_model=ProjectDetail,
    dependencies=[Depends(require_bearer_token)],
)
async def approve_project(
    project_id: str,
    settings: Settings = Depends(get_settings),
    ports: PortBundle = Depends(get_ports),
) -> ProjectDetail:
    # Graph is parked at interrupt_before=["publisher"] (graph/build.py) while
    # status=REVIEW; resuming with input=None lets it proceed into publisher,
    # which actually uploads to YouTube -- so this can take a while. Kept
    # synchronous (simplest option) rather than a background job queue; a
    # slow-client timeout here is a UI/infra concern, not a correctness one.
    config = RunnableConfig(configurable={"thread_id": project_id})
    async with postgres_checkpointer_context(settings) as checkpointer:
        graph = build_graph(settings=settings, ports=ports, checkpointer=checkpointer)
        snapshot = await graph.aget_state(config)
        if not snapshot.values:
            raise HTTPException(status_code=404, detail="project not found")
        if snapshot.values.get("status") != ProjectStatus.REVIEW:
            raise HTTPException(
                status_code=409,
                detail=f"project status is {snapshot.values.get('status')!r}, not REVIEW",
            )
        await graph.ainvoke(None, config=config)
    return await _project_detail(project_id, settings, ports)


@app.post(
    "/projects/{project_id}/reject",
    response_model=ProjectDetail,
    dependencies=[Depends(require_bearer_token)],
)
async def reject_project(
    project_id: str,
    settings: Settings = Depends(get_settings),
    ports: PortBundle = Depends(get_ports),
) -> ProjectDetail:
    # Marks REJECTED in place without resuming execution -- the graph never
    # proceeds into publisher, so nothing gets uploaded.
    config = RunnableConfig(configurable={"thread_id": project_id})
    async with postgres_checkpointer_context(settings) as checkpointer:
        graph = build_graph(settings=settings, ports=ports, checkpointer=checkpointer)
        snapshot = await graph.aget_state(config)
        if not snapshot.values:
            raise HTTPException(status_code=404, detail="project not found")
        if snapshot.values.get("status") != ProjectStatus.REVIEW:
            raise HTTPException(
                status_code=409,
                detail=f"project status is {snapshot.values.get('status')!r}, not REVIEW",
            )
        await graph.aupdate_state(config, {"status": ProjectStatus.REJECTED})
        state = VideoProject.model_validate(snapshot.values)
    await db.upsert_project_snapshot(
        settings.db_url,
        project_id=project_id,
        thread_id=project_id,
        status=str(ProjectStatus.REJECTED),
        mode=str(state.mode),
        brief=state.brief,
        title=state.manifest.title if state.manifest else None,
        total_cost_usd=state.total_cost,
        show_id=state.show_id,
    )
    return await _project_detail(project_id, settings, ports)


class RunRequest(BaseModel):
    brief: str
    mode: Mode = Mode.RHYME
    show_id: str | None = None


class RunResponse(BaseModel):
    project_id: str


@app.post("/runs", response_model=RunResponse, dependencies=[Depends(require_bearer_token)])
async def trigger_run(
    body: RunRequest,
    settings: Settings = Depends(get_settings),
    ports: PortBundle = Depends(get_ports),
) -> RunResponse:
    # Fail fast, synchronously, on an unknown show_id -- Pipeline.run() would
    # also raise ValueError for this, but only inside the fire-and-forget
    # background task below, where the caller never sees it (just a 200
    # "started" response and a run that silently dies). A cheap existence
    # check here turns that into an immediate 404 instead.
    if body.show_id is not None:
        db_url = _require_db_url(settings)
        if await db.load_show(db_url, show_id=body.show_id) is None:
            raise HTTPException(
                status_code=404, detail=f"no show found with show_id={body.show_id!r}"
            )

    project_id = str(uuid.uuid4())
    pipeline = Pipeline(settings=settings, ports=ports)
    # Fire-and-forget (§7: "trigger a worker job") -- the nightly pipeline is
    # long-running (LLM/video/music calls), so this endpoint returns
    # immediately with the project_id rather than blocking the HTTP request
    # for the whole run; progress is visible via GET /projects/{id}.
    asyncio.create_task(
        pipeline.run(brief=body.brief, mode=body.mode, project_id=project_id, show_id=body.show_id)
    )
    return RunResponse(project_id=project_id)


class CharacterInput(BaseModel):
    name: str
    description: str  # physical appearance -- feeds image-gen prompts verbatim
    personality: str = ""  # behavioral/voice guidance for the Director, not the image model
    voice_id: str | None = None


class CreateShowRequest(BaseModel):
    show_id: str
    name: str
    art_style: str
    palette: list[str] = []
    mood: str = "cheerful"
    tempo_bpm: int = 100
    aspect_ratio: str = "9:16"
    resolution: str = "1080x1920"
    pacing_rules: str = ""
    characters: list[CharacterInput]


class CharacterOut(BaseModel):
    name: str
    description: str
    personality: str
    voice_id: str | None
    image_asset_uri: str  # opaque -- pass straight to GET /assets/view


class ShowDetail(BaseModel):
    show_id: str
    name: str
    art_style: str
    characters: list[CharacterOut]


class ShowSummary(BaseModel):
    show_id: str
    name: str
    created_at: str


@app.post("/shows", response_model=ShowDetail, dependencies=[Depends(require_bearer_token)])
async def create_show(
    body: CreateShowRequest,
    settings: Settings = Depends(get_settings),
    ports: PortBundle = Depends(get_ports),
) -> ShowDetail:
    """User-authored, frozen cast (Amendment 02) -- no LLM involved. The
    user's own words become CharacterRef.description/personality directly;
    this only generates and locks in one reference portrait per character."""
    db_url = _require_db_url(settings)
    if not body.characters:
        raise HTTPException(status_code=422, detail="a show needs at least one character")

    base_terms = load_safety_negative_terms(settings.configs_dir)
    style = StyleContract(
        art_style=body.art_style,
        palette=body.palette,
        mood=body.mood,
        tempo_bpm=body.tempo_bpm,
        aspect_ratio=body.aspect_ratio,
        resolution=body.resolution,
        characters=[
            CharacterRef(
                name=c.name,
                description=c.description,
                personality=c.personality,
                voice_id=c.voice_id,
            )
            for c in body.characters
        ],
        pacing_rules=body.pacing_rules,
        negative_terms=base_terms,
    )

    async def _generate_avatar(character: CharacterRef) -> CharacterRef:
        prompt = build_char_ref_prompt(character, style.art_style)
        image_bytes, cost = await ports.image_gen.generate(
            prompt=prompt, aspect_ratio=CHAR_REF_ASPECT_RATIO
        )
        uri = await ports.storage.put(
            key=f"shows/{body.show_id}/char_{character.name}.png",
            data=image_bytes,
            content_type="image/png",
        )
        if settings.db_url:
            await db.record_cost_entries(
                settings.db_url,
                # Shows aren't projects -- a synthetic project_id keeps this
                # spend visible in the same cost ledger / daily budget cap
                # queries rather than falling outside them entirely.
                project_id=f"show:{body.show_id}",
                entries=[
                    CostEntry(
                        at=datetime.now(UTC),
                        item=f"show_avatar:{character.name}",
                        provider="image_gen",
                        cost_usd=cost,
                    )
                ],
            )
        return character.model_copy(update={"image_uri": uri})

    characters_with_avatars = await asyncio.gather(*(_generate_avatar(c) for c in style.characters))
    style = style.model_copy(update={"characters": list(characters_with_avatars)})

    await db.save_show(db_url, show_id=body.show_id, name=body.name, style=style)

    return ShowDetail(
        show_id=body.show_id,
        name=body.name,
        art_style=style.art_style,
        characters=[
            CharacterOut(
                name=c.name,
                description=c.description,
                personality=c.personality,
                voice_id=c.voice_id,
                image_asset_uri=c.image_uri or "",
            )
            for c in style.characters
        ],
    )


@app.get("/shows", response_model=list[ShowSummary], dependencies=[Depends(require_bearer_token)])
async def list_shows(settings: Settings = Depends(get_settings)) -> list[ShowSummary]:
    rows = await db.list_shows(_require_db_url(settings))
    return [
        ShowSummary(show_id=row.show_id, name=row.name, created_at=row.created_at.isoformat())
        for row in rows
    ]


@app.get("/assets/view", dependencies=[Depends(require_bearer_token)])
async def view_asset(uri: str, ports: PortBundle = Depends(get_ports)) -> Response:
    """Generic authenticated asset proxy (Amendment 02) -- streams any
    storage URI's bytes through the server via the same StoragePort.get()
    every agent already uses, sidestepping local storage's non-browser-
    fetchable local:// scheme entirely. Used client-side (fetch + Blob +
    object URL, since neither <img src> nor a plain link can carry the
    bearer header) for the show-creation avatar gallery and the final-video
    download button."""
    try:
        data = await ports.storage.get(uri=uri)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail="asset not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    content_type = mimetypes.guess_type(uri)[0] or "application/octet-stream"
    return Response(content=data, media_type=content_type)
