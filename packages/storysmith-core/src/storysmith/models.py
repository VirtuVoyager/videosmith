from __future__ import annotations

import operator
from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field


class Mode(StrEnum):
    RHYME = "rhyme"
    TOPICAL = "topical"


class ProjectStatus(StrEnum):
    CREATED = "created"
    STYLED = "styled"  # style contract done
    SCRIPTED = "scripted"  # scene manifest done
    GENERATING = "generating"
    QA = "qa"
    EDITING = "editing"
    REVIEW = "review"  # awaiting human approval
    PUBLISHED = "published"
    FAILED = "failed"
    BUDGET_ABORT = "budget_abort"


class CharacterRef(BaseModel):
    name: str
    description: str  # visual description used in every scene prompt
    image_uri: str | None = None  # storage URI of reference still


class StyleContract(BaseModel):
    language: str = "en"  # "en" | "hi" | "hi-en" (bilingual); drives lyrics, TTS voice, captions
    art_style: str  # e.g. "soft 2D cutout animation, thick outlines"
    palette: list[str]  # hex colors
    mood: str
    tempo_bpm: int
    aspect_ratio: str = "9:16"
    resolution: str = "1080x1920"
    characters: list[CharacterRef]
    pacing_rules: str  # prose rules the Director must obey
    negative_terms: list[str]  # things that must never appear (fed to video prompts)


class MusicCue(BaseModel):
    description: str
    start_s: float
    end_s: float


class Scene(BaseModel):
    index: int
    duration_s: float = Field(ge=3, le=10)
    video_prompt: str  # style-injected, self-contained
    narration: str  # text spoken/sung over this scene ("" if none)
    transition: str = "crossfade"  # crossfade | cut


class SceneManifest(BaseModel):
    title: str
    description: str  # YouTube description
    tags: list[str]
    total_duration_s: float  # 30-60
    lyrics: str | None = None  # full lyric sheet, rhyme mode only
    music_cues: list[MusicCue]
    scenes: list[Scene]


class AssetKind(StrEnum):
    CHAR_IMAGE = "char_image"
    SCENE_VIDEO = "scene_video"
    AUDIO_MASTER = "audio_master"
    FINAL_VIDEO = "final_video"
    THUMBNAIL = "thumbnail"


class AssetRef(BaseModel):
    kind: AssetKind
    scene_index: int | None = None
    attempt: int = 1
    uri: str  # storage URI
    content_hash: str
    cost_usd: float = 0.0
    meta: dict[str, str] = {}


class QAVerdict(StrEnum):
    PASS = "pass"
    RETRY = "retry"
    HUMAN_REVIEW = "human_review"


class QAReport(BaseModel):
    scene_index: int | None  # None = audio or final-cut report
    verdict: QAVerdict
    scores: dict[str, float]  # rubric_criterion -> 0..1
    safety_flags: list[str]
    critique: str  # actionable critique used to augment retry prompt


class CostEntry(BaseModel):
    at: datetime
    item: str  # e.g. "video:scene3:attempt2"
    provider: str
    cost_usd: float


class VideoProject(BaseModel):
    """The single LangGraph state object. Nodes receive and return this."""

    project_id: str
    mode: Mode
    brief: str  # human/theme input, e.g. "counting to five with ducks"
    status: ProjectStatus = ProjectStatus.CREATED
    style: StyleContract | None = None
    manifest: SceneManifest | None = None
    # SPEC-GAP: assets/cost_ledger use Annotated + operator.add reducers (spec's
    # literal block shows plain `list[...] = []`) because §1.4 fans char_refs out
    # into parallel videographer/music_director branches that both append to
    # these two fields in the same LangGraph superstep; without a reducer
    # LangGraph raises InvalidUpdateError ("can receive only one value per
    # step") on the concurrent write. qa_reports/retry_counts are written by
    # critic alone, never concurrently, so they stay plain.
    assets: Annotated[list[AssetRef], operator.add] = []
    qa_reports: list[QAReport] = []
    retry_counts: dict[int, int] = {}  # scene_index -> attempts used
    cost_ledger: Annotated[list[CostEntry], operator.add] = []
    budget_cap_usd: float = 12.0
    error: str | None = None

    @property
    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self.cost_ledger)
