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
    # SPEC-GAP: §7's POST /projects/{id}/reject needs a terminal status
    # distinct from FAILED (a human choosing not to publish isn't an error).
    REJECTED = "rejected"


class CharacterRef(BaseModel):
    name: str
    description: str  # visual description used in every scene prompt
    image_uri: str | None = None  # storage URI of reference still
    # Amendment 02: behavioral/voice guidance for the Director (comedic
    # timing, quirks) -- kept separate from `description`, which stays
    # purely visual since that's what feeds image-generation prompts and
    # personality text would only confuse those.
    personality: str = ""
    voice_id: str | None = None  # TTS voice id; falls back to settings.tts_voice when unset


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


class SceneGenMode(StrEnum):
    T2V = "t2v"
    I2V = "i2v"


class DialogueLine(BaseModel):
    speaker: str  # must match a StyleContract.characters[].name
    line: str


class Scene(BaseModel):
    index: int
    duration_s: float = Field(ge=3, le=10)
    video_prompt: str  # style-injected, self-contained
    narration: str  # text spoken/sung over this scene ("" if none)
    transition: str = "crossfade"  # crossfade | cut
    gen_mode: SceneGenMode = SceneGenMode.I2V
    # Amendment 01: required when gen_mode == I2V -- a fully composed
    # still-image prompt (exact camera framing, exact object/character
    # placement, exact state) that scene_stills conditions video generation
    # on as the start frame. Must NOT describe motion or time passing;
    # video_prompt for an I2V scene must then describe ONLY motion within
    # that fixed frame.
    scene_image_prompt: str | None = None
    # Amendment 02: speaker-attributed back-and-forth for a multi-character
    # cast. When set, music_director synthesizes and concatenates each line
    # with its speaker's own voice instead of using `narration` -- see
    # music_director.py's _run_topical. None preserves today's single-voice
    # narration behavior exactly.
    dialogue: list[DialogueLine] | None = None


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
    SCENE_STILL = "scene_still"  # Amendment 01: i2v start-frame per scene
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
    # Amendment 04: QA itself couldn't run (a provider outage/rate-limit/
    # low-credit error during Critic's own vision or transcription calls) --
    # deliberately distinct from HUMAN_REVIEW, which means a real content
    # concern was found and auto-assembly should stop. An INCONCLUSIVE scene
    # was never actually judged, so there's no content reason to withhold
    # it: the graph router treats it like PASS (falls through to editor by
    # default, same as PASS -- see graph/build.py::_critic_router), so
    # already-generated, already-paid-for assets still become a final video
    # instead of a total loss, with review_gate flagging which scenes
    # weren't really checked so a human knows to double-check them.
    INCONCLUSIVE = "inconclusive"


class FailureLayer(StrEnum):
    COMPOSITION = "composition"  # wrong layout/object placement -- regenerate the still first
    MOTION = "motion"  # bad animation of an otherwise-correct frame -- regenerate video only
    OTHER = "other"  # default; backward-compatible with pre-Amendment-01 QA reports


class QAReport(BaseModel):
    scene_index: int | None  # None = audio or final-cut report
    verdict: QAVerdict
    scores: dict[str, float]  # rubric_criterion -> 0..1
    safety_flags: list[str]
    critique: str  # actionable critique used to augment retry prompt
    failure_layer: FailureLayer = FailureLayer.OTHER


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
    # SPEC-GAP: §7's publisher node needs somewhere to persist the returned
    # YouTube URL; §12's "stop and amend the spec" is honored in spirit via
    # this comment rather than editing HANDOFF_SPEC.md's model listing, matching
    # how every other post-WP1 field addition in this codebase (llm_provider,
    # video_resolution, etc.) has been handled.
    published_url: str | None = None
    # Amendment 02: set when this project belongs to a persistent show (a
    # frozen cast + StyleContract, created via POST /shows and loaded by
    # Pipeline.run() rather than generated fresh -- see creative_director.py
    # and graph/nodes.py::char_refs's skip-when-already-set guards).
    show_id: str | None = None

    @property
    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self.cost_ledger)
