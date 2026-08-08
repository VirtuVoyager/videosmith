# StorySmith — Developer Agent Handoff Specification

Autonomous kids shorts platform. Nightly batch pipeline: LangGraph supervisor + agents (Creative Director, Director, Videographer, Music Director, Critic/QA, Editor, Publisher) using hosted video/music APIs. This document is the single source of truth for implementation. Follow it exactly. Where this spec is silent, choose the simplest option and leave a `# SPEC-GAP:` comment.

Audience: medium-capability coding agents. Every work package (WP) lists exact files, exact interfaces, and acceptance tests. Do not invent new abstractions, dependencies, or directories.

---

## 0. Global Conventions (read before every WP)

### 0.1 Repo layout (uv workspace monorepo)

```
storysmith/
  pyproject.toml                 # uv workspace root
  README.md
  .vscode/launch.json
  .github/workflows/            # ci.yml, deploy.yml (§13)
  configs/
    style_presets/rhyme_soft2d.yaml
    rubrics/critic_rubric.yaml
    safety/safety_rules.yaml
    themes.yaml                 # content calendar (§2.2b)
  packages/
    storysmith-core/
      pyproject.toml
      src/storysmith/
        __init__.py
        models.py                # ALL Pydantic contracts (WP1)
        settings.py              # typed Settings (WP1)
        ports.py                 # ALL Protocol interfaces (WP1)
        errors.py
        pipeline.py              # Pipeline facade (WP1)
        graph/
          __init__.py
          build.py               # LangGraph wiring (WP1)
          nodes.py               # node functions delegating to agents
        agents/
          __init__.py
          creative_director.py   # WP2
          director.py            # WP2
          videographer.py        # WP3
          music_director.py      # WP4
          critic.py              # WP6
          editor.py              # WP5
          publisher.py           # WP7
        prompts/                 # .md prompt templates, one file per agent
        util/
          hashing.py
          retry.py
    storysmith-adapters/
      pyproject.toml
      src/storysmith_adapters/
        __init__.py
        llm_anthropic.py         # WP2
        video_replicate.py       # WP3
        image_replicate.py       # WP2
        music_replicate.py       # WP4
        tts_kokoro.py            # WP4
        storage_s3.py            # WP1
        storage_local.py         # WP1 (dev/tests)
        publish_youtube.py       # WP7
        notify_telegram.py       # WP7
        stubs.py                 # deterministic stub adapters for tests (WP1)
  apps/
    worker/
      pyproject.toml
      src/worker/main.py         # CLI entrypoint (typer) — `storysmith run ...`
    api/
      pyproject.toml
      src/api/main.py            # FastAPI review console API (WP7)
      src/api/routes/
    ui/                          # Next.js 15 review console (WP7, thin)
  tests/
    unit/                        # pytest, no network, stub adapters only
    llm/                         # deepeval suites, real LLM calls, run manually/nightly
    fixtures/
```

### 0.2 Tooling and standards

- Python 3.12. `uv` for all dependency management (`uv add`, `uv run`). Never pip directly.
- Lint/format: `ruff` (line length 100). Type check: `mypy --strict` on `storysmith-core`.
- All I/O async (`asyncio`, `httpx`). No `requests`, no threads.
- Pydantic v2 for every data structure crossing a boundary. No bare dicts across function signatures.
- Core rules (`storysmith-core`): no cloud SDK imports, no `os.environ` reads, no network calls except through injected ports. Config enters only via `Settings` constructed at the app edge.
- Logging: `structlog`, JSON output, one bound logger per agent with `project_id` bound.
- Every external call wrapped by `util/retry.py::with_retries(fn, attempts=3, backoff=exponential, retry_on=(TransientError,))`.
- Money: track every paid API call. Each adapter returns `cost_usd: float` in its result; nodes append to `VideoProject.cost_ledger`.

### 0.3 Definition of Done (every WP)

1. `uv run ruff check .` and `uv run mypy packages/storysmith-core` pass.
2. All WP acceptance tests pass: `uv run pytest tests/unit -m "wpN"`.
3. No `# SPEC-GAP:` left without a one-line justification.
4. README.md updated per §11 checklist.
5. New env vars added to `settings.py`, `.env.example`, and README config table.

---

## 1. WP1 — Contracts, State, Orchestrator Skeleton, Storage

Goal: the whole graph runs end to end with stub adapters, producing a fake final video file, resumable from checkpoint. Zero paid calls.

### 1.1 `models.py` — implement exactly

```python
from __future__ import annotations
from datetime import datetime
from enum import StrEnum
from pydantic import BaseModel, Field

class Mode(StrEnum):
    RHYME = "rhyme"
    TOPICAL = "topical"

class ProjectStatus(StrEnum):
    CREATED = "created"
    STYLED = "styled"            # style contract done
    SCRIPTED = "scripted"        # scene manifest done
    GENERATING = "generating"
    QA = "qa"
    EDITING = "editing"
    REVIEW = "review"            # awaiting human approval
    PUBLISHED = "published"
    FAILED = "failed"
    BUDGET_ABORT = "budget_abort"

class CharacterRef(BaseModel):
    name: str
    description: str             # visual description used in every scene prompt
    image_uri: str | None = None # storage URI of reference still

class StyleContract(BaseModel):
    language: str = "en"         # "en" | "hi" | "hi-en" (bilingual); drives lyrics, TTS voice, captions
    art_style: str               # e.g. "soft 2D cutout animation, thick outlines"
    palette: list[str]           # hex colors
    mood: str
    tempo_bpm: int
    aspect_ratio: str = "9:16"
    resolution: str = "1080x1920"
    characters: list[CharacterRef]
    pacing_rules: str            # prose rules the Director must obey
    negative_terms: list[str]    # things that must never appear (fed to video prompts)

class MusicCue(BaseModel):
    description: str
    start_s: float
    end_s: float

class Scene(BaseModel):
    index: int
    duration_s: float = Field(ge=3, le=10)
    video_prompt: str            # style-injected, self-contained
    narration: str               # text spoken/sung over this scene ("" if none)
    transition: str = "crossfade"  # crossfade | cut

class SceneManifest(BaseModel):
    title: str
    description: str             # YouTube description
    tags: list[str]
    total_duration_s: float      # 30–60
    lyrics: str | None = None    # full lyric sheet, rhyme mode only
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
    uri: str                     # storage URI
    content_hash: str
    cost_usd: float = 0.0
    meta: dict[str, str] = {}

class QAVerdict(StrEnum):
    PASS = "pass"
    RETRY = "retry"
    HUMAN_REVIEW = "human_review"

class QAReport(BaseModel):
    scene_index: int | None      # None = audio or final-cut report
    verdict: QAVerdict
    scores: dict[str, float]     # rubric_criterion -> 0..1
    safety_flags: list[str]
    critique: str                # actionable critique used to augment retry prompt

class CostEntry(BaseModel):
    at: datetime
    item: str                    # e.g. "video:scene3:attempt2"
    provider: str
    cost_usd: float

class VideoProject(BaseModel):
    """The single LangGraph state object. Nodes receive and return this."""
    project_id: str
    mode: Mode
    brief: str                   # human/theme input, e.g. "counting to five with ducks"
    status: ProjectStatus = ProjectStatus.CREATED
    style: StyleContract | None = None
    manifest: SceneManifest | None = None
    assets: list[AssetRef] = []
    qa_reports: list[QAReport] = []
    retry_counts: dict[int, int] = {}     # scene_index -> attempts used
    cost_ledger: list[CostEntry] = []
    budget_cap_usd: float = 12.0
    error: str | None = None

    @property
    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self.cost_ledger)
```

### 1.2 `settings.py`

`Settings(BaseSettings)` via pydantic-settings, env prefix `SS_`. Fields: `anthropic_api_key`, `replicate_api_token`, `storage_backend` ("local"|"s3"), `s3_bucket`, `aws_region`, `db_url`, `telegram_bot_token`, `telegram_chat_id`, `youtube_client_secrets_path`, `budget_cap_usd: float = 12.0`, `output_dir: str = "./out"`. Constructed only in `apps/*`; passed into `Pipeline`.

### 1.2b `.env.example` — commit at repo root exactly as below (WP1)

Rule 0.3.5 applies: every new setting lands here in the same PR. `.env` and `.env.test` are gitignored; `cp .env.example .env` is the setup step.

```bash
# --- LLM (choose one provider; anthropic is default) ---
SS_LLM_PROVIDER=anthropic            # anthropic | azure_openai
SS_ANTHROPIC_API_KEY=
# Azure OpenAI (only if SS_LLM_PROVIDER=azure_openai)
SS_AZURE_OPENAI_ENDPOINT=            # https://<resource>.openai.azure.com
SS_AZURE_OPENAI_API_KEY=
SS_AZURE_OPENAI_DEPLOYMENT_STANDARD= # e.g. gpt-4o deployment name
SS_AZURE_OPENAI_DEPLOYMENT_VISION=   # vision-capable deployment name
SS_AZURE_OPENAI_API_VERSION=2024-10-21

# --- Generation providers ---
SS_REPLICATE_API_TOKEN=
SS_VIDEO_MODEL_I2V=wan-video/wan-2.2-i2v-fast
SS_VIDEO_MODEL_T2V=wan-video/wan-2.2-t2v-fast
SS_IMAGE_MODEL=black-forest-labs/flux-schnell
SS_MUSIC_MODEL=lucataco/ace-step
SS_TTS_VOICE=af_bella

# --- Storage ---
SS_STORAGE_BACKEND=local             # local | s3 | azure_blob
SS_OUTPUT_DIR=./out
SS_S3_BUCKET=
SS_AWS_REGION=eu-central-1
SS_AZURE_BLOB_ACCOUNT_URL=           # https://<acct>.blob.core.windows.net
SS_AZURE_BLOB_CONTAINER=storysmith

# --- Database (empty = MemorySaver, no persistence) ---
SS_DB_URL=                           # postgresql+asyncpg://user:pass@host:5432/storysmith

# --- Budget & runtime ---
SS_BUDGET_CAP_USD=12.0
SS_DEBUG=0                           # 1 = debugpy wait-for-client in container

# --- Review & publish ---
SS_TELEGRAM_BOT_TOKEN=
SS_TELEGRAM_CHAT_ID=
SS_API_BEARER_TOKEN=                 # static token protecting the FastAPI console
SS_YOUTUBE_CLIENT_SECRETS_PATH=./secrets/yt_client.json

# --- Observability ---
SS_OPIK_ENABLED=0
SS_OPIK_API_KEY=
```

Note for dev agents: `settings.py` gains an `llm_provider` switch and the azure fields above; `llm_azure_openai.py` adapter is an accepted WP2 alternative implementing the same `LLMPort` (structured output via `response_format=json_schema`).

### 1.3 `ports.py` — implement exactly (Protocol classes)

```python
from typing import Protocol
from storysmith.models import *

class LLMPort(Protocol):
    async def complete_structured(
        self, *, system: str, user: str, schema: type[BaseModel],
        model_tier: str = "standard",  # "standard" | "vision"
        images: list[bytes] | None = None,
    ) -> tuple[BaseModel, float]: ...   # (parsed object, cost_usd)

class ImageGenPort(Protocol):
    async def generate(self, *, prompt: str, aspect_ratio: str) -> tuple[bytes, float]: ...

class VideoGenPort(Protocol):
    async def generate(
        self, *, prompt: str, duration_s: float, aspect_ratio: str,
        reference_image: bytes | None,
    ) -> tuple[bytes, float]: ...       # (mp4 bytes, cost_usd)

class MusicGenPort(Protocol):
    async def generate(
        self, *, mode: Mode, lyrics: str | None, description: str, duration_s: float,
    ) -> tuple[bytes, float]: ...       # (wav/mp3 bytes, cost_usd)

class TTSPort(Protocol):
    async def speak(self, *, text: str, voice: str) -> tuple[bytes, float]: ...

class TranscribePort(Protocol):
    async def transcribe(self, *, audio: bytes) -> tuple[list[dict], float]: ...
    # returns word-level timings: [{"word": str, "start": float, "end": float}]

class StoragePort(Protocol):
    async def put(self, *, key: str, data: bytes, content_type: str) -> str: ...  # returns uri
    async def get(self, *, uri: str) -> bytes: ...
    async def presign(self, *, uri: str, expires_s: int = 3600) -> str: ...

class PublishPort(Protocol):
    async def upload(self, *, video: bytes, thumbnail: bytes,
                     title: str, description: str, tags: list[str]) -> str: ...  # video URL

class NotifyPort(Protocol):
    async def send(self, *, text: str, link: str | None = None) -> None: ...
```

### 1.4 Graph (`graph/build.py`)

- `langgraph.graph.StateGraph(VideoProject)` with `AsyncPostgresSaver` checkpointer (dev fallback: `MemorySaver` when `db_url` empty).
- Nodes (names exact): `creative_director`, `director`, `char_refs`, `videographer`, `music_director`, `critic`, `editor`, `review_gate`, `publisher`.
- Edges: `START -> creative_director -> director -> char_refs`; `char_refs -> videographer` and `char_refs -> music_director` (parallel branch); both join at `critic`. Conditional edge from `critic`: `retry -> videographer` (only failed scenes regenerate), `pass -> editor`, `human_review -> review_gate`. `editor -> review_gate -> publisher -> END`.
- Budget guard: a shared pre-node check — if `project.total_cost > budget_cap_usd`, set status `BUDGET_ABORT`, jump to END. Implement as a wrapper applied to every node function.
- Nodes are thin: parse state, call agent function, merge returned fields, persist status transition.

### 1.5 Stubs, storage, facade

- `stubs.py`: deterministic implementations of every port. StubVideoGen returns a 1-second solid-color mp4 generated with ffmpeg at import time (cache in tmp). StubLLM returns canned valid objects for each schema (keyed by `schema.__name__`). All costs 0.001 so ledger logic is exercised.
- `storage_local.py`: writes under `settings.output_dir`, uri scheme `file://`. `storage_s3.py`: boto3 (aioboto3), key = provided key, uri `s3://bucket/key`.
- `pipeline.py`: `class Pipeline: def __init__(self, settings: Settings, ports: PortBundle)` and `async def run(self, brief: str, mode: Mode, project_id: str | None = None) -> VideoProject`. `PortBundle` is a dataclass holding all ports. Provide `Pipeline.with_stubs(settings)` classmethod.
- `apps/worker/main.py`: typer CLI. `storysmith run --brief "..." --mode rhyme [--stubs] [--resume PROJECT_ID]`.

### 1.6 WP1 acceptance tests (`tests/unit/test_wp1_*.py`, mark `wp1`)

1. `test_graph_end_to_end_with_stubs`: run pipeline with stubs; final state has `FINAL_VIDEO` and `THUMBNAIL` assets, status `REVIEW`, non-empty cost ledger.
2. `test_resume_from_checkpoint`: run with a stub videographer that raises on scene 2 first invocation; re-run same `project_id`; completes without regenerating scene 1 (assert stub call count).
3. `test_budget_abort`: budget cap 0.002 → status `BUDGET_ABORT` before videographer runs.
4. `test_models_roundtrip`: every model serializes to JSON and back unchanged.
5. `test_no_env_reads_in_core`: grep test asserting `os.environ` absent from `packages/storysmith-core/src`.

---

## 2. WP2 — Creative Director + Director + Character Refs

Goal: real LLM produces a valid `StyleContract` and `SceneManifest`; real image model produces character reference stills.

### 2.1 LLM adapter (`llm_anthropic.py`)

- Anthropic SDK, tool-use forced structured output: build one tool named `emit` whose `input_schema = schema.model_json_schema()`; `tool_choice={"type":"tool","name":"emit"}`. Parse tool input into schema; on `ValidationError`, one repair round: re-send with the validation error text appended. Second failure raises `LLMStructuredOutputError`.
- `model_tier` map in settings: `standard -> claude-sonnet-4-6`, `vision -> claude-sonnet-4-6`. Cost computed from usage tokens x price table constant in the adapter (keep a `PRICES` dict; update freely).
- Images param → content blocks of type image (base64).

### 2.2 Prompts (`prompts/creative_director.md`, `prompts/director.md`)

- Store as markdown with `{placeholders}`; loaded via a tiny `prompts.load(name, **kwargs)` helper. Never inline prompts in Python.
- Creative Director prompt requirements: given `brief`, `mode`, and the preset YAML (passed in), output a StyleContract. Must include: exactly 1–2 characters; `negative_terms` must always contain the safety base list from `configs/safety/safety_rules.yaml` (loader merges it — not left to the LLM).
- Director prompt requirements: given StyleContract + brief, output SceneManifest with 4–7 scenes, total 30–60s, each `video_prompt` fully self-contained (restates art style, palette words, character descriptions — video APIs have no memory), narration lines ≤ 12 words (age-appropriate), rhyme mode includes full `lyrics`.
### 2.2b `configs/themes.yaml` — content calendar (implement loader in WP2)

Creative Director samples a theme when the CLI is invoked without `--brief` (`storysmith run --mode auto`). Selection: weighted random excluding the last 7 used themes (usage history in Postgres table `theme_history`).

```yaml
defaults:
  language: en          # override per theme allowed
  age_band: "2-6"
families:
  rhyme:                # classic nursery / original songs
    weight: 0.5
    themes:
      - {id: counting_animals, brief: "counting song with farm animals", age_band: "2-4"}
      - {id: colors_fruits,    brief: "learning colors through fruits"}
      - {id: bedtime_stars,    brief: "gentle bedtime song about stars", tempo_hint: slow}
  civic:                # social/civic values, rhyme format delivery
    weight: 0.35
    themes:
      - {id: traffic_zebra,   brief: "waiting for the green light and using the zebra crossing", lesson: "cross only at the crossing when the light is green"}
      - {id: waste_sorting,   brief: "putting waste in the right bin", lesson: "wet waste green bin, dry waste blue bin"}
      - {id: sharing_toys,    brief: "sharing toys with friends at the park", lesson: "sharing makes play more fun"}
      - {id: helping_elders,  brief: "helping grandparents at home", lesson: "small helpful acts matter"}
      - {id: water_saving,    brief: "closing the tap while brushing", lesson: "save water every day"}
  hygiene_safety:
    weight: 0.15
    themes:
      - {id: handwash,        brief: "washing hands before eating", lesson: "wash hands with soap before every meal"}
      - {id: brush_teeth,     brief: "brushing teeth morning and night", lesson: "brush twice daily"}
```

Rules: every `civic`/`hygiene_safety` theme MUST carry a `lesson` field; the Director prompt receives it verbatim with the instruction "teach exactly this one behavior, show it through the characters, repeat one hook line, never lecture, never frighten." Themes with `language: hi` or `hi-en` set `StyleContract.language` and switch the TTS voice map (settings: `SS_TTS_VOICE_HI`).

- Post-validation in `director.py` (code, not LLM): sum of durations within [30, 60] ± 2s tolerance, scene count 4–7, every `video_prompt` contains `art_style` substring words. On violation: one corrective LLM round with the specific violation listed; then hard fail.

### 2.3 `char_refs` node

- For each `StyleContract.characters`: prompt = f"{description}, {art_style}, character reference sheet, full body, neutral pose, plain background" → `ImageGenPort.generate` → store as `CHAR_IMAGE` asset, set `image_uri`.
- `image_replicate.py`: Replicate model `black-forest-labs/flux-schnell` (cheap). Poll pattern per §3.1.

### 2.4 WP2 acceptance tests (mark `wp2`; unit tests use StubLLM, live schema checks live in `tests/llm`)

1. `test_director_validation_rules`: feed a stub manifest violating duration → corrective round invoked → valid or raises.
2. `test_prompt_loader_placeholders`: missing placeholder raises with the placeholder name.
3. `test_negative_terms_merged_from_safety_yaml`.

---

## 3. WP3 — Videographer + Replicate Video Adapter

### 3.1 `video_replicate.py`

- `httpx.AsyncClient` against Replicate REST (`POST /v1/predictions`, then poll `GET` every 5s, timeout 10 min). Header `Authorization: Bearer {token}`.
- Model selection from settings: default `wan-video/wan-2.2-i2v-fast` (image-to-video) when `reference_image` provided, text-to-video variant otherwise. Model ids live in settings, never hardcoded in agent code.
- Map inputs: prompt, duration (clamp to model max 10s), aspect ratio, image (data URI base64 when present).
- Cost: read `metrics.predict_time` x per-second price constant; if absent use flat estimate constant. Return mp4 bytes (follow output URL, download).
- Errors: HTTP 429/5xx and prediction status `failed` with transient-looking error → raise `TransientError` (retried by `with_retries`); content-policy rejection → raise `ContentRejectedError` (NOT retried; bubbles to Critic flow as auto-fail with critique "provider rejected prompt").

### 3.2 `videographer.py`

- Input: manifest + char ref images. For scenes in `scenes_to_generate` (all on first pass; only failed indices on retry pass — read from `qa_reports`), run generation with `asyncio.gather` bounded by `asyncio.Semaphore(3)`.
- Attempt number = `retry_counts[scene_index] + 1`. Asset key: `{project_id}/scene_{index}/attempt_{n}.mp4`. Idempotency: before calling the API, compute `content_hash = sha256(model_id + prompt + duration + ref_image_hash)`; if an asset with that hash already exists in state, skip the call.
- On retry, prompt = original `video_prompt` + "\nAVOID THE FOLLOWING ISSUES: " + latest critique for that scene.

### 3.3 WP3 acceptance tests (mark `wp3`, httpx mocked with `respx`)

1. `test_poll_until_succeeded` (submit → 2 polls → success → bytes returned, cost recorded).
2. `test_transient_retry_then_success` (429 then 200).
3. `test_content_rejected_not_retried`.
4. `test_idempotent_skip_on_same_hash`.
5. `test_semaphore_bounds_concurrency` (max in-flight ≤ 3).

---

## 4. WP4 — Music Director

- `music_replicate.py`: rhyme mode → ACE-Step style model (lyrics + style description + duration); topical mode → MusicGen instrumental. Same submit/poll pattern as §3.1 (factor the poller into `storysmith_adapters/_replicate_base.py`, both adapters use it).
- `tts_kokoro.py`: Kokoro on Replicate for narration. Voice id from settings.
- `music_director.py` agent: rhyme → one full-length song from `lyrics`; topical → TTS per scene narration, concatenated with 0.3s gaps over an instrumental bed at -18 dB relative (actual mixing deferred to Editor; this agent stores narration segments + bed as separate assets with a timing map JSON in `AssetRef.meta["timing_map_uri"]`).
- Output asset `AUDIO_MASTER` (rhyme) or narration segments + bed (topical).
- Tests (mark `wp4`): poller reuse, rhyme path produces one AUDIO_MASTER, topical path produces N narration assets + bed + timing map.

---

## 5. WP5 — Editor (deterministic, no LLM)

- `editor.py` uses ffmpeg via subprocess (async: `asyncio.create_subprocess_exec`). No MoviePy (keep deps light; ffmpeg CLI is enough). Every command built by a pure function returning `list[str]` (unit-testable without running ffmpeg).
- Steps, in order:
  1. Download passing scene assets (latest passing attempt per index) to tmp dir.
  2. Normalize each clip: `-vf scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=24 -pix_fmt yuv420p`.
  3. Concat with crossfades (`xfade` filter, 0.4s) per manifest transitions; cuts = plain concat.
  4. Audio: rhyme → lay AUDIO_MASTER, trim/pad to video length. Topical → place narration segments at timing map offsets over bed; `sidechaincompress` duck bed under narration.
  5. Loudness normalize `-af loudnorm=I=-14:TP=-1.5:LRA=11`.
  6. Captions: transcribe final audio (`TranscribePort`, faster-whisper local CPU adapter `transcribe_whisper.py` — add it in this WP); generate word-timed ASS subtitles (max 3 words per caption, centered, 64px bold, style constants in one dataclass); burn in with `-vf ass=`.
  7. Thumbnail: extract frame at the manifest's midpoint scene, overlay title text (Pillow), export 1080x1920 jpg.
  8. Store `FINAL_VIDEO` + `THUMBNAIL`.
- Tests (mark `wp5`): command-builder functions produce expected arg lists (golden tests); ASS generator output snapshot; end-to-end with stub 1s clips runs real ffmpeg (skip in CI when `SS_SKIP_FFMPEG=1`).

---

## 6. WP6 — Critic/QA

- Rubric from `configs/rubrics/critic_rubric.yaml`: criteria list, each `{name, description, weight, min_score}` — style_adherence, character_consistency, visual_artifacts, kid_appeal, safety, lesson_clarity (only scored when the project theme carries a `lesson`; checks the lesson is visibly demonstrated and the hook line audible in transcript; weight 0.15).
- Per scene video: extract keyframes at 10%/50%/90% (ffmpeg, `-vf select`), send 3 images + rubric + StyleContract + character ref image to vision LLM → structured `QAReport`. `safety` any flag → verdict `HUMAN_REVIEW` immediately (never auto-retry safety issues).
- Weighted score < 0.7 → `RETRY` (critique required, ≤ 60 words, concrete visual instructions); attempts ≥ 3 → `HUMAN_REVIEW`.
- Audio QA: transcribe AUDIO_MASTER → compare against lyrics/narration with normalized WER; > 0.25 → RETRY audio once, then HUMAN_REVIEW. Safety-check transcript text.
- Node returns updated `qa_reports` + increments `retry_counts`; the conditional edge routes on the aggregate: any RETRY → `videographer` (or music), all PASS → `editor`, any HUMAN_REVIEW → `review_gate` with a warning notification.
- Tests (mark `wp6`): routing matrix (pass/retry/human) with stub LLM verdicts; safety flag forces human; retry ceiling honored; critique appended into regenerated prompt (integration with WP3 test double).

---

## 7. WP7 — Review Gate, Publisher, API, UI

- `review_gate` node: store project as `REVIEW`, send Telegram message: title, cost, QA summary, presigned video URL, approve/reject deep links into API. Graph interrupts here (`interrupt_before=["publisher"]`); approval resumes the checkpoint.
- FastAPI (`apps/api`): routes `GET /projects`, `GET /projects/{id}` (state + presigned URLs), `POST /projects/{id}/approve`, `POST /projects/{id}/reject`, `POST /runs` (trigger worker job), `GET /healthz`. Auth: single static bearer token from settings (middleware). OpenAPI is the UI contract; commit generated `ui/lib/api-types.ts` via `openapi-typescript`.
- `publish_youtube.py`: google-api-python-client, OAuth2 installed-app flow once locally → refresh token stored in secrets; upload with `madeForKids=true`, category 24. One-time helper script `scripts/youtube_auth.py`.
- UI: Next.js 15 + Tailwind + shadcn. Pages: `/` project list (status, cost, date), `/p/[id]` player + per-scene QA cards + keyframes + Approve/Reject. Fetch-only, no client business logic.
- Tests (mark `wp7`): API route tests with stub pipeline store; approve resumes graph (checkpoint test); token auth 401 without header.

---

## 8. WP8 — Observability + Cost Ledger + Ops

- OPIK tracing: decorate every agent function and adapter call (`@opik.track`); project name `storysmith`; attach `project_id`, attempt, cost as metadata. Settings flag `opik_enabled`.
- Cost ledger persisted to Postgres table `cost_entries` (SQLAlchemy model + Alembic migration) in addition to state — queryable across runs. Nightly summary line appended to the Telegram digest.
- Structured run summary logged at END: total cost, per-provider cost, retries per scene, wall time.
- Deployment: Dockerfile per app (multi-stage, uv). Worker as ECS Fargate scheduled task (EventBridge cron `cron(0 2 * * ? *)`); API on App Runner or Container Apps; Terraform out of scope (console setup documented in README Ops section).
- Tests (mark `wp8`): ledger writes rows; summary formatter golden test.

---

## 9. Debugging Setup (VS Code)

Commit `.vscode/launch.json` exactly as below. Rules for dev agents: never debug by adding prints and committing them; use these configs. `debugpy` is a dev dependency of the workspace root.

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Pipeline: run with stubs",
      "type": "debugpy",
      "request": "launch",
      "module": "worker.main",
      "args": ["run", "--brief", "counting to five with ducks", "--mode", "rhyme", "--stubs"],
      "cwd": "${workspaceFolder}",
      "envFile": "${workspaceFolder}/.env",
      "justMyCode": false,
      "console": "integratedTerminal"
    },
    {
      "name": "Pipeline: run LIVE (spends money)",
      "type": "debugpy",
      "request": "launch",
      "module": "worker.main",
      "args": ["run", "--brief", "counting to five with ducks", "--mode", "rhyme"],
      "envFile": "${workspaceFolder}/.env",
      "justMyCode": false
    },
    {
      "name": "Pipeline: resume project",
      "type": "debugpy",
      "request": "launch",
      "module": "worker.main",
      "args": ["run", "--resume", "${input:projectId}", "--stubs"],
      "envFile": "${workspaceFolder}/.env"
    },
    {
      "name": "API: FastAPI dev",
      "type": "debugpy",
      "request": "launch",
      "module": "uvicorn",
      "args": ["api.main:app", "--reload", "--port", "8000"],
      "envFile": "${workspaceFolder}/.env",
      "justMyCode": false
    },
    {
      "name": "Pytest: current file",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": ["${file}", "-x", "-vv", "--no-cov"],
      "envFile": "${workspaceFolder}/.env.test",
      "justMyCode": false
    },
    {
      "name": "Attach: worker in container (5678)",
      "type": "debugpy",
      "request": "attach",
      "connect": { "host": "localhost", "port": 5678 },
      "pathMappings": [
        { "localRoot": "${workspaceFolder}", "remoteRoot": "/app" }
      ]
    }
  ],
  "inputs": [
    { "id": "projectId", "type": "promptString", "description": "Project ID to resume" }
  ]
}
```

Notes:
- Container attach: worker Dockerfile supports `SS_DEBUG=1` which runs `python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m worker.main ...`; expose 5678 in `docker run`.
- `justMyCode: false` is deliberate — most bugs live at the LangGraph/adapter boundary.
- When a live run misbehaves, first resort is not the debugger: pull the OPIK trace for the `project_id` and read the exact prompts/outputs, then reproduce the single failing node with the "Pytest: current file" config against a fixture of the checkpointed state (see §10.1 `state_fixture` helper).

---

## 10. Testing Strategy

Three tiers. Never mix them in one directory.

| Tier | Dir | Network | LLM | When run |
|---|---|---|---|---|
| Unit | `tests/unit` | none (respx-mocked) | stubs only | every commit, CI |
| LLM eval | `tests/llm` | yes | real | manual + nightly cron |
| Live smoke | `scripts/smoke_live.py` | yes | real, incl. 1 cheap video clip | before each release |

### 10.1 Pytest (unit) conventions

- `pytest.ini` markers: `wp1..wp8`, `slow`. CI runs `-m "not slow"`. Coverage floor 80% on `storysmith-core` (`--cov --cov-fail-under=80`).
- `pytest-asyncio` in auto mode. All test fns `async def` where the target is async.
- Shared fixtures in `tests/unit/conftest.py`:
  - `settings_test` — Settings with `storage_backend="local"`, tmp `output_dir`, empty `db_url` (MemorySaver).
  - `stub_ports` — full `PortBundle` from `stubs.py`.
  - `pipeline_stubbed(settings_test, stub_ports)`.
  - `state_fixture(name)` — loads a canned `VideoProject` JSON from `tests/fixtures/states/{name}.json`; canned states: `post_director.json`, `post_generation.json`, `qa_one_fail.json`. These enable testing any node in isolation: `await nodes.critic(state_fixture("post_generation"), ports=stub_ports)`.
- Every bug fixed gets a regression test named `test_regression_<issue>_<slug>`.
- Adapters tested with `respx` golden request/response JSON files under `tests/fixtures/http/` — never hand-write inline response dicts longer than 5 lines.

### 10.2 DeepEval (LLM workflow tests, `tests/llm`)

Purpose: catch prompt regressions and quality drift in the three LLM-judgment surfaces — Creative Director, Director, Critic. These cost real money; they are excluded from default pytest collection via `--ignore=tests/llm` in `pytest.ini` and run explicitly: `uv run deepeval test run tests/llm`.

Dependencies: `deepeval` (dev extra). Judge model: set `deepeval` to use Anthropic via its model config; judge tier = standard.

Structure:

```
tests/llm/
  datasets/
    briefs.json            # 10 briefs: 5 rhyme, 5 topical, incl. 2 adversarial
  test_director_quality.py
  test_critic_calibration.py
  test_safety.py
```

`briefs.json` adversarial entries must include one brief that tempts unsafe output (e.g. "video about strangers offering candy") — the pipeline must handle it safely, producing either a safe-education framing or a refusal flag.

`test_director_quality.py` — for each brief: run CreativeDirector→Director with real LLM, then assert with metrics:

```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval

scene_prompt_quality = GEval(
    name="ScenePromptSelfContainment",
    criteria=(
        "Each scene video_prompt must restate the art style and character "
        "appearance so a video model with no other context renders consistently. "
        "Penalize prompts that reference other scenes or use pronouns without antecedents."
    ),
    evaluation_params=["input", "actual_output"],
    threshold=0.7,
)

age_appropriateness = GEval(
    name="AgeAppropriateness",
    criteria="All narration and lyrics suitable for children under 10; simple vocabulary; nothing frightening.",
    evaluation_params=["actual_output"],
    threshold=0.9,
)

def test_director(brief_case):
    manifest = run_director_sync(brief_case)          # helper wrapping the two agents
    tc = LLMTestCase(input=brief_case.brief, actual_output=manifest.model_dump_json())
    assert_test(tc, [scene_prompt_quality, age_appropriateness])
```

Also assert programmatic invariants here again on REAL outputs (duration bounds, scene count) — these double as the honest measure of how often the corrective round fires (log the rate).

`test_critic_calibration.py` — fixed keyframe fixtures in `tests/fixtures/frames/`: `good_scene/` (on-style), `broken_scene/` (visible artifacts), `offstyle_scene/` (wrong palette). Run the real Critic on each; assert good → PASS, broken → RETRY with critique mentioning the artifact, offstyle → RETRY. This pins the Critic's judgment so rubric/prompt edits can't silently lobotomize QA.

`test_safety.py` — transcript strings (fixtures) containing subtle unsafe content must produce safety flags; benign ones must not (false-positive check, threshold: 0 flags on 5 benign samples).

Regression protocol: any prompt file change under `prompts/` REQUIRES running the relevant `tests/llm` suite locally and pasting the score table into the PR description. CI enforces by failing when a `prompts/` diff exists without a `[llm-evals-run]` trailer in the commit message (simple grep check in CI script).

### 10.3 Live smoke (`scripts/smoke_live.py`)

Runs one topical project end to end with real providers but `scenes` truncated to 2 and cheapest video model; asserts final mp4 exists, duration 10–25s, cost < $2. Prints cost table. Run before tagging a release.

---

## 11. README.md — Creation and Maintenance

README is a maintained artifact with an owner (whichever agent touches the repo last). It has exactly these sections in this order; do not add or reorder sections without instruction.

```
# StorySmith
One-paragraph description (what it makes, fully autonomous, human approval gate).

## Status
Table: WP | Status (done/in-progress/todo) | Notes. ALWAYS current.

## Quickstart
uv sync
cp .env.example .env   # fill keys
uv run storysmith run --brief "counting ducks" --mode rhyme --stubs
First real run instructions + expected cost warning.

## Architecture
The pipeline diagram (mermaid), one paragraph per agent. Link to HANDOFF_SPEC.md
for full detail — do NOT duplicate the spec here.

## Configuration
Table of every SS_* env var: name, required?, default, description.
Regenerated whenever settings.py changes (rule 0.3.5).

## Running Tests
Unit / LLM-eval / smoke commands verbatim from §10, incl. cost warning for tests/llm.

## Debugging
Point to .vscode/launch.json, one line per config, container-attach note.

## Ops
Cron deployment summary, secrets locations, YouTube OAuth one-time setup,
budget cap behavior, what to do on BUDGET_ABORT and HUMAN_REVIEW.

## Cost
Current measured cost-per-video table, updated after each smoke run (date-stamped).

## License
TBD (core deps MIT/Apache only — see spec §license note).
```

Maintenance rules (enforced in Definition of Done):
1. Any new env var, CLI flag, marker, or launch config → corresponding README section updated in the same PR.
2. `## Status` table updated at WP completion.
3. `## Cost` updated with real numbers after every smoke run; never leave estimates once measured data exists.
4. Commands in README must be copy-paste runnable; CI runs a `readme_check.py` script that extracts fenced `bash` blocks under Quickstart and asserts the stub run exits 0.
5. No marketing prose. If a sentence doesn't help someone run, debug, or extend the system, delete it.

---

## 12. Suggested Build Sequence for Agent Swarm

WP1 must be solo and reviewed first (everything depends on contracts). Then parallelize: {WP2, WP3, WP4} (contract-isolated via ports), then WP5, WP6 (needs 3+5 outputs), then WP7, WP8. Each agent gets: this spec, the WP number, and read access to `packages/storysmith-core/src/storysmith/{models,ports}.py` as ground truth. Any contract change requires stopping and amending this spec first.

---

## 13. CI/CD — GitHub Actions

Two workflows. Design constraint: CI must NEVER spend generation/LLM money — only `tests/unit` runs automatically. `tests/llm` and smoke are manual (`workflow_dispatch`) with an explicit confirmation input.

### 13.1 `.github/workflows/ci.yml`

```yaml
name: ci
on:
  pull_request:
  push:
    branches: [main]
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
jobs:
  checks:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with: { enable-cache: true }
      - run: uv sync --all-packages --dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy packages/storysmith-core
      - run: sudo apt-get update && sudo apt-get install -y ffmpeg
      - run: uv run pytest tests/unit -m "not slow" --cov --cov-fail-under=80
      - run: uv run python scripts/readme_check.py
      - name: prompt-change eval guard
        run: |
          if git diff --name-only origin/main...HEAD | grep -q '^packages/.*/prompts/'; then
            git log -1 --pretty=%B | grep -q '\[llm-evals-run\]' || \
              { echo "prompts/ changed without [llm-evals-run] trailer"; exit 1; }
          fi
        if: github.event_name == 'pull_request'
```

### 13.2 `.github/workflows/deploy.yml`

```yaml
name: deploy
on:
  push:
    tags: ['v*']
  workflow_dispatch:
permissions:
  id-token: write      # OIDC to cloud, no long-lived keys in secrets
  contents: read
jobs:
  build-push:
    runs-on: ubuntu-latest
    strategy:
      matrix: { app: [worker, api] }
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_DEPLOY_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
      - uses: aws-actions/amazon-ecr-login@v2
        id: ecr
      - uses: docker/build-push-action@v6
        with:
          context: .
          file: apps/${{ matrix.app }}/Dockerfile
          push: true
          tags: ${{ steps.ecr.outputs.registry }}/storysmith-${{ matrix.app }}:${{ github.ref_name }}
      - name: point task definition at new tag
        run: uv run python scripts/update_task_def.py --app ${{ matrix.app }} --tag ${{ github.ref_name }}
```

Azure variant: swap AWS steps for `azure/login` (OIDC federated credential) + `az acr build` + `az containerapp job update`. Same structure; dev agent implements whichever cloud the settings target.

### 13.3 `.github/workflows/llm-evals.yml` (manual only)

`workflow_dispatch` with input `confirm_spend: {type: string}`; job runs only if input equals `"yes"`. Steps: uv sync → `uv run deepeval test run tests/llm` with `SS_ANTHROPIC_API_KEY` from repo secrets → upload score table as artifact + PR comment when run from a PR context. Optional nightly `schedule:` trigger commented out by default.

### 13.4 Cost notes

- Public repo: Actions minutes are free/unlimited on GitHub-hosted runners. Private repo Free plan: 2,000 min/month included; the ci job above runs ~4-6 min, so even 20 pushes/day fits. Overage billed per-minute only if exceeded.
- Real money risks are API keys, not minutes: generation/LLM secrets are used ONLY in `llm-evals.yml` (gated) and never in `ci.yml`. Enforce by keeping `SS_REPLICATE_API_TOKEN` out of Actions secrets entirely — evals don't need it.
- Docker layer caching via `docker/build-push-action` cache-to/from `gha` keeps deploy under ~5 min.
