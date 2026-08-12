# StorySmith

Autonomous kids shorts platform: a nightly batch pipeline (LangGraph supervisor + agents)
that writes, generates, scores, edits, and publishes short-form video, gated by human
approval before anything goes live.

## Status

| WP | Status | Notes |
|---|---|---|
| WP1 | done | Contracts, state, orchestrator skeleton, storage. Graph runs end to end with stub adapters; real LLM/image/video/music/tts/publish/notify adapters land in WP2-WP7 |
| WP2 | done | Creative Director + Director + character refs. Multi-provider LLM adapter (anthropic, groq) plus image_replicate.py; CLI still requires `--stubs` for a full run until WP3-7 land |
| WP3 | done | Videographer + Replicate video adapter (i2v/t2v selection, idempotency hashing, semaphore-bounded concurrency). CLI still requires `--stubs` for a full run until WP4-7 land |
| WP4 | done | Music Director: ACE-Step (rhyme) + MusicGen (topical instrumental) + Kokoro TTS, all sharing a new `_replicate_base.py` poller (image/video adapters refactored onto it too). CLI still requires `--stubs` for a full run until WP5-7 land |
| WP5 | done | Editor: real ffmpeg pipeline (normalize, crossfade/cut concat, audio mix+duck+loudnorm, faster-whisper transcribe, ASS caption burn-in, Pillow thumbnail overlay). Requires an ffmpeg build with libass (`ffmpeg-full` on Homebrew; Ubuntu's apt package already includes it). CLI still requires `--stubs` for a full run until WP6-7 land |
| WP6 | done | Critic / QA: rubric-driven vision LLM scoring, keyframe extraction, safety-forces-human-review, retry ceilings (3 scene / 1 audio), audio WER check. Critic routing now fans out independently to `videographer` vs `music_director` depending on which failed, instead of always regenerating scenes for an audio-only issue |
| WP7 | done | Review gate (Telegram: title/cost/QA summary/presigned video/approve-reject link), real `publish_youtube.py` (YouTube Data API v3, madeForKids) + `notify_telegram.py` adapters, FastAPI review console (`GET/POST /projects`, `/healthz`, bearer auth) backed by a `projects` Postgres snapshot table, and a minimal Next.js 15 + Tailwind UI (project list + player/QA-cards/approve-reject page), verified end-to-end in a real browser against the real API and Postgres |
| WP8 | in-progress | Observability + resumability + cost ledger done: structlog JSON logs + run summary at every node (choke point in `graph/build.py`'s `_instrumented`), `AsyncPostgresSaver` checkpointing (real cross-process resume, see Ops), `cost_entries` Postgres table + daily budget cap, local self-hosted Opik tracing. AWS/ECS deployment scaffolding (§8 Dockerfiles, Terraform-out-of-scope console setup) not started -- not needed until an actual deploy target exists |

## Quickstart

```bash
uv sync
cp .env.example .env   # fill keys
uv run storysmith run --brief "counting ducks" --mode rhyme --stubs
```

First real run (no `--stubs`) spends money against the configured LLM, image, video, and
music providers — see `## Cost` before running live.

The Editor (WP5) needs an ffmpeg build with `libass` for caption burn-in. Ubuntu's
`apt-get install ffmpeg` already includes it; on macOS Homebrew's default `ffmpeg`
formula does not -- install `ffmpeg-full` instead:

```bash
brew uninstall ffmpeg --ignore-dependencies && brew install ffmpeg-full
```

For real cross-process resumability (survive a killed/crashed run without regenerating
already-completed scenes) and a durable, cross-run cost ledger, point `SS_DB_URL` at a real
Postgres instead of leaving it empty:

```bash
docker compose up -d postgres
uv run alembic upgrade head   # creates cost_entries (checkpoint tables self-migrate on first run)
```

Without `SS_DB_URL` set, the pipeline still runs (using an in-process `MemorySaver`), but a
killed process loses all checkpoint state and the next run starts from scratch.

To run the review console (needs `SS_DB_URL` -- it reads the `projects` snapshot table and
resumes checkpoints):

```bash
uv run uvicorn api.main:app --reload --port 8000   # apps/api
cd apps/ui && npm install && npm run dev            # apps/ui, http://localhost:3000
```

The UI asks for the API bearer token once (stored in the browser's localStorage) and talks
directly to the API on `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000`).

## Architecture

_SPEC-GAP: pipeline mermaid diagram and per-agent paragraph land here once WP1 graph
wiring exists. Full detail lives in [HANDOFF_SPEC.md](HANDOFF_SPEC.md) — do not duplicate
the spec here._

## Configuration

| Var | Required | Default | Description |
|---|---|---|---|
| `SS_LLM_PROVIDER` | no | `anthropic` | `anthropic` \| `groq` \| `azure_openai` |
| `SS_ANTHROPIC_API_KEY` | yes (if anthropic) | — | Anthropic API key |
| `SS_ANTHROPIC_MODEL_STANDARD` | no | `claude-sonnet-4-6` | Standard-tier Anthropic model id |
| `SS_ANTHROPIC_MODEL_VISION` | no | `claude-sonnet-4-6` | Vision-tier Anthropic model id |
| `SS_GROQ_API_KEY` | yes (if groq) | — | Groq API key |
| `SS_GROQ_MODEL_STANDARD` | no | `llama-3.3-70b-versatile` | Standard-tier Groq model id |
| `SS_GROQ_MODEL_VISION` | no | `llama-3.2-90b-vision-preview` | Vision-tier Groq model id |
| `SS_AZURE_OPENAI_ENDPOINT` | yes (if azure) | — | Azure OpenAI resource endpoint |
| `SS_AZURE_OPENAI_API_KEY` | yes (if azure) | — | Azure OpenAI API key |
| `SS_AZURE_OPENAI_DEPLOYMENT_STANDARD` | yes (if azure) | — | Standard-tier deployment name |
| `SS_AZURE_OPENAI_DEPLOYMENT_VISION` | yes (if azure) | — | Vision-tier deployment name |
| `SS_AZURE_OPENAI_API_VERSION` | no | `2024-10-21` | Azure OpenAI API version |
| `SS_REPLICATE_API_TOKEN` | yes | — | Replicate API token |
| `SS_VIDEO_MODEL_I2V` | no | `xai/grok-imagine-video` | Image-to-video model id |
| `SS_VIDEO_MODEL_T2V` | no | `xai/grok-imagine-video` | Text-to-video model id |
| `SS_VIDEO_RESOLUTION` | no | `480p` | `480p` \| `720p` -- 720p is a 2.5x cost multiplier |
| `SS_IMAGE_MODEL` | no | `black-forest-labs/flux-schnell` | Character ref image model id |
| `SS_MUSIC_MODEL` | no | `lucataco/ace-step` | Rhyme-mode music model id (lyrics-driven full song) |
| `SS_MUSIC_MODEL_INSTRUMENTAL` | no | `meta/musicgen` | Topical-mode music model id (instrumental bed) |
| `SS_TTS_MODEL` | no | `jaaari/kokoro-82m` | Kokoro TTS model id on Replicate |
| `SS_TTS_VOICE` | no | `af_bella` | Kokoro TTS voice id |
| `SS_TTS_VOICE_HI` | no | — | TTS voice id used when `StyleContract.language` is `hi`/`hi-en` |
| `SS_STORAGE_BACKEND` | no | `local` | `local` \| `s3` \| `azure_blob` |
| `SS_OUTPUT_DIR` | no | `./out` | Local storage root |
| `SS_CONFIGS_DIR` | no | `./configs` | Repo-root configs/ (style presets, safety rules, rubrics) |
| `SS_S3_BUCKET` | yes (if s3) | — | S3 bucket name |
| `SS_AWS_REGION` | no | `eu-central-1` | AWS region |
| `SS_AZURE_BLOB_ACCOUNT_URL` | yes (if azure_blob) | — | Azure Blob account URL |
| `SS_AZURE_BLOB_CONTAINER` | no | `storysmith` | Azure Blob container name |
| `SS_DB_URL` | no | empty (MemorySaver, in-process only) | `postgresql+psycopg://user:pass@host:5432/db` -- enables `AsyncPostgresSaver` checkpointing (real cross-process resume) and `cost_entries` persistence |
| `SS_BUDGET_CAP_USD` | no | `12.0` | Per-run budget cap (checked every node, mid-run) |
| `SS_DAILY_BUDGET_CAP_USD` | no | `0` (disabled) | Cross-run cap checked against `cost_entries` before a new run starts; requires `SS_DB_URL` |
| `SS_DEBUG` | no | `0` | `1` waits for debugpy client in container |
| `SS_SKIP_FFMPEG` | no | `0` | `1` skips ffmpeg-dependent tests (§5) |
| `SS_TELEGRAM_BOT_TOKEN` | yes (for review gate) | — | Telegram bot token |
| `SS_TELEGRAM_CHAT_ID` | yes (for review gate) | — | Telegram chat id |
| `SS_API_BEARER_TOKEN` | yes | — | Static bearer token for the FastAPI console |
| `SS_CONSOLE_BASE_URL` | no | `http://localhost:3000` | UI origin -- review_gate's Telegram link target and the API's CORS allow-origin |
| `SS_YOUTUBE_CLIENT_SECRETS_PATH` | yes (for publish) | `./secrets/yt_client.json` | OAuth client secrets JSON (Desktop app type) from Google Cloud Console |
| `SS_YOUTUBE_TOKEN_PATH` | no | `./secrets/yt_token.json` | Refresh token written by `scripts/youtube_auth.py`, read/refreshed by `publish_youtube.py` |
| `SS_OPIK_ENABLED` | no | `0` | Enable Opik tracing (one span per graph node) |
| `SS_OPIK_API_KEY` | no | — | Only needed for cloud Opik; self-hosted local instances need no key |
| `SS_OPIK_URL` | no | `http://localhost:5173/api` | Self-hosted local Opik instance base URL (note the `/api` suffix) |

## Running Tests

```bash
# Unit — no network, stub adapters only, runs on every commit
uv run pytest tests/unit -m "not slow" --cov --cov-fail-under=80
# WP8 tests that need real Postgres (checkpoint resume, cost ledger, daily cap)
# auto-skip if nothing is reachable at SS_DB_URL / localhost:5432; to run them:
docker compose up -d postgres && uv run alembic upgrade head
uv run pytest tests/unit -m wp8
uv run pytest tests/unit -m wp7   # API route tests -- also needs Postgres, same as above

# UI: type-check, build, lint
cd apps/ui && npm install && npm run build && npm run lint

# LLM eval — real LLM calls, costs money, run manually/nightly
uv run deepeval test run tests/llm

# Live smoke — real providers, one truncated topical project, run before release
uv run python scripts/smoke_live.py
```

## Debugging

Configs live in [.vscode/launch.json](.vscode/launch.json):

- **Pipeline: run with stubs** — zero-cost end-to-end run.
- **Pipeline: run LIVE (spends money)** — real providers.
- **Pipeline: resume project** — resume from checkpoint by `project_id`.
- **API: FastAPI dev** — review console API with reload.
- **Pytest: current file** — debug the open test file.
- **Attach: worker in container (5678)** — attach to a running container with `SS_DEBUG=1`.

## Ops

**Diagnosability.** Every graph node emits structured (JSON, via `structlog`) `node_start` /
`node_done` / `node_failed` events with `project_id` bound, at the single choke point every
node passes through (`graph/build.py`'s `_instrumented` wrapper). `Pipeline.run()` also logs
`run_starting`/`run_resuming`, `run_failed` (with wall time) on exception, and a `run_summary`
at the end of every run: total cost, cost broken down by provider, retries per scene, wall
time. To find out what happened on a failed `project_id`, grep the worker's JSON logs for
that `project_id` -- `node_failed` marks exactly which node and includes the exception.

**Resumability.** Set `SS_DB_URL` to a real Postgres and the graph checkpoints there via
`AsyncPostgresSaver` instead of an in-process `MemorySaver`. If a run is killed or crashes
mid-way, `storysmith run --resume <project_id>` (or `Pipeline.run(..., project_id=...)` from
a brand-new process) picks up from the last completed node's checkpoint -- already-generated
scenes/audio/etc. are not regenerated, so a crash doesn't waste the money already spent on
them. Without `SS_DB_URL`, resumption only works within the same process (e.g. after a
single node's internal retry), not across a process restart.

**Cost ledger & daily budget cap.** With `SS_DB_URL` set, every `CostEntry` a node adds is
also written to the `cost_entries` Postgres table (in addition to the in-memory/checkpointed
`VideoProject.cost_ledger`), so spend is queryable across runs and processes -- not just
within one project's state. `SS_BUDGET_CAP_USD` remains the per-run cap (checked before every
node); `SS_DAILY_BUDGET_CAP_USD` is a separate cross-run cap checked once at the start of
`Pipeline.run()` against today's `cost_entries` total, and raises `BudgetExceededError` before
anything runs if today's spend is already at or over the cap. This is the app-level daily
aggregation layer that plugs the gap left by Replicate's July 2025 removal of monthly spend
limits (Replicate itself only offers a hard prepaid-credit stop, no daily granularity).

**Migrations.** `cost_entries`' schema is defined by the Alembic migration at
`alembic/versions/0001_create_cost_entries.py`; run `uv run alembic upgrade head` against
`SS_DB_URL` to apply it to a shared/production database. `Pipeline.run()` also calls
`CREATE TABLE IF NOT EXISTS` via the ORM model on every run as a dev/test convenience (see
`db.ensure_schema`) -- safe to leave in place alongside Alembic since it's a no-op once the
migration has run. LangGraph's own checkpoint tables (`checkpoints`, `checkpoint_writes`,
`checkpoint_blobs`, `checkpoint_migrations`) are created separately by
`AsyncPostgresSaver.setup()` the first time a Postgres-backed run happens -- they're not part
of the Alembic migration.

**Observability (Opik).** `SS_OPIK_ENABLED=1` adds one Opik span per graph node (same choke
point as the structlog logging), pointed at `SS_OPIK_URL` (a self-hosted local instance by
default, e.g. `http://localhost:5173/api` on OrbStack). No `opik.configure()` call is made
(that writes a persistent `~/.opik.config` dotfile); instead `OPIK_URL_OVERRIDE` /
`OPIK_PROJECT_NAME` / `OPIK_API_KEY` are set as process env vars at the app edge
(`apps/worker/main.py`, only when `settings.opik_enabled`), which the Opik SDK reads on its
own. Self-hosted instances need no API key.

**YouTube OAuth one-time setup.** In the Google Cloud Console: create a project, enable the
YouTube Data API v3, and create an OAuth client of type "Desktop app" -- download its JSON to
`SS_YOUTUBE_CLIENT_SECRETS_PATH` (default `./secrets/yt_client.json`). Then run:

```bash
uv run python scripts/youtube_auth.py
```

This opens a browser for one-time consent and writes a refresh token to
`SS_YOUTUBE_TOKEN_PATH` (default `./secrets/yt_token.json`). `publish_youtube.py` reads and
refreshes that file on every upload -- the script never needs to run again unless the token
file is deleted or access is revoked. Uploads are created with `privacyStatus=private` and
`selfDeclaredMadeForKids=true`; review and change visibility manually if needed.

**Telegram bot setup.** Create a bot via [@BotFather](https://t.me/BotFather), set
`SS_TELEGRAM_BOT_TOKEN` to the token it gives you, message the bot once, then get your chat id
from `https://api.telegram.org/bot<token>/getUpdates` and set `SS_TELEGRAM_CHAT_ID`.
`review_gate` sends the title, cost, QA summary, presigned final-video URL (once one exists),
and a link into the UI console (`SS_CONSOLE_BASE_URL/p/<project_id>`) -- the UI holds the
bearer token client-side and calls the API's approve/reject endpoints, since a bare Telegram
link can't carry an `Authorization` header.

**Review console.** `apps/api` (FastAPI) exposes `GET /healthz`, `GET /projects`,
`GET /projects/{id}`, `POST /projects/{id}/approve`, `POST /projects/{id}/reject`, and
`POST /runs`; every route except `/healthz` requires `Authorization: Bearer <SS_API_BEARER_TOKEN>`.
It reads/writes state via the `projects` Postgres snapshot table (kept in sync from the same
node-wrapper choke point as the cost ledger) and resumes the LangGraph checkpoint directly on
approve/reject -- both need `SS_DB_URL` set. `apps/ui` (Next.js) is a thin, fetch-only client:
a project list and a per-project page with the video player, QA-report cards, and
Approve/Reject buttons. Regenerate its typed API client after changing any route in
`apps/api/src/api/main.py`:

```bash
uv run python scripts/gen_openapi.py
cd apps/ui && npm run gen-api-types
```

_SPEC-GAP: cron deployment summary and AWS container deployment land once an actual deploy
target exists -- not needed for local/dev use._

## Cost

_No measured data yet — populated after the first `scripts/smoke_live.py` run._

## License

TBD (core deps MIT/Apache only).
