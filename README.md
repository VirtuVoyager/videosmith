# StorySmith

An autonomous short-form video studio. Give it a one-line brief at night, review a
finished, captioned, quality-scored video in the morning. A human approves every
video before it goes anywhere near YouTube.

One command, one video:

    uv sync
    cp .env.example .env   # add provider keys
    uv run storysmith run --brief "counting ducks" --mode rhyme --stubs

`--stubs` runs the entire pipeline end to end with fake adapters, zero cost. Drop
the flag to run against real providers (this spends money, see Cost below).

## How it works

The pipeline is a LangGraph graph of eight single-purpose agents, plus a human
review gate and a publisher:

| Agent | Job |
|---|---|
| Creative Director | Locks art style, palette, and characters up front |
| Director | Writes the script as a shot list: per-scene image prompt + motion prompt |
| Character Refs | Generates a reference portrait per character |
| Scene Stills | Composes one fixed still image per scene before any video is generated |
| Videographer | Animates each still into a clip (image-to-video) |
| Music Director | Full song via ACE-Step (rhyme mode) or instrumental + Kokoro TTS (topical) |
| Critic | Vision-LLM scoring against a rubric: style, consistency, artifacts, safety |
| Editor | ffmpeg: normalize, concat, mix/duck audio, whisper QA, caption burn-in, thumbnail |

Then a Telegram message with the video, cost, and QA summary. Approve or reject
from your phone or the web console. Only on approval does the Publisher upload
to YouTube (private, made-for-kids flagged).

### The two design decisions that matter

**Stills first.** Text-to-video models are bad at holding a scene together:
characters teleport, backgrounds mutate. So the video model is never asked to
invent a scene. Scene Stills generates an exact photograph of each moment first,
and the video model only animates motion within that fixed frame.

**Targeted retries.** When the Critic rejects output, it classifies the failure
layer. Wrong composition regenerates the still. Wrong motion regenerates only the
clip. Bad audio goes back to Music Director without touching video. Retry ceilings
(3 per scene, 1 for audio) bound the loop. Safety flags always force human review.

### Built to be swapped

Every provider sits behind an adapter. LLM (Anthropic / Groq / Azure OpenAI),
image, video, music, TTS, and storage (local / S3 / Azure Blob) are each one
environment variable. Nothing requires a specific vendor.

### Built to survive a crash

With `SS_DB_URL` set to Postgres, the graph checkpoints every node via
AsyncPostgresSaver. Kill the process mid-run and `storysmith run --resume <id>`
continues from the last completed node. Already-generated scenes are not
regenerated, so a crash does not re-spend money.

### Built for recurring casts

A show is a cast frozen once and reused forever: describe each character's
appearance, personality, and voice through the UI (or `POST /shows`), their
avatars generate and lock in immediately, and every future episode loads that
exact cast instead of inventing a new one — Creative Director and Character
Refs both skip (zero cost) once a project's style already has every
character's portrait. Scenes can carry speaker-attributed `dialogue` instead
of single-voice narration; Music Director synthesizes each line in its
speaker's own voice and stitches them into one clip before anything
downstream (Editor, timing map) ever sees it — nothing else in the pipeline
needs to know a scene had more than one voice.

```bash
# once, per show:
curl -X POST localhost:8000/shows -H "Authorization: Bearer $SS_API_BEARER_TOKEN" -d '{...}'
# every episode after that:
uv run storysmith run --show-id bob-and-miko --brief "Bob won't share the couch" --mode topical
```

### Built to not bankrupt you

Every provider call writes a cost entry to Postgres. A per-run cap is checked
before every node; a cross-run daily cap is checked before a run starts. Video
generation dominates cost; a ~40s, 5-scene video at 480p prices out around $2.
Structured JSON logs and optional self-hosted Opik tracing (one span per node)
cover the 3am-failure case.

## Setup

### Quickstart

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
resumes checkpoints), either locally:

```bash
uv run uvicorn api.main:app --reload --port 8000   # apps/api
cd apps/ui && npm install && npm run dev            # apps/ui, http://localhost:3000
```

...or fully containerized (OrbStack/Docker Desktop -- both apps/api and apps/ui ship
Dockerfiles, wired into `docker-compose.yml` alongside `postgres`):

```bash
docker compose up -d --build
```

This brings up Postgres, the API on `http://localhost:8000`, and the UI on
`http://localhost:3000` together. The API container loads `.env` for provider keys but
overrides `SS_DB_URL` to reach the `postgres` service by its container-network hostname
(not `localhost`), and `./out` (local storage) and `./secrets` (YouTube OAuth files) are
bind-mounted in, not baked into the image -- so a project's assets are readable both from
a container run and a host-side `uv run storysmith run --resume <id>` against the same
directory. `NEXT_PUBLIC_API_BASE_URL` is baked into the UI's client bundle at *build* time
(`docker compose build ui`), since it's fetched from the browser, not container-to-container
-- rebuild the `ui` image if that URL needs to change.

The UI asks for the API bearer token once (stored in the browser's localStorage) and talks
directly to the API on `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000`). The
home page's "Start a new video" form calls `POST /runs` directly -- the same real,
money-spending pipeline the CLI runs, just triggered from a browser instead of a terminal.

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


## Configuration

| Var | Required | Default | Description |
|---|---|---|---|
| `SS_LLM_PROVIDER` | no | `anthropic` | `anthropic` \| `groq` \| `replicate` \| `azure_openai` |
| `SS_ANTHROPIC_API_KEY` | yes (if anthropic) | — | Anthropic API key |
| `SS_ANTHROPIC_MODEL_STANDARD` | no | `claude-sonnet-4-6` | Standard-tier Anthropic model id |
| `SS_ANTHROPIC_MODEL_VISION` | no | `claude-sonnet-4-6` | Vision-tier Anthropic model id |
| `SS_GROQ_API_KEY` | yes (if groq) | — | Groq API key |
| `SS_GROQ_MODEL_STANDARD` | no | `openai/gpt-oss-120b` | Standard-tier Groq model id -- Groq's Llama lineup (this project's original default) was fully removed at some point; verify current availability via `GET /openai/v1/models` before relying on this. This account's free tier also caps every current chat model at 8000 tokens/minute -- smaller than a single Director request, confirmed live -- see `SS_LLM_PROVIDER=replicate` below |
| `SS_GROQ_MODEL_VISION` | no | `openai/gpt-oss-120b` | Vision-tier Groq model id -- no vision-capable model was found in Groq's lineup as of writing; use `SS_LLM_PROVIDER=anthropic` if Critic's vision calls need to stay reliable |
| `SS_REPLICATE_MODEL_STANDARD` | no | `openai/gpt-oss-120b` | Standard-tier model id on Replicate (only if `SS_LLM_PROVIDER=replicate`) -- a raw text-completion model, no tool-calling/JSON mode; see `llm_replicate.py`. Metered pay-as-you-go on `SS_REPLICATE_API_TOKEN` below, with no TPM cap -- the practical fix for Groq's 8000 TPM ceiling above |
| `SS_REPLICATE_MODEL_VISION` | no | `openai/gpt-oss-120b` | Vision-tier model id on Replicate -- SPEC-GAP: no vision-capable model is wired in, falls back to the same text-only model, so Critic's keyframe QA runs blind under this provider; use `SS_LLM_PROVIDER=anthropic` if that needs to stay reliable |
| `SS_AZURE_OPENAI_ENDPOINT` | yes (if azure) | — | Azure OpenAI resource endpoint |
| `SS_AZURE_OPENAI_API_KEY` | yes (if azure) | — | Azure OpenAI API key |
| `SS_AZURE_OPENAI_DEPLOYMENT_STANDARD` | yes (if azure) | — | Standard-tier deployment name |
| `SS_AZURE_OPENAI_DEPLOYMENT_VISION` | yes (if azure) | — | Vision-tier deployment name |
| `SS_AZURE_OPENAI_API_VERSION` | no | `2024-10-21` | Azure OpenAI API version |
| `SS_REPLICATE_API_TOKEN` | yes | — | Replicate API token |
| `SS_VIDEO_MODEL_I2V` | no | `prunaai/p-video` | Image-to-video model id -- cheaper and higher-res than the prior `xai/grok-imagine-video` @ 480p default at every quality tier, live-tested; also supports a `draft` mode (~5-10x cheaper, not wired in) for future cheap-retry iterations |
| `SS_VIDEO_MODEL_T2V` | no | `prunaai/p-video` | Text-to-video model id |
| `SS_VIDEO_RESOLUTION` | no | `720p` | `720p` \| `1080p` -- 1080p is a 2x cost multiplier |
| `SS_IMAGE_MODEL` | no | `black-forest-labs/flux-schnell` | Character ref image model id |
| `SS_MUSIC_MODEL` | no | `fishaudio/ace-step-1.5` | Rhyme-mode music model id (lyrics-driven full song) |
| `SS_MUSIC_MODEL_INSTRUMENTAL` | no | `meta/musicgen` | Topical-mode music model id (instrumental bed) |
| `SS_TTS_MODEL` | no | `jaaari/kokoro-82m` | Kokoro TTS model id on Replicate |
| `SS_TTS_VOICE` | no | `af_bella` | Kokoro TTS voice id |
| `SS_TTS_VOICE_HI` | no | — | TTS voice id used when `StyleContract.language` is `hi`/`hi-en` |
| `SS_DEFAULT_SCENE_GEN_MODE` | no | `i2v` | Director's default per-scene `gen_mode` (`i2v` \| `t2v`) when unsure, see Amendment 01 |
| `SS_SCENE_IMAGE_MODEL` | no | `black-forest-labs/flux-kontext-pro` | Image-conditioning-capable model id, reserved for a future adapter -- `scene_stills` currently reuses `SS_IMAGE_MODEL` (text-only), see settings.py |
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

## Tests

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

## Development history

Built spec-first: [HANDOFF_SPEC.md](HANDOFF_SPEC.md) is the frozen original
design, with changes layered as numbered amendments
([Amendment 01](HANDOFF_SPEC_AMENDMENT_01_scene_composition.md) added
stills-first scene composition, [Amendment 02](HANDOFF_SPEC_AMENDMENT_02_show_character_library.md)
added user-authored show casts and multi-character dialogue).

## License

MIT