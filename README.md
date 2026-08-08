# StorySmith

Autonomous kids shorts platform: a nightly batch pipeline (LangGraph supervisor + agents)
that writes, generates, scores, edits, and publishes short-form video, gated by human
approval before anything goes live.

## Status

| WP | Status | Notes |
|---|---|---|
| WP1 | done | Contracts, state, orchestrator skeleton, storage. Graph runs end to end with stub adapters; real LLM/image/video/music/tts/publish/notify adapters land in WP2-WP7 |
| WP2 | done | Creative Director + Director + character refs. Multi-provider LLM adapter (anthropic, groq) plus image_replicate.py; CLI still requires `--stubs` for a full run until WP3-7 land |
| WP3 | todo | Videographer + Replicate video adapter |
| WP4 | todo | Music Director |
| WP5 | todo | Editor (ffmpeg, deterministic) |
| WP6 | todo | Critic / QA |
| WP7 | todo | Review gate, Publisher, API, UI |
| WP8 | todo | Observability + cost ledger + ops |

## Quickstart

```bash
uv sync
cp .env.example .env   # fill keys
uv run storysmith run --brief "counting ducks" --mode rhyme --stubs
```

First real run (no `--stubs`) spends money against the configured LLM, image, video, and
music providers — see `## Cost` before running live.

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
| `SS_VIDEO_MODEL_I2V` | no | `wan-video/wan-2.2-i2v-fast` | Image-to-video model id |
| `SS_VIDEO_MODEL_T2V` | no | `wan-video/wan-2.2-t2v-fast` | Text-to-video model id |
| `SS_IMAGE_MODEL` | no | `black-forest-labs/flux-schnell` | Character ref image model id |
| `SS_MUSIC_MODEL` | no | `lucataco/ace-step` | Music generation model id |
| `SS_TTS_VOICE` | no | `af_bella` | Kokoro TTS voice id |
| `SS_TTS_VOICE_HI` | no | — | TTS voice id used when `StyleContract.language` is `hi`/`hi-en` |
| `SS_STORAGE_BACKEND` | no | `local` | `local` \| `s3` \| `azure_blob` |
| `SS_OUTPUT_DIR` | no | `./out` | Local storage root |
| `SS_CONFIGS_DIR` | no | `./configs` | Repo-root configs/ (style presets, safety rules, rubrics) |
| `SS_S3_BUCKET` | yes (if s3) | — | S3 bucket name |
| `SS_AWS_REGION` | no | `eu-central-1` | AWS region |
| `SS_AZURE_BLOB_ACCOUNT_URL` | yes (if azure_blob) | — | Azure Blob account URL |
| `SS_AZURE_BLOB_CONTAINER` | no | `storysmith` | Azure Blob container name |
| `SS_DB_URL` | no | empty (MemorySaver) | Postgres URL for checkpointing |
| `SS_BUDGET_CAP_USD` | no | `12.0` | Per-project budget cap |
| `SS_DEBUG` | no | `0` | `1` waits for debugpy client in container |
| `SS_TELEGRAM_BOT_TOKEN` | yes (for review gate) | — | Telegram bot token |
| `SS_TELEGRAM_CHAT_ID` | yes (for review gate) | — | Telegram chat id |
| `SS_API_BEARER_TOKEN` | yes | — | Static bearer token for the FastAPI console |
| `SS_YOUTUBE_CLIENT_SECRETS_PATH` | yes (for publish) | `./secrets/yt_client.json` | YouTube OAuth client secrets path |
| `SS_OPIK_ENABLED` | no | `0` | Enable OPIK tracing |
| `SS_OPIK_API_KEY` | yes (if opik enabled) | — | OPIK API key |

## Running Tests

```bash
# Unit — no network, stub adapters only, runs on every commit
uv run pytest tests/unit -m "not slow" --cov --cov-fail-under=80

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

_SPEC-GAP: cron deployment summary, secrets locations, YouTube OAuth one-time setup,
budget cap behavior, and `BUDGET_ABORT`/`HUMAN_REVIEW` runbook land here at WP8._

## Cost

_No measured data yet — populated after the first `scripts/smoke_live.py` run._

## License

TBD (core deps MIT/Apache only).
