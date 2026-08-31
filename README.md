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

### Built to not bankrupt you

Every provider call writes a cost entry to Postgres. A per-run cap is checked
before every node; a cross-run daily cap is checked before a run starts. Video
generation dominates cost; a ~40s, 5-scene video at 480p prices out around $2.
Structured JSON logs and optional self-hosted Opik tracing (one span per node)
cover the 3am-failure case.

## Setup

[keep existing Quickstart details: ffmpeg/libass note, Postgres, review console,
Docker compose, YouTube OAuth, Telegram bot — unchanged, in this order]

## Configuration

[existing env var table, unchanged]

## Tests

[existing test section, unchanged]

## Development history

Built spec-first: [HANDOFF_SPEC.md](HANDOFF_SPEC.md) is the frozen original
design, with changes layered as numbered amendments
([Amendment 01](HANDOFF_SPEC_AMENDMENT_01_scene_composition.md) added
stills-first scene composition).

## License

MIT