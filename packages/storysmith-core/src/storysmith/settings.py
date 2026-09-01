from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed config. Constructed only at the app edge (apps/worker, apps/api) and
    passed into Pipeline — storysmith-core never reads process env vars directly."""

    model_config = SettingsConfigDict(env_prefix="SS_", env_file=".env", extra="ignore")

    # --- LLM ---
    llm_provider: str = "anthropic"  # anthropic | groq | replicate | azure_openai
    anthropic_api_key: str = ""
    anthropic_model_standard: str = "claude-sonnet-4-6"
    anthropic_model_vision: str = "claude-sonnet-4-6"
    # Groq: OpenAI-compatible API, hosts free/cheap open-weight models --
    # useful for zero-cost development before switching llm_provider back to
    # anthropic once the system is stable.
    groq_api_key: str = ""
    # llama-3.3-70b-versatile / llama-3.2-90b-vision-preview (this project's
    # original defaults) were both fully removed from Groq's lineup at some
    # point -- confirmed live via GET /openai/v1/models, no llama chat model
    # remains at all, only meta-llama/llama-prompt-guard-2-* (safety
    # classifiers, not general chat). openai/gpt-oss-120b is Groq's current
    # closest equivalent for the standard tier.
    # SPEC-GAP: no vision-capable model was found in Groq's current lineup
    # (GET /openai/v1/models) at all -- if Critic's vision-tier calls need to
    # stay reliable, set SS_LLM_PROVIDER=anthropic instead of chasing a Groq
    # vision replacement here.
    groq_model_standard: str = "openai/gpt-oss-120b"
    groq_model_vision: str = "openai/gpt-oss-120b"
    # Replicate: raw text-completion models (no tool-calling/JSON mode), used
    # as a rate-limit escape hatch from Groq's free-tier 8000 TPM cap -- see
    # llm_replicate.py's docstring. Same SS_REPLICATE_API_TOKEN as the
    # image/video/music/TTS adapters, metered pay-as-you-go (no TPM cap).
    # SPEC-GAP: no vision-capable Replicate model wired in -- vision tier
    # falls back to the same text-only model; Critic's keyframe QA runs
    # blind under this provider (see llm_replicate.py).
    replicate_model_standard: str = "openai/gpt-oss-120b"
    replicate_model_vision: str = "openai/gpt-oss-120b"
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment_standard: str = ""
    azure_openai_deployment_vision: str = ""
    azure_openai_api_version: str = "2024-10-21"

    # --- Generation providers ---
    replicate_api_token: str = ""
    # xai/grok-imagine-video handles i2v and t2v via the same model id (image
    # param present or absent); Wan needs two distinct repos, hence two
    # settings fields -- point both at the same value for single-model
    # providers, or back at wan-video/wan-2.2-{i2v,t2v}-fast to switch back.
    # prunaai/p-video: cheaper AND higher resolution than the prior default
    # (xai/grok-imagine-video @ 480p, $0.05/sec flat) at every quality level --
    # $0.02/sec @720p or $0.04/sec @1080p, confirmed live. Also supports a
    # `draft: bool` input (~5-10x cheaper, lower quality) not wired in here
    # yet -- a future lever for cheap Critic-retry iterations before a final
    # full-quality render, not used by default.
    video_model_i2v: str = "prunaai/p-video"
    video_model_t2v: str = "prunaai/p-video"
    video_resolution: str = "720p"  # 720p | 1080p -- 1080p is a 2x cost multiplier
    image_model: str = "black-forest-labs/flux-schnell"
    # lucataco/ace-step was removed from Replicate; fishaudio/ace-step-1.5 is
    # its current replacement (same underlying ACE-Step model).
    music_model: str = "fishaudio/ace-step-1.5"  # rhyme mode: lyrics-driven full song
    music_model_instrumental: str = "meta/musicgen"  # topical mode: instrumental bed
    tts_model: str = "jaaari/kokoro-82m"  # Kokoro TTS on Replicate, for topical narration
    tts_voice: str = "af_bella"
    tts_voice_hi: str = ""
    # Amendment 01: Director's default gen_mode when a scene isn't clearly
    # geometry-light (threaded into the director prompt, not enforced in
    # code -- the LLM still picks per-scene, this is just its starting bias).
    default_scene_gen_mode: str = "i2v"  # i2v | t2v
    # SPEC-GAP: flux-kontext-pro supports image-conditioned generation (the
    # capability §6 of the amendment gates i2v-with-character-identity on),
    # but wiring a second image adapter/port to actually use that
    # conditioning is deferred -- scene_stills currently reuses `image_model`
    # (flux-schnell, text-only) via the existing ImageGenPort, folding
    # character appearance into the prompt text instead (the amendment's own
    # documented fallback for text-only image adapters). This setting is
    # accepted now so the config surface matches the amendment; unused until
    # that adapter work lands.
    scene_image_model: str = "black-forest-labs/flux-kontext-pro"

    # --- Storage ---
    storage_backend: str = "local"  # local | s3 | azure_blob
    output_dir: str = "./out"
    configs_dir: str = "./configs"  # repo-root configs/ (style presets, safety rules, rubrics)
    s3_bucket: str = ""
    aws_region: str = "eu-central-1"
    azure_blob_account_url: str = ""
    azure_blob_container: str = "storysmith"

    # --- Database ---
    # empty = MemorySaver checkpointing, no cost-ledger persistence (in-process
    # only, no cross-run resume). postgresql+psycopg://user:pass@host:5432/db
    # enables both AsyncPostgresSaver checkpointing and the cost_entries table.
    db_url: str = ""

    # --- Budget & runtime ---
    budget_cap_usd: float = 12.0
    # Cross-run cap, checked against cost_entries before a new run starts
    # (§ Replicate spend controls) -- 0 disables the check. Requires db_url;
    # budget_cap_usd alone only guards a single run's own spend.
    daily_budget_cap_usd: float = 0.0
    debug: bool = False  # SS_DEBUG=1 => debugpy wait-for-client in container
    skip_ffmpeg: bool = False  # SS_SKIP_FFMPEG=1 => skip ffmpeg-dependent tests (§5)

    # --- Review & publish ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    api_bearer_token: str = ""
    # review_gate's Telegram message links here (§7's "approve/reject deep
    # links") -- the UI page holds the bearer token client-side and calls the
    # API's approve/reject endpoints, since a bare Telegram link can't carry
    # an Authorization header.
    console_base_url: str = "http://localhost:3000"
    youtube_client_secrets_path: str = "./secrets/yt_client.json"
    # Written by scripts/youtube_auth.py's one-time OAuth flow; read (and
    # refreshed in place) by publish_youtube.py on every upload.
    youtube_token_path: str = "./secrets/yt_token.json"

    # --- Observability ---
    opik_enabled: bool = False
    opik_api_key: str = ""
    opik_url: str = "http://localhost:5173/api"  # self-hosted local Opik instance
