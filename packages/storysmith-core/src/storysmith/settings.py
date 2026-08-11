from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed config. Constructed only at the app edge (apps/worker, apps/api) and
    passed into Pipeline — storysmith-core never reads process env vars directly."""

    model_config = SettingsConfigDict(env_prefix="SS_", env_file=".env", extra="ignore")

    # --- LLM ---
    llm_provider: str = "anthropic"  # anthropic | groq | azure_openai
    anthropic_api_key: str = ""
    anthropic_model_standard: str = "claude-sonnet-4-6"
    anthropic_model_vision: str = "claude-sonnet-4-6"
    # Groq: OpenAI-compatible API, hosts free/cheap open-weight models --
    # useful for zero-cost development before switching llm_provider back to
    # anthropic once the system is stable.
    groq_api_key: str = ""
    groq_model_standard: str = "llama-3.3-70b-versatile"
    groq_model_vision: str = "llama-3.2-90b-vision-preview"
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
    video_model_i2v: str = "xai/grok-imagine-video"
    video_model_t2v: str = "xai/grok-imagine-video"
    video_resolution: str = "480p"  # 480p | 720p -- 720p is a 2.5x cost multiplier
    image_model: str = "black-forest-labs/flux-schnell"
    music_model: str = "lucataco/ace-step"  # rhyme mode: lyrics-driven full song
    music_model_instrumental: str = "meta/musicgen"  # topical mode: instrumental bed
    tts_model: str = "jaaari/kokoro-82m"  # Kokoro TTS on Replicate, for topical narration
    tts_voice: str = "af_bella"
    tts_voice_hi: str = ""

    # --- Storage ---
    storage_backend: str = "local"  # local | s3 | azure_blob
    output_dir: str = "./out"
    configs_dir: str = "./configs"  # repo-root configs/ (style presets, safety rules, rubrics)
    s3_bucket: str = ""
    aws_region: str = "eu-central-1"
    azure_blob_account_url: str = ""
    azure_blob_container: str = "storysmith"

    # --- Database ---
    db_url: str = ""  # empty = MemorySaver, no persistence

    # --- Budget & runtime ---
    budget_cap_usd: float = 12.0
    debug: bool = False  # SS_DEBUG=1 => debugpy wait-for-client in container
    skip_ffmpeg: bool = False  # SS_SKIP_FFMPEG=1 => skip ffmpeg-dependent tests (§5)

    # --- Review & publish ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    api_bearer_token: str = ""
    youtube_client_secrets_path: str = "./secrets/yt_client.json"

    # --- Observability ---
    opik_enabled: bool = False
    opik_api_key: str = ""
