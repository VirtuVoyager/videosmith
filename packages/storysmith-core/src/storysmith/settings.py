from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed config. Constructed only at the app edge (apps/worker, apps/api) and
    passed into Pipeline — storysmith-core never reads process env vars directly."""

    model_config = SettingsConfigDict(env_prefix="SS_", env_file=".env", extra="ignore")

    # --- LLM ---
    llm_provider: str = "anthropic"  # anthropic | azure_openai
    anthropic_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment_standard: str = ""
    azure_openai_deployment_vision: str = ""
    azure_openai_api_version: str = "2024-10-21"

    # --- Generation providers ---
    replicate_api_token: str = ""
    video_model_i2v: str = "wan-video/wan-2.2-i2v-fast"
    video_model_t2v: str = "wan-video/wan-2.2-t2v-fast"
    image_model: str = "black-forest-labs/flux-schnell"
    music_model: str = "lucataco/ace-step"
    tts_voice: str = "af_bella"
    tts_voice_hi: str = ""

    # --- Storage ---
    storage_backend: str = "local"  # local | s3 | azure_blob
    output_dir: str = "./out"
    s3_bucket: str = ""
    aws_region: str = "eu-central-1"
    azure_blob_account_url: str = ""
    azure_blob_container: str = "storysmith"

    # --- Database ---
    db_url: str = ""  # empty = MemorySaver, no persistence

    # --- Budget & runtime ---
    budget_cap_usd: float = 12.0
    debug: bool = False  # SS_DEBUG=1 => debugpy wait-for-client in container

    # --- Review & publish ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    api_bearer_token: str = ""
    youtube_client_secrets_path: str = "./secrets/yt_client.json"

    # --- Observability ---
    opik_enabled: bool = False
    opik_api_key: str = ""
