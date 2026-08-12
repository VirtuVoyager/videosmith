from __future__ import annotations

from storysmith.pipeline import PortBundle
from storysmith.ports import LLMPort, StoragePort
from storysmith.settings import Settings


def _select_llm(settings: Settings) -> LLMPort:
    if settings.llm_provider == "anthropic":
        from storysmith_adapters.llm_anthropic import AnthropicLLM

        return AnthropicLLM(settings)
    if settings.llm_provider == "groq":
        from storysmith_adapters.llm_groq import GroqLLM

        return GroqLLM(settings)
    # SPEC-GAP: azure_openai is an accepted llm_provider value per §1.2b but
    # has no adapter yet -- no llm_azure_openai.py has been written.
    raise NotImplementedError(
        f"llm_provider={settings.llm_provider!r} has no adapter yet "
        "(anthropic and groq are implemented)"
    )


def _select_storage(settings: Settings) -> StoragePort:
    if settings.storage_backend == "local":
        from storysmith_adapters.storage_local import LocalStorage

        return LocalStorage(settings)
    if settings.storage_backend == "s3":
        from storysmith_adapters.storage_s3 import S3Storage

        return S3Storage(settings)
    raise NotImplementedError(f"storage_backend={settings.storage_backend!r} not implemented")


def build_port_bundle(settings: Settings) -> PortBundle:
    """Real (non-stub) adapters for every port, shared by apps/worker (live
    `storysmith run`) and apps/api (approve -> resume graph, POST /runs)
    so the two apps can't drift on adapter selection."""
    from storysmith_adapters.image_replicate import ReplicateImageGen
    from storysmith_adapters.music_replicate import ReplicateMusicGen
    from storysmith_adapters.notify_telegram import TelegramNotify
    from storysmith_adapters.publish_youtube import YouTubePublish
    from storysmith_adapters.transcribe_whisper import WhisperTranscribe
    from storysmith_adapters.tts_kokoro import KokoroTTS
    from storysmith_adapters.video_replicate import ReplicateVideoGen

    return PortBundle(
        llm=_select_llm(settings),
        image_gen=ReplicateImageGen(settings),
        video_gen=ReplicateVideoGen(settings),
        music_gen=ReplicateMusicGen(settings),
        tts=KokoroTTS(settings),
        transcribe=WhisperTranscribe(),
        storage=_select_storage(settings),
        publish=YouTubePublish(settings),
        notify=TelegramNotify(settings),
    )
