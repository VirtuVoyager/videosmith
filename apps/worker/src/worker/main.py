from __future__ import annotations

import asyncio
from typing import Any

import typer
from storysmith.models import Mode
from storysmith.pipeline import Pipeline
from storysmith.settings import Settings

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _callback() -> None:
    """StorySmith worker CLI."""


def _select_llm_adapter(settings: Settings) -> Any:
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


def _select_storage(settings: Settings) -> Any:
    if settings.storage_backend == "local":
        from storysmith_adapters.storage_local import LocalStorage

        return LocalStorage(settings)
    if settings.storage_backend == "s3":
        from storysmith_adapters.storage_s3 import S3Storage

        return S3Storage(settings)
    raise NotImplementedError(f"storage_backend={settings.storage_backend!r} not implemented")


def _select_video_gen(settings: Settings) -> Any:
    from storysmith_adapters.video_replicate import ReplicateVideoGen

    return ReplicateVideoGen(settings)


def _select_music_gen(settings: Settings) -> Any:
    from storysmith_adapters.music_replicate import ReplicateMusicGen

    return ReplicateMusicGen(settings)


def _select_tts(settings: Settings) -> Any:
    from storysmith_adapters.tts_kokoro import KokoroTTS

    return KokoroTTS(settings)


def _select_transcribe(settings: Settings) -> Any:
    # No settings needed -- faster-whisper runs locally on CPU, no API key.
    del settings
    from storysmith_adapters.transcribe_whisper import WhisperTranscribe

    return WhisperTranscribe()


@app.command()
def run(
    brief: str = typer.Option(
        ..., "--brief", help="Video brief, e.g. 'counting to five with ducks'"
    ),
    mode: Mode = typer.Option(Mode.RHYME, "--mode", help="rhyme | topical"),
    stubs: bool = typer.Option(
        False, "--stubs", help="Use deterministic stub adapters -- zero paid calls"
    ),
    resume: str | None = typer.Option(
        None, "--resume", help="Resume an existing project_id from its checkpoint"
    ),
) -> None:
    """Run the StorySmith pipeline end to end."""
    settings = Settings()
    if stubs:
        pipeline = Pipeline.with_stubs(settings)
    else:
        # Fail fast on whichever piece is misconfigured/missing before
        # touching anything else -- llm/image_gen/storage (WP2), video_gen
        # (WP3), music_gen/tts (WP4), and transcribe (WP5) are real;
        # publish/notify land in WP6-7, so a full non-stub run still isn't
        # possible yet.
        _select_llm_adapter(settings)
        _select_storage(settings)
        _select_video_gen(settings)
        _select_music_gen(settings)
        _select_tts(settings)
        _select_transcribe(settings)
        raise NotImplementedError(
            "publish/notify adapters land in WP6-7. LLM (anthropic/groq), image_gen, "
            "video_gen, music_gen, tts, and transcribe (replicate/kokoro/whisper) are "
            "ready, but a full non-stub run isn't possible yet -- pass --stubs for now."
        )

    project = asyncio.run(pipeline.run(brief=brief, mode=mode, project_id=resume))
    typer.echo(
        f"project_id={project.project_id} status={project.status.value} "
        f"total_cost={project.total_cost:.4f}"
    )


if __name__ == "__main__":
    app()
