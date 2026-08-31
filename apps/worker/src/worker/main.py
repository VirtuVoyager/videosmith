from __future__ import annotations

import asyncio
import os

import structlog
import typer
from storysmith.models import Mode
from storysmith.pipeline import Pipeline
from storysmith.settings import Settings

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _configure_structlog() -> None:
    # §0.2: JSON output, one bound logger per agent (project_id bound at the
    # node-wrapper choke point in graph/build.py's _instrumented).
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
        logger_factory=structlog.PrintLoggerFactory(),
    )


def _configure_opik(settings: Settings) -> None:
    # opik.configure() writes a persistent ~/.opik.config dotfile; the SDK
    # reads these two env vars on its own before that, so setting them here
    # (app edge only -- storysmith-core never reads os.environ) is enough to
    # point @opik.track traces at the local self-hosted instance with no
    # global side effect and no API key needed.
    if not settings.opik_enabled:
        return
    os.environ["OPIK_URL_OVERRIDE"] = settings.opik_url
    os.environ["OPIK_PROJECT_NAME"] = "storysmith"
    if settings.opik_api_key:
        os.environ["OPIK_API_KEY"] = settings.opik_api_key


@app.callback()
def _callback() -> None:
    """StorySmith worker CLI."""
    _configure_structlog()


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
    show_id: str | None = typer.Option(
        None,
        "--show-id",
        help="Run an episode against a frozen show cast (created via POST /shows) "
        "instead of generating a fresh one; --brief becomes this episode's topic",
    ),
) -> None:
    """Run the StorySmith pipeline end to end."""
    settings = Settings()
    _configure_opik(settings)
    if stubs:
        pipeline = Pipeline.with_stubs(settings)
    else:
        from storysmith_adapters.factory import build_port_bundle

        pipeline = Pipeline(settings=settings, ports=build_port_bundle(settings))

    project = asyncio.run(pipeline.run(brief=brief, mode=mode, project_id=resume, show_id=show_id))
    typer.echo(
        f"project_id={project.project_id} status={project.status.value} "
        f"total_cost={project.total_cost:.4f}"
    )


if __name__ == "__main__":
    app()
