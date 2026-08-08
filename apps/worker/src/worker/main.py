from __future__ import annotations

import asyncio

import typer
from storysmith.models import Mode
from storysmith.pipeline import Pipeline
from storysmith.settings import Settings

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _callback() -> None:
    """StorySmith worker CLI."""


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
        # SPEC-GAP: real LLM/image/video/music/tts/publish/notify adapters
        # land across WP2-WP7 -- only storage_local/storage_s3 exist so far,
        # so a non-stub run has nothing else to wire up yet.
        raise NotImplementedError(
            "Real adapters are not implemented until WP2-WP7 land. Pass --stubs for now."
        )

    project = asyncio.run(pipeline.run(brief=brief, mode=mode, project_id=resume))
    typer.echo(
        f"project_id={project.project_id} status={project.status.value} "
        f"total_cost={project.total_cost:.4f}"
    )


if __name__ == "__main__":
    app()
