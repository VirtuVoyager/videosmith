from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from storysmith.models import VideoProject
from storysmith.pipeline import Pipeline, PortBundle
from storysmith.settings import Settings
from storysmith_adapters.stubs import stub_port_bundle

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "states"


@pytest.fixture
def settings_test(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        storage_backend="local",
        output_dir=str(tmp_path / "out"),
        db_url="",
    )


@pytest.fixture
def settings_test_pg(tmp_path: Path) -> Settings:
    # WP8: real Postgres for checkpoint/cost-ledger persistence tests.
    # SS_DB_URL is set by CI's postgres service (see .github/workflows/ci.yml)
    # and by `docker compose up -d postgres` locally; skip if neither is up.
    db_url = os.environ.get(
        "SS_DB_URL", "postgresql+psycopg://storysmith:storysmith@localhost:5432/storysmith"
    )
    return Settings(
        _env_file=None,
        storage_backend="local",
        output_dir=str(tmp_path / "out"),
        db_url=db_url,
    )


async def _postgres_reachable(db_url: str) -> bool:
    from storysmith import db

    try:
        await db.ensure_schema(db_url)
    except Exception:
        return False
    return True


@pytest.fixture
def pg_required(settings_test_pg: Settings) -> Settings:
    import asyncio

    if not asyncio.run(_postgres_reachable(settings_test_pg.db_url)):
        pytest.skip("no reachable Postgres at SS_DB_URL / localhost:5432 -- see docker-compose.yml")
    return settings_test_pg


@pytest.fixture
async def shows_cleanup(pg_required: Settings) -> AsyncIterator[list[str]]:
    """GET /shows / apps/ui's show picker lists every row in the real
    `shows` table -- a test that writes one directly (db.save_show or
    POST /shows) against pg_required must delete it afterward or it
    clutters that picker indefinitely (confirmed live: 17 of 19 rows in the
    real dev Postgres were test artifacts). Tests append whichever show_id(s)
    they created to the yielded list; cleanup runs even if the test fails.
    Never reuse a real show's name/id here (e.g. "bob-and-miko") even with
    this cleanup -- db.save_show is a full-overwrite upsert, so a test run
    would first clobber that show's real data before this fixture deletes
    the row entirely, which is worse than just leaving clutter."""
    from storysmith import db

    show_ids: list[str] = []
    try:
        yield show_ids
    finally:
        for show_id in show_ids:
            await db.delete_show(pg_required.db_url, show_id=show_id)


@pytest.fixture
def stub_ports() -> PortBundle:
    return stub_port_bundle()


@pytest.fixture
def pipeline_stubbed(settings_test: Settings, stub_ports: PortBundle) -> Pipeline:
    return Pipeline(settings=settings_test, ports=stub_ports)


@pytest.fixture
def state_fixture() -> Callable[[str], VideoProject]:
    def _load(name: str) -> VideoProject:
        data = json.loads((FIXTURES_DIR / f"{name}.json").read_text())
        return VideoProject.model_validate(data)

    return _load
