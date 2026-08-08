from __future__ import annotations

import json
from collections.abc import Callable
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
