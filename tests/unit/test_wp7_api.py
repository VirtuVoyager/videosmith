from __future__ import annotations

from collections.abc import Iterator

import pytest
from api.main import app, get_ports, get_settings
from fastapi.testclient import TestClient
from storysmith.models import Mode, ProjectStatus
from storysmith.pipeline import Pipeline, PortBundle
from storysmith.settings import Settings
from storysmith_adapters.stubs import stub_port_bundle

pytestmark = pytest.mark.wp7


@pytest.fixture
def api_settings(pg_required: Settings) -> Settings:
    return pg_required.model_copy(update={"api_bearer_token": "test-token"})


@pytest.fixture
def api_ports() -> PortBundle:
    return stub_port_bundle()


@pytest.fixture
def client(api_settings: Settings, api_ports: PortBundle) -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = lambda: api_settings
    app.dependency_overrides[get_ports] = lambda: api_ports
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _auth(token: str = "test-token") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _run_to_review(settings: Settings, ports: PortBundle, project_id: str) -> None:
    pipeline = Pipeline(settings=settings, ports=ports)
    result = await pipeline.run(brief="counting ducks", mode=Mode.RHYME, project_id=project_id)
    assert result.status == ProjectStatus.REVIEW


def test_healthz_needs_no_auth(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_projects_401_without_token(client: TestClient) -> None:
    response = client.get("/projects")
    assert response.status_code == 401


def test_list_projects_401_with_wrong_token(client: TestClient) -> None:
    response = client.get("/projects", headers=_auth("wrong"))
    assert response.status_code == 401


async def test_list_and_get_project(
    client: TestClient, api_settings: Settings, api_ports: PortBundle
) -> None:
    project_id = "wp7-list-test"
    await _run_to_review(api_settings, api_ports, project_id)

    listed = client.get("/projects", headers=_auth())
    assert listed.status_code == 200
    ids = [p["project_id"] for p in listed.json()]
    assert project_id in ids

    detail = client.get(f"/projects/{project_id}", headers=_auth())
    assert detail.status_code == 200
    body = detail.json()
    assert body["status"] == "review"
    assert body["title"]
    assert len(body["qa_reports"]) == 6  # 5 scenes + 1 audio report
    assert any(a["kind"] == "final_video" for a in body["assets"])
    assert body["published_url"] is None


def test_get_project_404_when_missing(client: TestClient) -> None:
    response = client.get("/projects/does-not-exist", headers=_auth())
    assert response.status_code == 404


async def test_approve_resumes_graph_to_published(
    client: TestClient, api_settings: Settings, api_ports: PortBundle
) -> None:
    project_id = "wp7-approve-test"
    await _run_to_review(api_settings, api_ports, project_id)

    response = client.post(f"/projects/{project_id}/approve", headers=_auth())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "published"
    assert body["published_url"] is not None
    assert body["published_url"].startswith("https://stub.youtube.example/")


async def test_approve_409_when_not_in_review(
    client: TestClient, api_settings: Settings, api_ports: PortBundle
) -> None:
    project_id = "wp7-approve-twice"
    await _run_to_review(api_settings, api_ports, project_id)
    first = client.post(f"/projects/{project_id}/approve", headers=_auth())
    assert first.status_code == 200

    second = client.post(f"/projects/{project_id}/approve", headers=_auth())
    assert second.status_code == 409


async def test_reject_marks_rejected_without_publishing(
    client: TestClient, api_settings: Settings, api_ports: PortBundle
) -> None:
    project_id = "wp7-reject-test"
    await _run_to_review(api_settings, api_ports, project_id)

    response = client.post(f"/projects/{project_id}/reject", headers=_auth())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert body["published_url"] is None

    listed = client.get("/projects", headers=_auth())
    row = next(p for p in listed.json() if p["project_id"] == project_id)
    assert row["status"] == "rejected"


def test_trigger_run_returns_project_id(client: TestClient) -> None:
    response = client.post(
        "/runs", json={"brief": "counting ducks", "mode": "rhyme"}, headers=_auth()
    )
    assert response.status_code == 200
    assert "project_id" in response.json()
