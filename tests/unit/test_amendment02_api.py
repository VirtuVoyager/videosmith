from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from api.main import app, get_ports, get_settings
from fastapi.testclient import TestClient
from storysmith.pipeline import PortBundle
from storysmith.settings import Settings
from storysmith_adapters.stubs import stub_port_bundle

pytestmark = pytest.mark.amendment02


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


# NOTE: never hardcode a real show's name/id here (e.g. "bob-and-miko") --
# these tests hit real Postgres via pg_required, and db.save_show is a
# full-overwrite upsert, so a shared literal would clobber that show's real
# data on every test run. _show_payload() takes a fresh id per test instead.
def _show_payload(show_id: str) -> dict[str, object]:
    return {
        "show_id": show_id,
        "name": "Bob & Miko",
        "art_style": "soft 2D cutout animation",
        "characters": [
            {
                "name": "Bob",
                "description": "an orange cat with a chipped ear",
                "personality": "sarcastic and lazy",
                "voice_id": "am_adam",
            },
            {
                "name": "Miko",
                "description": "a golden retriever with a red bandana",
                "personality": "earnest and easily excited",
                "voice_id": "af_bella",
            },
        ],
    }


def test_create_show_requires_auth(client: TestClient) -> None:
    response = client.post("/shows", json=_show_payload("unauthed-show"))
    assert response.status_code == 401  # rejected before any write happens


def test_create_show_generates_and_freezes_avatars(
    client: TestClient, shows_cleanup: list[str]
) -> None:
    show_id = f"show-{uuid.uuid4()}"
    shows_cleanup.append(show_id)
    response = client.post("/shows", json=_show_payload(show_id), headers=_auth())

    assert response.status_code == 200
    body = response.json()
    assert body["show_id"] == show_id
    assert {c["name"] for c in body["characters"]} == {"Bob", "Miko"}
    for character in body["characters"]:
        assert character["image_asset_uri"]  # an avatar was actually generated


def test_create_show_rejects_empty_cast(client: TestClient) -> None:
    payload = {**_show_payload("empty-cast"), "characters": []}
    response = client.post("/shows", json=payload, headers=_auth())
    assert response.status_code == 422  # rejected before any write happens


def test_list_shows_includes_created_show(client: TestClient, shows_cleanup: list[str]) -> None:
    show_id = f"show-{uuid.uuid4()}"
    shows_cleanup.append(show_id)
    client.post("/shows", json=_show_payload(show_id), headers=_auth())

    response = client.get("/shows", headers=_auth())

    assert response.status_code == 200
    ids = [s["show_id"] for s in response.json()]
    assert show_id in ids


def test_trigger_run_with_unknown_show_id_404s_synchronously(client: TestClient) -> None:
    response = client.post(
        "/runs",
        json={"brief": "a topic", "mode": "topical", "show_id": "does-not-exist"},
        headers=_auth(),
    )
    assert response.status_code == 404


def test_trigger_run_with_known_show_id_returns_project_id(
    client: TestClient, shows_cleanup: list[str]
) -> None:
    show_id = f"show-{uuid.uuid4()}"
    shows_cleanup.append(show_id)
    create = client.post("/shows", json=_show_payload(show_id), headers=_auth())
    assert create.status_code == 200

    response = client.post(
        "/runs",
        json={"brief": "a topic", "mode": "topical", "show_id": show_id},
        headers=_auth(),
    )

    assert response.status_code == 200
    assert "project_id" in response.json()


def test_view_asset_requires_auth(client: TestClient) -> None:
    response = client.get("/assets/view", params={"uri": "stub://x"})
    assert response.status_code == 401


def test_view_asset_streams_bytes(client: TestClient, api_ports: PortBundle) -> None:
    import asyncio

    uri = asyncio.run(
        api_ports.storage.put(key="test.png", data=b"IMAGEBYTES", content_type="image/png")
    )

    response = client.get("/assets/view", params={"uri": uri}, headers=_auth())

    assert response.status_code == 200
    assert response.content == b"IMAGEBYTES"


def test_view_asset_404s_for_missing_uri(client: TestClient) -> None:
    response = client.get(
        "/assets/view", params={"uri": "stub://does/not/exist.png"}, headers=_auth()
    )
    assert response.status_code == 404
