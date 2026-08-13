from __future__ import annotations

import httpx
import pytest
import respx
from storysmith.errors import ContentRejectedError
from storysmith.settings import Settings
from storysmith_adapters.video_replicate import ReplicateVideoGen

pytestmark = pytest.mark.wp3

_BASE = "https://api.replicate.com/v1"
# Provider-agnostic ids: keeps this test decoupled from whichever real
# provider settings.video_model_{i2v,t2v} default to.
_T2V_MODEL = "test-provider/t2v-model"
_I2V_MODEL = "test-provider/i2v-model"


async def _instant_sleep(_seconds: float) -> None:
    return None


@pytest.fixture
def settings_video() -> Settings:
    return Settings(
        _env_file=None,
        replicate_api_token="tok",
        video_model_i2v=_I2V_MODEL,
        video_model_t2v=_T2V_MODEL,
    )


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("storysmith_adapters._replicate_base.asyncio.sleep", _instant_sleep)
    monkeypatch.setattr("storysmith.util.retry.asyncio.sleep", _instant_sleep)


async def test_poll_until_succeeded(settings_video: Settings) -> None:
    gen = ReplicateVideoGen(settings_video)
    with respx.mock(base_url=_BASE) as mock:
        mock.post(f"/models/{_T2V_MODEL}/predictions").mock(
            return_value=httpx.Response(201, json={"id": "p1", "status": "starting"})
        )
        mock.get("/predictions/p1").mock(
            side_effect=[
                httpx.Response(200, json={"id": "p1", "status": "processing"}),
                httpx.Response(200, json={"id": "p1", "status": "processing"}),
                httpx.Response(
                    200,
                    json={
                        "id": "p1",
                        "status": "succeeded",
                        "output": ["https://example.com/v.mp4"],
                    },
                ),
            ]
        )
        mock.get("https://example.com/v.mp4").mock(
            return_value=httpx.Response(200, content=b"MP4DATA")
        )

        data, cost = await gen.generate(
            prompt="a duck swims", duration_s=8.0, aspect_ratio="9:16", reference_image=None
        )

        assert data == b"MP4DATA"
        assert cost == pytest.approx(8 * 0.05)  # billed by requested output duration, not GPU time


async def test_transient_retry_then_success(settings_video: Settings) -> None:
    gen = ReplicateVideoGen(settings_video)
    with respx.mock(base_url=_BASE) as mock:
        mock.post(f"/models/{_T2V_MODEL}/predictions").mock(
            side_effect=[
                httpx.Response(429, text="rate limited"),
                httpx.Response(201, json={"id": "p2", "status": "starting"}),
            ]
        )
        mock.get("/predictions/p2").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "p2",
                    "status": "succeeded",
                    "output": ["https://example.com/v2.mp4"],
                },
            )
        )
        mock.get("https://example.com/v2.mp4").mock(return_value=httpx.Response(200, content=b"V2"))

        data, cost = await gen.generate(
            prompt="a duck swims", duration_s=5.0, aspect_ratio="9:16", reference_image=None
        )

        assert data == b"V2"
        assert cost == pytest.approx(5 * 0.05)


async def test_content_rejected_not_retried(settings_video: Settings) -> None:
    gen = ReplicateVideoGen(settings_video)
    with respx.mock(base_url=_BASE) as mock:
        submit_route = mock.post(f"/models/{_T2V_MODEL}/predictions").mock(
            return_value=httpx.Response(201, json={"id": "p3", "status": "starting"})
        )
        mock.get("/predictions/p3").mock(
            return_value=httpx.Response(
                200,
                json={"id": "p3", "status": "failed", "error": "flagged for unsafe content"},
            )
        )

        with pytest.raises(ContentRejectedError):
            await gen.generate(
                prompt="something bad", duration_s=5.0, aspect_ratio="9:16", reference_image=None
            )

        assert submit_route.call_count == 1  # not retried


async def test_regression_download_does_not_leak_replicate_auth_header(
    settings_video: Settings,
) -> None:
    """Prediction outputs are served from a presigned storage URL (S3/R2)
    that carries its own auth in the query string -- forwarding the client's
    `Authorization: Bearer <replicate_token>` header there breaks presigned-
    URL signature validation (a real 400 hit live: the same reused httpx
    client was sending its default header to a Cloudflare R2 URL)."""
    gen = ReplicateVideoGen(settings_video)
    with respx.mock(base_url=_BASE) as mock:
        mock.post(f"/models/{_T2V_MODEL}/predictions").mock(
            return_value=httpx.Response(201, json={"id": "p5", "status": "starting"})
        )
        mock.get("/predictions/p5").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "p5",
                    "status": "succeeded",
                    "output": ["https://cdn.example.com/v5.mp4"],
                },
            )
        )
        download_route = mock.get("https://cdn.example.com/v5.mp4").mock(
            return_value=httpx.Response(200, content=b"V5")
        )

        await gen.generate(
            prompt="a duck swims", duration_s=5.0, aspect_ratio="9:16", reference_image=None
        )

        assert download_route.call_count == 1
        assert "authorization" not in download_route.calls[0].request.headers


async def test_i2v_model_selected_when_reference_image_present(settings_video: Settings) -> None:
    gen = ReplicateVideoGen(settings_video)
    with respx.mock(base_url=_BASE) as mock:
        submit_route = mock.post(f"/models/{_I2V_MODEL}/predictions").mock(
            return_value=httpx.Response(201, json={"id": "p4", "status": "starting"})
        )
        mock.get("/predictions/p4").mock(
            return_value=httpx.Response(
                200,
                json={"id": "p4", "status": "succeeded", "output": "https://example.com/v4.mp4"},
            )
        )
        mock.get("https://example.com/v4.mp4").mock(return_value=httpx.Response(200, content=b"V4"))

        await gen.generate(
            prompt="a duck swims",
            duration_s=5.0,
            aspect_ratio="9:16",
            reference_image=b"REF_IMAGE_BYTES",
        )

        assert submit_route.call_count == 1
        sent_payload = submit_route.calls[0].request.content
        assert b"REF_IMAGE_BYTES" not in sent_payload  # base64-encoded, not raw bytes
        assert b"image" in sent_payload
