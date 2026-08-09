from __future__ import annotations

import httpx
import pytest
import respx
from storysmith.errors import ContentRejectedError
from storysmith.settings import Settings
from storysmith_adapters.video_replicate import ReplicateVideoGen

pytestmark = pytest.mark.wp3

_BASE = "https://api.replicate.com/v1"


async def _instant_sleep(_seconds: float) -> None:
    return None


@pytest.fixture
def settings_video() -> Settings:
    return Settings(
        _env_file=None,
        replicate_api_token="tok",
        video_model_i2v="wan-video/wan-2.2-i2v-fast",
        video_model_t2v="wan-video/wan-2.2-t2v-fast",
    )


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("storysmith_adapters.video_replicate.asyncio.sleep", _instant_sleep)
    monkeypatch.setattr("storysmith.util.retry.asyncio.sleep", _instant_sleep)


async def test_poll_until_succeeded(settings_video: Settings) -> None:
    gen = ReplicateVideoGen(settings_video)
    with respx.mock(base_url=_BASE) as mock:
        mock.post("/models/wan-video/wan-2.2-t2v-fast/predictions").mock(
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
                        "metrics": {"predict_time": 4.0},
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
        assert cost == pytest.approx(4.0 * 0.05)


async def test_transient_retry_then_success(settings_video: Settings) -> None:
    gen = ReplicateVideoGen(settings_video)
    with respx.mock(base_url=_BASE) as mock:
        mock.post("/models/wan-video/wan-2.2-t2v-fast/predictions").mock(
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
        assert cost == pytest.approx(0.5)  # no metrics.predict_time -> flat cost fallback


async def test_content_rejected_not_retried(settings_video: Settings) -> None:
    gen = ReplicateVideoGen(settings_video)
    with respx.mock(base_url=_BASE) as mock:
        submit_route = mock.post("/models/wan-video/wan-2.2-t2v-fast/predictions").mock(
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
