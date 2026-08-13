from __future__ import annotations

import httpx
import pytest
import respx
from storysmith.models import Mode
from storysmith.settings import Settings
from storysmith_adapters.music_replicate import ReplicateMusicGen
from storysmith_adapters.tts_kokoro import KokoroTTS

pytestmark = pytest.mark.wp4

_BASE = "https://api.replicate.com/v1"


async def _instant_sleep(_seconds: float) -> None:
    return None


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both adapters funnel through the same shared ReplicatePoller (§4) --
    # patching it once here proves the reuse, rather than patching each
    # adapter module separately as WP2/WP3's now-refactored tests did.
    monkeypatch.setattr("storysmith_adapters._replicate_base.asyncio.sleep", _instant_sleep)
    monkeypatch.setattr("storysmith.util.retry.asyncio.sleep", _instant_sleep)


@pytest.fixture
def settings_music() -> Settings:
    return Settings(
        _env_file=None,
        replicate_api_token="tok",
        music_model="fishaudio/ace-step-1.5",
        music_model_instrumental="meta/musicgen",
        tts_model="jaaari/kokoro-82m",
    )


async def test_music_replicate_poller_reuse_rhyme(settings_music: Settings) -> None:
    gen = ReplicateMusicGen(settings_music)
    with respx.mock(base_url=_BASE) as mock:
        mock.post("/models/fishaudio/ace-step-1.5/predictions").mock(
            return_value=httpx.Response(201, json={"id": "m1", "status": "starting"})
        )
        mock.get("/predictions/m1").mock(
            side_effect=[
                httpx.Response(200, json={"id": "m1", "status": "processing"}),
                httpx.Response(
                    200,
                    json={
                        "id": "m1",
                        "status": "succeeded",
                        "output": ["https://example.com/song.mp3"],
                        "metrics": {"predict_time": 10.0},
                    },
                ),
            ]
        )
        mock.get("https://example.com/song.mp3").mock(
            return_value=httpx.Response(200, content=b"SONG")
        )

        data, cost = await gen.generate(
            mode=Mode.RHYME, lyrics="one two three", description="cheerful", duration_s=40.0
        )

        assert data == b"SONG"
        assert cost == pytest.approx(10.0 * 0.02)


async def test_music_replicate_poller_reuse_topical(settings_music: Settings) -> None:
    gen = ReplicateMusicGen(settings_music)
    with respx.mock(base_url=_BASE) as mock:
        mock.post("/models/meta/musicgen/predictions").mock(
            return_value=httpx.Response(201, json={"id": "m2", "status": "starting"})
        )
        mock.get("/predictions/m2").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "m2",
                    "status": "succeeded",
                    "output": "https://example.com/bed.mp3",
                },
            )
        )
        mock.get("https://example.com/bed.mp3").mock(
            return_value=httpx.Response(200, content=b"BED")
        )

        data, cost = await gen.generate(
            mode=Mode.TOPICAL, lyrics=None, description="instrumental bed", duration_s=40.0
        )

        assert data == b"BED"
        assert cost == pytest.approx(0.3)  # no metrics.predict_time -> flat fallback


async def test_kokoro_tts_poller_reuse(settings_music: Settings) -> None:
    tts = KokoroTTS(settings_music)
    with respx.mock(base_url=_BASE) as mock:
        mock.post("/models/jaaari/kokoro-82m/predictions").mock(
            return_value=httpx.Response(201, json={"id": "t1", "status": "starting"})
        )
        mock.get("/predictions/t1").mock(
            return_value=httpx.Response(
                200,
                json={"id": "t1", "status": "succeeded", "output": ["https://example.com/v.mp3"]},
            )
        )
        mock.get("https://example.com/v.mp3").mock(return_value=httpx.Response(200, content=b"HI"))

        data, cost = await tts.speak(text="one two three", voice="af_bella")

        assert data == b"HI"
        assert cost == pytest.approx(0.01)
