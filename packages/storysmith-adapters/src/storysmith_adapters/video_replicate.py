from __future__ import annotations

import base64
from typing import Any

from storysmith.settings import Settings

from storysmith_adapters._replicate_base import ReplicatePoller

_MAX_DURATION_S = 10.0  # our own Scene.duration_s ceiling, not a model limit

# prunaai/p-video bills per second of *requested output duration* at a flat
# rate per resolution tier (confirmed against Replicate's published pricing,
# live-tested) -- not metrics.predict_time (GPU wall-clock), which runs far
# shorter than output length for this model and would badly understate cost
# here. $0.02/sec @720p (this constant, matching settings.video_resolution's
# default) or $0.04/sec @1080p -- update this constant if video_resolution is
# changed to 1080p. Draft mode (not used here) is ~5-10x cheaper still.
_PER_SECOND_PRICE_USD = 0.02


class ReplicateVideoGen:
    """VideoGenPort via Replicate's REST prediction API (§3.1)."""

    def __init__(self, settings: Settings) -> None:
        self._model_i2v = settings.video_model_i2v
        self._model_t2v = settings.video_model_t2v
        self._resolution = settings.video_resolution
        self._poller = ReplicatePoller(token=settings.replicate_api_token)

    async def generate(
        self,
        *,
        prompt: str,
        duration_s: float,
        aspect_ratio: str,
        reference_image: bytes | None,
    ) -> tuple[bytes, float]:
        model = self._model_i2v if reference_image is not None else self._model_t2v
        duration = int(round(min(duration_s, _MAX_DURATION_S)))
        payload: dict[str, Any] = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "resolution": self._resolution,
        }
        if reference_image is not None:
            b64 = base64.b64encode(reference_image).decode("ascii")
            payload["image"] = f"data:image/png;base64,{b64}"

        _prediction, video_bytes = await self._poller.run(model=model, input_payload=payload)
        return video_bytes, duration * _PER_SECOND_PRICE_USD
