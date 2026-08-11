from __future__ import annotations

import base64
from typing import Any

from storysmith.settings import Settings

from storysmith_adapters._replicate_base import ReplicatePoller

_MAX_DURATION_S = 10.0  # our own Scene.duration_s ceiling, not a model limit

# xai/grok-imagine-video bills $0.05 per second of *requested output
# duration*, confirmed against Replicate's own pricing -- not
# metrics.predict_time (GPU wall-clock), which can run 5-30x longer than the
# output length for video diffusion models and would badly overstate cost
# here. Not confirmed whether 480p vs 720p changes the per-second rate;
# treated as flat until proven otherwise -- update if that's wrong.
_PER_SECOND_PRICE_USD = 0.05


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
