from __future__ import annotations

import base64
from typing import Any

from storysmith.settings import Settings

from storysmith_adapters._replicate_base import ReplicatePoller

_MAX_DURATION_S = 10.0

# SPEC-GAP: §3.1's "per-second price constant" isn't given a value in the
# spec; this is a placeholder until real Replicate billing data is on hand.
# Update freely -- it's a local constant, not a contract.
_PER_SECOND_PRICE_USD = 0.05
_FLAT_COST_USD = 0.5  # used when metrics.predict_time is absent


class ReplicateVideoGen:
    """VideoGenPort via Replicate's REST prediction API (§3.1)."""

    def __init__(self, settings: Settings) -> None:
        self._model_i2v = settings.video_model_i2v
        self._model_t2v = settings.video_model_t2v
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
        payload: dict[str, Any] = {
            "prompt": prompt,
            "duration": min(duration_s, _MAX_DURATION_S),
            "aspect_ratio": aspect_ratio,
        }
        if reference_image is not None:
            b64 = base64.b64encode(reference_image).decode("ascii")
            payload["image"] = f"data:image/png;base64,{b64}"

        prediction, video_bytes = await self._poller.run(model=model, input_payload=payload)
        return video_bytes, self._cost(prediction)

    @staticmethod
    def _cost(prediction: dict[str, Any]) -> float:
        metrics = prediction.get("metrics") or {}
        predict_time = metrics.get("predict_time")
        if predict_time is not None:
            return float(predict_time) * _PER_SECOND_PRICE_USD
        return _FLAT_COST_USD
