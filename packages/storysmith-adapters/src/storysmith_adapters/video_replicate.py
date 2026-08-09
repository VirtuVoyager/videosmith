from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx
from storysmith.errors import ContentRejectedError, TransientError
from storysmith.settings import Settings
from storysmith.util.retry import with_retries

_API_BASE = "https://api.replicate.com/v1"
_POLL_INTERVAL_S = 5.0
_TIMEOUT_S = 600.0  # 10 minutes
_MAX_DURATION_S = 10.0

# SPEC-GAP: §3.1's "per-second price constant" isn't given a value in the
# spec; this is a placeholder until real Replicate billing data is on hand.
# Update freely -- it's a local constant, not a contract.
_PER_SECOND_PRICE_USD = 0.05
_FLAT_COST_USD = 0.5  # used when metrics.predict_time is absent

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_CONTENT_REJECTION_MARKERS = ("content", "flagged", "safety", "nsfw", "policy")


class ReplicateVideoGen:
    """VideoGenPort via Replicate's REST prediction API (§3.1)."""

    def __init__(self, settings: Settings) -> None:
        self._token = settings.replicate_api_token
        self._model_i2v = settings.video_model_i2v
        self._model_t2v = settings.video_model_t2v

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

        async with httpx.AsyncClient(
            base_url=_API_BASE,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=30.0,
        ) as client:
            prediction = await with_retries(lambda: self._submit(client, model, payload))
            prediction = await with_retries(lambda: self._poll_until_done(client, prediction["id"]))
            output_url = self._extract_output_url(prediction)
            video_bytes = await self._download(client, output_url)
        return video_bytes, self._cost(prediction)

    async def _submit(
        self, client: httpx.AsyncClient, model: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        resp = await client.post(f"/models/{model}/predictions", json={"input": payload})
        self._raise_for_transient(resp)
        resp.raise_for_status()
        return dict(resp.json())

    async def _poll_until_done(
        self, client: httpx.AsyncClient, prediction_id: str
    ) -> dict[str, Any]:
        deadline = asyncio.get_event_loop().time() + _TIMEOUT_S
        url = f"/predictions/{prediction_id}"
        while True:
            resp = await client.get(url)
            self._raise_for_transient(resp)
            resp.raise_for_status()
            data = dict(resp.json())
            status = data["status"]
            if status == "succeeded":
                return data
            if status == "failed":
                error = str(data.get("error", "")).lower()
                if any(marker in error for marker in _CONTENT_REJECTION_MARKERS):
                    raise ContentRejectedError(f"provider rejected prompt: {data.get('error')}")
                raise TransientError(f"prediction failed: {data.get('error')}")
            if status == "canceled":
                raise TransientError("prediction canceled")
            if asyncio.get_event_loop().time() > deadline:
                raise TransientError(f"prediction {prediction_id} timed out after {_TIMEOUT_S}s")
            await asyncio.sleep(_POLL_INTERVAL_S)

    @staticmethod
    def _raise_for_transient(resp: httpx.Response) -> None:
        if resp.status_code in _TRANSIENT_STATUS_CODES:
            raise TransientError(f"HTTP {resp.status_code} from Replicate: {resp.text[:200]}")

    @staticmethod
    def _extract_output_url(prediction: dict[str, Any]) -> str:
        output = prediction["output"]
        if isinstance(output, list):
            return str(output[0])
        return str(output)

    @staticmethod
    def _cost(prediction: dict[str, Any]) -> float:
        metrics = prediction.get("metrics") or {}
        predict_time = metrics.get("predict_time")
        if predict_time is not None:
            return float(predict_time) * _PER_SECOND_PRICE_USD
        return _FLAT_COST_USD

    @staticmethod
    async def _download(client: httpx.AsyncClient, url: str) -> bytes:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content
