from __future__ import annotations

import asyncio
from typing import Any

import httpx
from storysmith.errors import ContentRejectedError, TransientError
from storysmith.settings import Settings
from storysmith.util.retry import with_retries

_API_BASE = "https://api.replicate.com/v1"
_POLL_INTERVAL_S = 2.0
_TIMEOUT_S = 120.0
# Flat per-image cost estimate for flux-schnell. Replicate doesn't surface a
# per-second predict_time cost signal for image models the way it does for
# video (§3.1's cost formula is video-specific); update this constant freely.
_FLAT_COST_USD = 0.003

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_CONTENT_REJECTION_MARKERS = ("content", "flagged", "safety", "nsfw")


class ReplicateImageGen:
    """ImageGenPort via Replicate's REST prediction API (§2.3, poll pattern per §3.1)."""

    def __init__(self, settings: Settings) -> None:
        self._token = settings.replicate_api_token
        self._model = settings.image_model

    async def generate(self, *, prompt: str, aspect_ratio: str) -> tuple[bytes, float]:
        async with httpx.AsyncClient(
            base_url=_API_BASE,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=30.0,
        ) as client:
            prediction = await with_retries(lambda: self._submit(client, prompt, aspect_ratio))
            prediction = await with_retries(lambda: self._poll_until_done(client, prediction["id"]))
            output_url = self._extract_output_url(prediction)
            image_bytes = await self._download(client, output_url)
        return image_bytes, _FLAT_COST_USD

    async def _submit(
        self, client: httpx.AsyncClient, prompt: str, aspect_ratio: str
    ) -> dict[str, Any]:
        resp = await client.post(
            f"/models/{self._model}/predictions",
            json={"input": {"prompt": prompt, "aspect_ratio": aspect_ratio}},
        )
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
    async def _download(client: httpx.AsyncClient, url: str) -> bytes:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content
