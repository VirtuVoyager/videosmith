from __future__ import annotations

import asyncio
from typing import Any

import httpx
from storysmith.errors import ContentRejectedError, TransientError
from storysmith.util.retry import with_retries

API_BASE = "https://api.replicate.com/v1"

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_CONTENT_REJECTION_MARKERS = ("content", "flagged", "safety", "nsfw", "policy")


class ReplicatePoller:
    """Shared submit/poll/download logic for every Replicate-backed adapter
    (image, video, music, tts). Factored out here per §4 once a third and
    fourth consumer (music_replicate.py, tts_kokoro.py) needed the same
    submit -> poll -> download cycle already duplicated between
    image_replicate.py (WP2) and video_replicate.py (WP3).

    Callers own model selection and cost computation (those differ per
    adapter); this class only owns the REST mechanics and error mapping.
    """

    def __init__(
        self,
        *,
        token: str,
        poll_interval_s: float = 5.0,
        timeout_s: float = 600.0,
    ) -> None:
        self._token = token
        self._poll_interval_s = poll_interval_s
        self._timeout_s = timeout_s

    async def run(
        self, *, model: str, input_payload: dict[str, Any]
    ) -> tuple[dict[str, Any], bytes]:
        """Submit a prediction, poll until done, download the output.

        Returns (prediction, output_bytes) -- the full prediction dict is
        handed back so callers can read cost signals (e.g. metrics.predict_time).
        """
        prediction = await self._submit_and_poll(model=model, input_payload=input_payload)
        output_url = self._extract_output_url(prediction)
        output_bytes = await self._download(output_url)
        return prediction, output_bytes

    async def run_text(
        self, *, model: str, input_payload: dict[str, Any]
    ) -> tuple[dict[str, Any], str]:
        """Like run(), but for text-generation models (llm_replicate.py):
        `output` is a list of streamed text chunks to join (or, for some
        models, a single string already) rather than a media URL to
        download -- there's nothing to fetch from storage here.
        """
        prediction = await self._submit_and_poll(model=model, input_payload=input_payload)
        output = prediction["output"]
        text = "".join(output) if isinstance(output, list) else str(output)
        return prediction, text

    async def _submit_and_poll(
        self, *, model: str, input_payload: dict[str, Any]
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=API_BASE,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=30.0,
        ) as client:
            prediction = await with_retries(lambda: self._submit(client, model, input_payload))
            return await with_retries(lambda: self._poll_until_done(client, prediction["id"]))

    async def _submit(
        self, client: httpx.AsyncClient, model: str, input_payload: dict[str, Any]
    ) -> dict[str, Any]:
        resp = await client.post(f"/models/{model}/predictions", json={"input": input_payload})
        if resp.status_code == 404:
            # Not every model exposes the /models/{owner}/{name}/predictions
            # shorthand -- hit live for fishaudio/ace-step-1.5 (model exists,
            # GET /models/{model} succeeds, but this endpoint 404s). The
            # generic /predictions endpoint (pinned to a resolved version)
            # works for every model, so fall back to it instead of failing
            # the whole call over what's really just an API-shape mismatch.
            version = await self._resolve_latest_version(client, model)
            resp = await client.post(
                "/predictions", json={"version": version, "input": input_payload}
            )
        self._raise_for_transient(resp)
        resp.raise_for_status()
        return dict(resp.json())

    @staticmethod
    async def _resolve_latest_version(client: httpx.AsyncClient, model: str) -> str:
        resp = await client.get(f"/models/{model}")
        resp.raise_for_status()
        version = resp.json().get("latest_version", {}).get("id")
        if not version:
            raise TransientError(f"model {model!r} has no latest_version to resolve")
        return str(version)

    async def _poll_until_done(
        self, client: httpx.AsyncClient, prediction_id: str
    ) -> dict[str, Any]:
        deadline = asyncio.get_event_loop().time() + self._timeout_s
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
                raise TransientError(
                    f"prediction {prediction_id} timed out after {self._timeout_s}s"
                )
            await asyncio.sleep(self._poll_interval_s)

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
    async def _download(url: str) -> bytes:
        # Not a Replicate API call -- prediction outputs are served from a
        # presigned storage URL (S3/R2) that already carries its own auth in
        # the query string. Reusing the Replicate-authenticated client would
        # attach its default `Authorization: Bearer <replicate_token>` header
        # to this request too, which presigned URLs reject (400: signature/
        # auth mismatch), since presigned auth is meant to be the only auth
        # on the request.
        async with httpx.AsyncClient() as download_client:
            resp = await download_client.get(url)
            resp.raise_for_status()
            return resp.content
