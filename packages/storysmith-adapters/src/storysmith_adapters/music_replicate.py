from __future__ import annotations

from typing import Any

from storysmith.models import Mode
from storysmith.settings import Settings

from storysmith_adapters._replicate_base import ReplicatePoller

_MAX_DURATION_S = 240.0  # ACE-Step/MusicGen practical ceiling

# SPEC-GAP: no real Replicate billing data on hand yet for either model;
# placeholder pricing, update freely once real costs are known.
_PER_SECOND_PRICE_USD = 0.02
_FLAT_COST_USD = 0.3


class ReplicateMusicGen:
    """MusicGenPort via Replicate: ACE-Step for rhyme mode, MusicGen instrumental
    for topical mode (§4), sharing the poll pattern from §3.1."""

    def __init__(self, settings: Settings) -> None:
        self._model_rhyme = settings.music_model
        self._model_topical = settings.music_model_instrumental
        self._poller = ReplicatePoller(token=settings.replicate_api_token)

    async def generate(
        self,
        *,
        mode: Mode,
        lyrics: str | None,
        description: str,
        duration_s: float,
    ) -> tuple[bytes, float]:
        duration = min(duration_s, _MAX_DURATION_S)
        payload: dict[str, Any]
        if mode == Mode.RHYME:
            model = self._model_rhyme
            payload = {"lyrics": lyrics or "", "tags": description, "duration": duration}
        else:
            model = self._model_topical
            payload = {"prompt": description, "duration": duration}

        prediction, audio_bytes = await self._poller.run(model=model, input_payload=payload)
        return audio_bytes, self._cost(prediction)

    @staticmethod
    def _cost(prediction: dict[str, Any]) -> float:
        metrics = prediction.get("metrics") or {}
        predict_time = metrics.get("predict_time")
        if predict_time is not None:
            return float(predict_time) * _PER_SECOND_PRICE_USD
        return _FLAT_COST_USD
