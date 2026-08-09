from __future__ import annotations

from storysmith.settings import Settings

from storysmith_adapters._replicate_base import ReplicatePoller

# SPEC-GAP: no real Replicate billing data on hand yet; Kokoro is a small,
# fast model so a flat per-call estimate is used rather than predict_time
# scaling. Update freely once real costs are known.
_FLAT_COST_USD = 0.01


class KokoroTTS:
    """TTSPort via Kokoro on Replicate, for topical-mode narration (§4)."""

    def __init__(self, settings: Settings) -> None:
        self._model = settings.tts_model
        self._poller = ReplicatePoller(token=settings.replicate_api_token)

    async def speak(self, *, text: str, voice: str) -> tuple[bytes, float]:
        _prediction, audio_bytes = await self._poller.run(
            model=self._model,
            input_payload={"text": text, "voice": voice},
        )
        return audio_bytes, _FLAT_COST_USD
