from __future__ import annotations

from storysmith.settings import Settings

from storysmith_adapters._replicate_base import ReplicatePoller

# Flat per-image cost estimate for flux-schnell. Replicate doesn't surface a
# per-second predict_time cost signal for image models the way it does for
# video (§3.1's cost formula is video-specific); update this constant freely.
_FLAT_COST_USD = 0.003


class ReplicateImageGen:
    """ImageGenPort via Replicate's REST prediction API (§2.3, poll pattern per §3.1)."""

    def __init__(self, settings: Settings) -> None:
        self._model = settings.image_model
        self._poller = ReplicatePoller(token=settings.replicate_api_token)

    async def generate(self, *, prompt: str, aspect_ratio: str) -> tuple[bytes, float]:
        _prediction, image_bytes = await self._poller.run(
            model=self._model,
            input_payload={"prompt": prompt, "aspect_ratio": aspect_ratio},
        )
        return image_bytes, _FLAT_COST_USD
