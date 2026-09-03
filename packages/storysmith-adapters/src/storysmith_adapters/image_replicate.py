from __future__ import annotations

import base64

from storysmith.settings import Settings

from storysmith_adapters._replicate_base import ReplicatePoller

# Flat per-image cost estimates. Replicate doesn't surface a per-second
# predict_time cost signal for image models the way it does for video
# (§3.1's cost formula is video-specific); update these constants freely.
_FLAT_COST_USD = 0.003  # flux-schnell (text-only)
_REFERENCE_COST_USD = 0.04  # flux-kontext-pro (image-conditioned) -- estimate,
# Replicate's own published per-image price for this model; not yet
# confirmed against a real invoice, unlike flux-schnell's number above.


class ReplicateImageGen:
    """ImageGenPort via Replicate's REST prediction API (§2.3, poll pattern per §3.1).

    Same model-selection-by-argument-presence pattern video_replicate.py
    already uses for i2v/t2v: a plain prompt routes to `image_model` (flux-
    schnell, pure text-to-image, used for char_refs/POST /shows -- there's
    no prior appearance to condition on when generating the reference sheet
    itself); a `reference_image` routes to `scene_image_model` (flux-
    kontext-pro, an image-editing model: `input_image` + a text edit
    instruction, not independent text-to-image) -- used by scene_stills.py
    to keep a character's appearance consistent with its frozen reference
    sheet instead of re-imagining it from text alone every scene (Amendment
    03; confirmed live that text-only generation renders a visibly
    different-looking character on almost every independent call).
    """

    def __init__(self, settings: Settings) -> None:
        self._model = settings.image_model
        self._reference_model = settings.scene_image_model
        self._poller = ReplicatePoller(token=settings.replicate_api_token)

    async def generate(
        self, *, prompt: str, aspect_ratio: str, reference_image: bytes | None = None
    ) -> tuple[bytes, float]:
        if reference_image is not None:
            b64 = base64.b64encode(reference_image).decode("ascii")
            input_payload = {
                "prompt": prompt,
                "input_image": f"data:image/png;base64,{b64}",
                # Explicit, not the "match_input_image" default: the
                # reference sheet is a wide turnaround (3:2, see
                # character_prompts.CHAR_REF_ASPECT_RATIO), but the scene
                # still itself must be the show's actual video aspect ratio.
                "aspect_ratio": aspect_ratio,
            }
            _prediction, image_bytes = await self._poller.run(
                model=self._reference_model, input_payload=input_payload
            )
            return image_bytes, _REFERENCE_COST_USD

        _prediction, image_bytes = await self._poller.run(
            model=self._model,
            input_payload={"prompt": prompt, "aspect_ratio": aspect_ratio},
        )
        return image_bytes, _FLAT_COST_USD
