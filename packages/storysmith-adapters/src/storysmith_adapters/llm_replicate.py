from __future__ import annotations

import base64
import json
import re

from pydantic import BaseModel, ValidationError
from storysmith.errors import LLMStructuredOutputError
from storysmith.settings import Settings
from storysmith.util.llm_schema import strict_tool_schema

from storysmith_adapters._replicate_base import ReplicatePoller

# USD per million tokens: (input, output). Replicate bills LLM predictions by
# metered token count (see metrics.token_input_count/token_output_count on
# every prediction) rather than the flat-per-call estimate used for image/
# video models (§3.1's cost formula is video-specific) -- but Replicate
# doesn't expose per-model list pricing through the API the way it publishes
# it on each model's web page, so these are hand-entered from that page and,
# like every other adapter's PRICES table, meant to be updated freely if a
# model's listed price changes.
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.05, 0.25),
}
_DEFAULT_PRICE = (0.10, 0.50)  # conservative guess for an unlisted model

# lucataco/qwen2-vl-7b-instruct isn't published with per-call list pricing
# (it's a small 7B model on cheap hardware) -- flat estimate, same "update
# freely" convention as image_replicate.py's _FLAT_COST_USD, until proven
# wrong. This model's own input schema is `{media: uri, prompt: str,
# max_new_tokens: int}` -- SPEC-GAP: that field name (`media`, not `image`)
# is specific to this model; swapping SS_REPLICATE_VISION_CAPTION_MODEL to a
# different model likely needs a matching change to _caption_one_image's
# input_payload below, this isn't a generic multi-model vision interface.
_VISION_CAPTION_COST_USD = 0.002
_VISION_CAPTION_PROMPT = (
    "Describe this image in detail for someone who cannot see it: the setting, "
    "any characters present and exactly what they look like (species/shape, "
    "clothing, colors, accessories), the art style, camera framing, and any "
    "visible defects, artifacts, or inconsistencies. Be concrete and specific."
)

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class ReplicateLLM:
    """LLMPort via Replicate-hosted open-weight instruct models.

    Unlike Anthropic/Groq, Replicate's LLM models are raw text-completion
    endpoints (a single `prompt` string in, streamed text chunks out) --
    no native chat roles, no tool-calling/forced function-calling, no JSON
    mode. Structured output is extracted by asking the model, in the prompt
    itself, to emit nothing but JSON matching a given JSON Schema, then
    validating the result with Pydantic and running the same one-shot
    repair round (re-send with the validation error appended) every other
    LLMPort adapter uses on a schema mismatch.

    This exists as a rate-limit escape hatch: Groq's free tier caps every
    current chat model at 8000 tokens/minute (confirmed live), too small for
    this project's own Director/Creative-Director requests regardless of
    which Groq model is selected. Replicate is metered pay-as-you-go with no
    such cap, and the same SS_REPLICATE_API_TOKEN this project already uses
    for image/video/music/TTS covers it.

    Vision (Critic's keyframe QA) works around the same "no multimodal chat
    API" limitation with a two-stage pipeline instead of a direct
    multi-image call: no raw-completion model on Replicate takes several
    images in one request the way Anthropic/Groq's chat-completions image
    blocks do, so `model_tier="vision"` first captions each image
    independently (SS_REPLICATE_VISION_CAPTION_MODEL, a real single-image
    vision model -- confirmed live against this project's actual character
    art), then folds those descriptions into `user` as plain text before
    running the exact same text-only structured-JSON extraction every other
    call uses. `critic.py` and its prompts need no changes: the prompt
    template already tells the model what order the images are shown in
    ("three keyframes ... followed by the character reference image"), and
    the numbered descriptions preserve that same order.
    """

    def __init__(self, settings: Settings) -> None:
        self._poller = ReplicatePoller(token=settings.replicate_api_token, timeout_s=300.0)
        self._settings = settings

    def _model_for_tier(self, model_tier: str) -> str:
        if model_tier == "vision":
            return self._settings.replicate_model_vision
        return self._settings.replicate_model_standard

    def _cost(self, model: str, prediction: dict[str, object]) -> float:
        metrics = prediction.get("metrics") or {}
        assert isinstance(metrics, dict)
        in_tokens = metrics.get("token_input_count", 0) or 0
        out_tokens = metrics.get("token_output_count", 0) or 0
        in_price, out_price = PRICES.get(model, _DEFAULT_PRICE)
        return (in_tokens * in_price + out_tokens * out_price) / 1_000_000

    async def _caption_one_image(self, image_bytes: bytes) -> tuple[str, float]:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"
        try:
            _prediction, text = await self._poller.run_text(
                model=self._settings.replicate_vision_caption_model,
                input_payload={
                    "media": data_uri,
                    "prompt": _VISION_CAPTION_PROMPT,
                    "max_new_tokens": 200,
                },
            )
        except Exception as exc:  # noqa: BLE001 -- one bad image shouldn't sink the whole QA call
            return f"[description unavailable: {exc}]", 0.0
        return text.strip(), _VISION_CAPTION_COST_USD

    async def _describe_images(self, user: str, images: list[bytes]) -> tuple[str, float]:
        results = [await self._caption_one_image(image) for image in images]
        total_cost = sum(cost for _, cost in results)
        described = "\n".join(
            f"Image {i + 1} description: {text}" for i, (text, _) in enumerate(results)
        )
        augmented_user = (
            f"{user}\n\n"
            "You cannot see images directly. Automated visual descriptions of each "
            "image, in the same order they were supplied, are given below -- use "
            "them exactly as if you had seen the images yourself:\n"
            f"{described}"
        )
        return augmented_user, total_cost

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        model_tier: str = "standard",
        images: list[bytes] | None = None,
    ) -> tuple[BaseModel, float]:
        model = self._model_for_tier(model_tier)
        vision_cost = 0.0
        if model_tier == "vision" and images:
            user, vision_cost = await self._describe_images(user, images)
        json_schema = json.dumps(strict_tool_schema(schema.model_json_schema()))
        prompt = _build_prompt(system=system, user=user, json_schema=json_schema)

        parsed, cost1, error, raw_text = await self._generate_and_parse(model, prompt, schema)
        if parsed is not None:
            return parsed, cost1 + vision_cost

        repair_prompt = (
            f"{prompt}\n\nYour previous response was invalid: {error}\n"
            f"Your previous response was:\n{raw_text}\n\n"
            "Re-emit ONLY the corrected JSON object, fixing that problem."
        )
        parsed2, cost2, error2, _ = await self._generate_and_parse(model, repair_prompt, schema)
        total_cost = cost1 + cost2 + vision_cost
        if parsed2 is None:
            raise LLMStructuredOutputError(
                f"failed to produce valid {schema.__name__} after repair round: {error2}"
            )
        return parsed2, total_cost

    async def _generate_and_parse(
        self, model: str, prompt: str, schema: type[BaseModel]
    ) -> tuple[BaseModel | None, float, str | None, str]:
        prediction, text = await self._poller.run_text(
            model=model,
            input_payload={"prompt": prompt, "max_tokens": 4096, "temperature": 0.2},
        )
        cost = self._cost(model, prediction)
        cleaned = _JSON_FENCE.sub("", text.strip())
        try:
            return schema.model_validate_json(cleaned), cost, None, text
        except ValidationError as exc:
            return None, cost, str(exc), text


def _build_prompt(*, system: str, user: str, json_schema: str) -> str:
    return (
        f"{system}\n\n{user}\n\n"
        "Respond with ONLY a single valid JSON object -- no markdown code fences, no prose "
        "before or after -- that strictly matches this JSON Schema:\n"
        f"{json_schema}"
    )
