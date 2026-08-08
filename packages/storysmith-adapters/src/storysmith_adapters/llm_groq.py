from __future__ import annotations

import base64
import json

import groq
from pydantic import BaseModel, ValidationError
from storysmith.errors import LLMStructuredOutputError, TransientError
from storysmith.settings import Settings
from storysmith.util.retry import with_retries

# USD per million tokens: (input, output). Left at 0.0 by default -- Groq's
# free developer tier is the whole point of this adapter (null-cost testing
# before switching SS_LLM_PROVIDER back to anthropic). Fill in real per-model
# prices here if/when the account moves to a paid tier.
PRICES: dict[str, tuple[float, float]] = {}
_DEFAULT_PRICE = (0.0, 0.0)

_TRANSIENT_EXCEPTIONS = (
    groq.RateLimitError,
    groq.APIConnectionError,
    groq.APITimeoutError,
    groq.InternalServerError,
)


class GroqLLM:
    """LLMPort via Groq's OpenAI-compatible chat API, forced tool-call structured output."""

    def __init__(self, settings: Settings) -> None:
        self._client = groq.AsyncGroq(api_key=settings.groq_api_key)
        self._settings = settings

    def _model_for_tier(self, model_tier: str) -> str:
        if model_tier == "vision":
            return self._settings.groq_model_vision
        return self._settings.groq_model_standard

    def _cost(self, model: str, usage: object) -> float:
        in_price, out_price = PRICES.get(model, _DEFAULT_PRICE)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000

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
        user_content: list[dict[str, object]] = [{"type": "text", "text": user}]
        for image_bytes in images or []:
            b64 = base64.b64encode(image_bytes).decode("ascii")
            user_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )
        tool = {
            "type": "function",
            "function": {
                "name": "emit",
                "description": f"Emit a valid {schema.__name__}.",
                "parameters": schema.model_json_schema(),
            },
        }
        messages: list[dict[str, object]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        async def _call(msgs: list[dict[str, object]]) -> groq.types.chat.ChatCompletion:
            try:
                return await self._client.chat.completions.create(
                    model=model,
                    messages=msgs,  # type: ignore[arg-type]
                    tools=[tool],  # type: ignore[list-item]
                    tool_choice={"type": "function", "function": {"name": "emit"}},
                )
            except _TRANSIENT_EXCEPTIONS as exc:
                raise TransientError(str(exc)) from exc

        response = await with_retries(lambda: _call(messages))
        parsed, cost, error = self._parse(response, model, schema)
        if parsed is not None:
            return parsed, cost

        repair_messages = [
            *messages,
            {"role": "assistant", "content": None, "tool_calls": _tool_calls(response)},
            {
                "role": "tool",
                "tool_call_id": _tool_calls(response)[0]["id"],
                "content": f"Validation error, fix and resend: {error}",
            },
        ]
        response2 = await with_retries(lambda: _call(repair_messages))
        parsed2, cost2, error2 = self._parse(response2, model, schema)
        total_cost = cost + cost2
        if parsed2 is None:
            raise LLMStructuredOutputError(
                f"failed to produce valid {schema.__name__} after repair round: {error2}"
            )
        return parsed2, total_cost

    def _parse(
        self, response: groq.types.chat.ChatCompletion, model: str, schema: type[BaseModel]
    ) -> tuple[BaseModel | None, float, str | None]:
        cost = self._cost(model, response.usage)
        tool_calls = _tool_calls(response)
        if not tool_calls:
            return None, cost, "model did not call the emit tool"
        try:
            args = json.loads(tool_calls[0]["function"]["arguments"])
            return schema.model_validate(args), cost, None
        except (ValidationError, json.JSONDecodeError) as exc:
            return None, cost, str(exc)


def _tool_calls(response: groq.types.chat.ChatCompletion) -> list[dict[str, object]]:
    message = response.choices[0].message
    calls = message.tool_calls or []
    return [
        {
            "id": c.id,
            "type": "function",
            "function": {"name": c.function.name, "arguments": c.function.arguments},
        }
        for c in calls
    ]
