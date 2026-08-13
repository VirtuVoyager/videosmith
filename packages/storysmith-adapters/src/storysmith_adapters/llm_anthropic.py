from __future__ import annotations

import base64

import anthropic
from pydantic import BaseModel, ValidationError
from storysmith.errors import LLMStructuredOutputError, TransientError
from storysmith.settings import Settings
from storysmith.util.llm_schema import strict_tool_schema
from storysmith.util.retry import with_retries

# USD per million tokens: (input, output). Update freely as pricing changes (§2.1).
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
}
_DEFAULT_PRICE = (3.0, 15.0)

_TRANSIENT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


class AnthropicLLM:
    """LLMPort via Anthropic's tool-use API, forced structured output (§2.1)."""

    def __init__(self, settings: Settings) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._settings = settings

    def _model_for_tier(self, model_tier: str) -> str:
        if model_tier == "vision":
            return self._settings.anthropic_model_vision
        return self._settings.anthropic_model_standard

    def _cost(self, model: str, usage: anthropic.types.Usage) -> float:
        in_price, out_price = PRICES.get(model, _DEFAULT_PRICE)
        return (usage.input_tokens * in_price + usage.output_tokens * out_price) / 1_000_000

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
        content: list[anthropic.types.ContentBlockParam] = [{"type": "text", "text": user}]
        for image_bytes in images or []:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(image_bytes).decode("ascii"),
                    },
                }
            )
        tool: anthropic.types.ToolParam = {
            "name": "emit",
            "description": f"Emit a valid {schema.__name__}.",
            "input_schema": strict_tool_schema(schema.model_json_schema()),
        }
        messages: list[anthropic.types.MessageParam] = [{"role": "user", "content": content}]

        async def _call(msgs: list[anthropic.types.MessageParam]) -> anthropic.types.Message:
            try:
                return await self._client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system,
                    messages=msgs,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": "emit"},
                )
            except _TRANSIENT_EXCEPTIONS as exc:
                raise TransientError(str(exc)) from exc

        response = await with_retries(lambda: _call(messages))
        parsed, cost, error = self._parse(response, model, schema)
        if parsed is not None:
            return parsed, cost

        tool_use_block = next(b for b in response.content if b.type == "tool_use")
        repair_messages = [
            *messages,
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": f"Validation error, fix and resend: {error}",
                        "is_error": True,
                    }
                ],
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
        self, response: anthropic.types.Message, model: str, schema: type[BaseModel]
    ) -> tuple[BaseModel | None, float, str | None]:
        cost = self._cost(model, response.usage)
        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is None:
            return None, cost, "model did not call the emit tool"
        try:
            return schema.model_validate(tool_use.input), cost, None
        except ValidationError as exc:
            return None, cost, str(exc)
