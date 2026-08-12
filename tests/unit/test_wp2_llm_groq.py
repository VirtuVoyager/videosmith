from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import groq
import httpx
import pytest
from pydantic import BaseModel
from storysmith.settings import Settings
from storysmith_adapters.llm_groq import GroqLLM

pytestmark = pytest.mark.wp2


class _Widget(BaseModel):
    name: str


def _fake_response(args: dict[str, object]) -> SimpleNamespace:
    tool_call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="emit", arguments=json.dumps(args))
    )
    message = SimpleNamespace(tool_calls=[tool_call])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _bad_request_error(detail: str) -> groq.BadRequestError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request, json={"error": {"message": detail}})
    return groq.BadRequestError(detail, response=response, body=None)


async def test_regression_groq_bad_request_triggers_repair_not_crash(
    settings_test: Settings,
) -> None:
    """groq.BadRequestError (tool_use_failed) used to escape complete_structured
    entirely and crash the whole pipeline run instead of triggering the
    one-repair-round logic -- this pins that it's now treated as a
    recoverable schema-validation failure."""
    adapter = GroqLLM(settings_test)
    adapter._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        side_effect=[_bad_request_error("tool_use_failed"), _fake_response({"name": "ok"})]
    )

    parsed, cost = await adapter.complete_structured(system="sys", user="usr", schema=_Widget)

    assert parsed == _Widget(name="ok")
    assert cost == 0.0  # first (failed) call priced at 0.0, second uses empty PRICES table
    assert adapter._client.chat.completions.create.call_count == 2


async def test_regression_groq_bad_request_twice_raises_structured_output_error(
    settings_test: Settings,
) -> None:
    from storysmith.errors import LLMStructuredOutputError

    adapter = GroqLLM(settings_test)
    adapter._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _bad_request_error("tool_use_failed"),
            _bad_request_error("tool_use_failed"),
        ]
    )

    with pytest.raises(LLMStructuredOutputError):
        await adapter.complete_structured(system="sys", user="usr", schema=_Widget)
