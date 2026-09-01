from __future__ import annotations

import httpx
import pytest
import respx
from pydantic import BaseModel
from storysmith.errors import LLMStructuredOutputError
from storysmith.settings import Settings
from storysmith_adapters.llm_replicate import ReplicateLLM

pytestmark = pytest.mark.wp2

_BASE = "https://api.replicate.com/v1"


async def _instant_sleep(_seconds: float) -> None:
    return None


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("storysmith_adapters._replicate_base.asyncio.sleep", _instant_sleep)
    monkeypatch.setattr("storysmith.util.retry.asyncio.sleep", _instant_sleep)


@pytest.fixture
def settings_llm() -> Settings:
    return Settings(
        _env_file=None,
        replicate_api_token="tok",
        replicate_model_standard="openai/gpt-oss-120b",
        replicate_model_vision="openai/gpt-oss-120b",
    )


class _Greeting(BaseModel):
    greeting: str
    count: int


async def test_valid_json_on_first_try(settings_llm: Settings) -> None:
    llm = ReplicateLLM(settings_llm)
    with respx.mock(base_url=_BASE) as mock:
        mock.post("/models/openai/gpt-oss-120b/predictions").mock(
            return_value=httpx.Response(201, json={"id": "p1", "status": "starting"})
        )
        mock.get("/predictions/p1").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "p1",
                    "status": "succeeded",
                    "output": ['{"greeting":"hi","count":3}'],
                    "metrics": {"token_input_count": 100, "token_output_count": 20},
                },
            )
        )

        parsed, cost = await llm.complete_structured(system="sys", user="say hi", schema=_Greeting)

        assert parsed == _Greeting(greeting="hi", count=3)
        assert cost == pytest.approx((100 * 0.05 + 20 * 0.25) / 1_000_000)


async def test_strips_markdown_fences(settings_llm: Settings) -> None:
    llm = ReplicateLLM(settings_llm)
    with respx.mock(base_url=_BASE) as mock:
        mock.post("/models/openai/gpt-oss-120b/predictions").mock(
            return_value=httpx.Response(201, json={"id": "p2", "status": "starting"})
        )
        mock.get("/predictions/p2").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "p2",
                    "status": "succeeded",
                    "output": ["```json\n", '{"greeting":"hi","count":1}', "\n```"],
                },
            )
        )

        parsed, _ = await llm.complete_structured(system="sys", user="say hi", schema=_Greeting)

        assert parsed == _Greeting(greeting="hi", count=1)


async def test_repair_round_on_invalid_json_then_succeeds(settings_llm: Settings) -> None:
    llm = ReplicateLLM(settings_llm)
    with respx.mock(base_url=_BASE) as mock:
        mock.post("/models/openai/gpt-oss-120b/predictions").mock(
            side_effect=[
                httpx.Response(201, json={"id": "p3", "status": "starting"}),
                httpx.Response(201, json={"id": "p4", "status": "starting"}),
            ]
        )
        mock.get("/predictions/p3").mock(
            return_value=httpx.Response(
                200, json={"id": "p3", "status": "succeeded", "output": ["not json at all"]}
            )
        )
        mock.get("/predictions/p4").mock(
            return_value=httpx.Response(
                200,
                json={"id": "p4", "status": "succeeded", "output": ['{"greeting":"hi","count":2}']},
            )
        )

        parsed, _ = await llm.complete_structured(system="sys", user="say hi", schema=_Greeting)

        assert parsed == _Greeting(greeting="hi", count=2)


async def test_repair_round_fails_twice_raises(settings_llm: Settings) -> None:
    llm = ReplicateLLM(settings_llm)
    with respx.mock(base_url=_BASE) as mock:
        mock.post("/models/openai/gpt-oss-120b/predictions").mock(
            side_effect=[
                httpx.Response(201, json={"id": "p5", "status": "starting"}),
                httpx.Response(201, json={"id": "p6", "status": "starting"}),
            ]
        )
        mock.get("/predictions/p5").mock(
            return_value=httpx.Response(
                200, json={"id": "p5", "status": "succeeded", "output": ["nope"]}
            )
        )
        mock.get("/predictions/p6").mock(
            return_value=httpx.Response(
                200, json={"id": "p6", "status": "succeeded", "output": ["still nope"]}
            )
        )

        with pytest.raises(LLMStructuredOutputError):
            await llm.complete_structured(system="sys", user="say hi", schema=_Greeting)


async def test_vision_tier_falls_back_to_standard_model(settings_llm: Settings) -> None:
    llm = ReplicateLLM(settings_llm)
    with respx.mock(base_url=_BASE) as mock:
        route = mock.post("/models/openai/gpt-oss-120b/predictions").mock(
            return_value=httpx.Response(201, json={"id": "p7", "status": "starting"})
        )
        mock.get("/predictions/p7").mock(
            return_value=httpx.Response(
                200,
                json={"id": "p7", "status": "succeeded", "output": ['{"greeting":"hi","count":1}']},
            )
        )

        await llm.complete_structured(
            system="sys", user="describe", schema=_Greeting, model_tier="vision", images=[b"PNG"]
        )

        assert route.called
