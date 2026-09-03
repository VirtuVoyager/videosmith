from __future__ import annotations

import json

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
        replicate_vision_caption_model="lucataco/qwen2-vl-7b-instruct",
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


async def test_vision_tier_with_no_images_skips_captioning(settings_llm: Settings) -> None:
    llm = ReplicateLLM(settings_llm)
    with respx.mock(base_url=_BASE) as mock:
        # No route registered for the caption model at all -- if
        # complete_structured tried to call it anyway, respx would raise on
        # the unmatched request, failing this test loudly.
        mock.post("/models/openai/gpt-oss-120b/predictions").mock(
            return_value=httpx.Response(201, json={"id": "p7", "status": "starting"})
        )
        mock.get("/predictions/p7").mock(
            return_value=httpx.Response(
                200,
                json={"id": "p7", "status": "succeeded", "output": ['{"greeting":"hi","count":1}']},
            )
        )

        parsed, _ = await llm.complete_structured(
            system="sys", user="describe", schema=_Greeting, model_tier="vision", images=None
        )

        assert parsed == _Greeting(greeting="hi", count=1)


async def test_vision_tier_captions_each_image_then_extracts_json(settings_llm: Settings) -> None:
    """The real two-stage pipeline: each image is captioned independently
    first (no Replicate raw-completion model takes several images in one
    request), then those descriptions are folded into `user` as plain text
    before the same text-only JSON-extraction call every other request uses."""
    llm = ReplicateLLM(settings_llm)
    with respx.mock(base_url=_BASE) as mock:
        mock.post("/models/lucataco/qwen2-vl-7b-instruct/predictions").mock(
            side_effect=[
                httpx.Response(201, json={"id": "cap1", "status": "starting"}),
                httpx.Response(201, json={"id": "cap2", "status": "starting"}),
            ]
        )
        mock.get("/predictions/cap1").mock(
            return_value=httpx.Response(
                200, json={"id": "cap1", "status": "succeeded", "output": ["a red cube"]}
            )
        )
        mock.get("/predictions/cap2").mock(
            return_value=httpx.Response(
                200, json={"id": "cap2", "status": "succeeded", "output": ["a blue sphere"]}
            )
        )
        extract_route = mock.post("/models/openai/gpt-oss-120b/predictions").mock(
            return_value=httpx.Response(201, json={"id": "p8", "status": "starting"})
        )
        mock.get("/predictions/p8").mock(
            return_value=httpx.Response(
                200,
                json={"id": "p8", "status": "succeeded", "output": ['{"greeting":"hi","count":1}']},
            )
        )

        parsed, _ = await llm.complete_structured(
            system="sys",
            user="describe the scene",
            schema=_Greeting,
            model_tier="vision",
            images=[b"IMG1", b"IMG2"],
        )

        assert parsed == _Greeting(greeting="hi", count=1)
        sent_prompt = json.loads(extract_route.calls.last.request.content)["input"]["prompt"]
        assert "Image 1 description: a red cube" in sent_prompt
        assert "Image 2 description: a blue sphere" in sent_prompt
        assert "describe the scene" in sent_prompt  # original user text preserved


async def test_vision_caption_failure_falls_back_to_placeholder_not_crash(
    settings_llm: Settings,
) -> None:
    """One bad image (rejected, timed out, whatever) shouldn't sink the
    whole QA call -- captioning failures degrade to a placeholder string
    instead of propagating."""
    llm = ReplicateLLM(settings_llm)
    with respx.mock(base_url=_BASE) as mock:
        mock.post("/models/lucataco/qwen2-vl-7b-instruct/predictions").mock(
            return_value=httpx.Response(500, json={"detail": "internal error"})
        )
        mock.post("/models/openai/gpt-oss-120b/predictions").mock(
            return_value=httpx.Response(201, json={"id": "p9", "status": "starting"})
        )
        mock.get("/predictions/p9").mock(
            return_value=httpx.Response(
                200,
                json={"id": "p9", "status": "succeeded", "output": ['{"greeting":"hi","count":1}']},
            )
        )

        parsed, cost = await llm.complete_structured(
            system="sys", user="describe", schema=_Greeting, model_tier="vision", images=[b"IMG"]
        )

        assert parsed == _Greeting(greeting="hi", count=1)  # didn't crash
        assert cost == pytest.approx(0.0)  # failed caption contributes no cost
