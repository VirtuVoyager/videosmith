from __future__ import annotations

import pytest
from storysmith.util import prompts

pytestmark = pytest.mark.wp2


def test_prompt_loader_fills_placeholders() -> None:
    text = prompts.load(
        "creative_director", brief="counting ducks", mode="rhyme", style_preset_yaml="art_style: x"
    )
    assert "counting ducks" in text
    assert "rhyme" in text
    assert "art_style: x" in text


def test_prompt_loader_placeholders_raises_with_missing_name() -> None:
    with pytest.raises(KeyError) as exc_info:
        prompts.load("director", brief="b", mode="rhyme", style_json="{}")

    assert "violation_note" in str(exc_info.value)
