from __future__ import annotations

import pytest
from storysmith.agents.editor import CaptionStyle, build_ass_subtitles

pytestmark = pytest.mark.wp5

_WORDS: list[dict[str, str | float]] = [
    {"word": "one", "start": 0.0, "end": 0.3},
    {"word": "two", "start": 0.3, "end": 0.6},
    {"word": "three", "start": 0.6, "end": 1.0},
    {"word": "four", "start": 1.0, "end": 1.3},
    {"word": "five", "start": 1.3, "end": 1.6},
]

_EXPECTED = (
    "[Script Info]\n"
    "ScriptType: v4.00+\n"
    "PlayResX: 1080\n"
    "PlayResY: 1920\n"
    "\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, Bold, "
    "Alignment, MarginL, MarginR, MarginV\n"
    "Style: Default,Arial,64,&H00FFFFFF,&H00000000,-1,5,10,10,10\n"
    "\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Text\n"
    "Dialogue: 0,0:00:00.00,0:00:01.00,Default,one two three\n"
    "Dialogue: 0,0:00:01.00,0:00:01.60,Default,four five\n"
)


def test_build_ass_subtitles_snapshot() -> None:
    assert build_ass_subtitles(_WORDS) == _EXPECTED


def test_build_ass_subtitles_empty_words() -> None:
    result = build_ass_subtitles([])
    assert "[Events]" in result
    assert "Dialogue:" not in result


def test_build_ass_subtitles_respects_custom_max_words() -> None:
    style = CaptionStyle(max_words_per_caption=2)
    result = build_ass_subtitles(_WORDS, style=style)
    dialogue_lines = [line for line in result.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogue_lines) == 3  # groups of 2,2,1
    assert dialogue_lines[0].endswith("one two")
    assert dialogue_lines[1].endswith("three four")
    assert dialogue_lines[2].endswith("five")


def test_build_ass_subtitles_custom_style_fields() -> None:
    style = CaptionStyle(font_name="Comic Sans MS", font_size=48, bold=False, alignment=2)
    result = build_ass_subtitles(_WORDS, style=style)
    assert "Style: Default,Comic Sans MS,48,&H00FFFFFF,&H00000000,0,2,10,10,10" in result
