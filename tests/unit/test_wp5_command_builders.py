from __future__ import annotations

from pathlib import Path

import pytest
from storysmith.agents.editor import (
    _build_audio_topical_cmd,
    _build_audio_trim_pad_cmd,
    _build_caption_cmd,
    _build_concat_cmd,
    _build_concat_filter,
    _build_loudnorm_cmd,
    _build_mux_cmd,
    _build_normalize_cmd,
    _scene_midpoint_timestamp,
)
from storysmith.util.ffmpeg import build_frame_extract_cmd

pytestmark = pytest.mark.wp5


def test_build_normalize_cmd() -> None:
    cmd = _build_normalize_cmd(Path("in.mp4"), Path("out.mp4"))
    assert cmd == [
        "ffmpeg",
        "-y",
        "-i",
        "in.mp4",
        "-vf",
        "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=24",
        "-pix_fmt",
        "yuv420p",
        "out.mp4",
    ]


def test_concat_filter_single_clip() -> None:
    filter_complex, out_label = _build_concat_filter([1.0], ["crossfade"])
    assert filter_complex == ""
    assert out_label == "0:v"


def test_concat_filter_all_cuts_uses_concat_filter() -> None:
    filter_complex, out_label = _build_concat_filter([1.0, 1.0, 1.0], ["crossfade", "cut", "cut"])
    assert filter_complex == "[0:v][1:v][2:v]concat=n=3:v=1:a=0[vout]"
    assert out_label == "vout"


def test_concat_filter_all_crossfades_chains_xfade() -> None:
    filter_complex, out_label = _build_concat_filter(
        [2.0, 2.0, 2.0], ["crossfade", "crossfade", "crossfade"]
    )
    assert filter_complex == (
        "[0:v][1:v]xfade=transition=fade:duration=0.4:offset=1.600[x1];"
        "[x1][2:v]xfade=transition=fade:duration=0.4:offset=3.200[x2]"
    )
    assert out_label == "x2"


def test_concat_filter_mixed_cut_and_crossfade() -> None:
    filter_complex, out_label = _build_concat_filter(
        [2.0, 2.0, 2.0], ["crossfade", "cut", "crossfade"]
    )
    assert filter_complex == (
        "[0:v][1:v]concat=n=2:v=1:a=0[x1];"
        "[x1][2:v]xfade=transition=fade:duration=0.4:offset=3.600[x2]"
    )
    assert out_label == "x2"


def test_build_concat_cmd_wraps_filter_in_brackets_only_when_present() -> None:
    single = _build_concat_cmd([Path("a.mp4")], [1.0], ["crossfade"], Path("out.mp4"))
    assert "-filter_complex" not in single
    single_map_index = single.index("-map")
    assert single[single_map_index + 1] == "0:v"

    multi = _build_concat_cmd(
        [Path("a.mp4"), Path("b.mp4")], [1.0, 1.0], ["crossfade", "crossfade"], Path("out.mp4")
    )
    assert "-filter_complex" in multi
    map_index = multi.index("-map")
    assert multi[map_index + 1] == "[x1]"


def test_build_audio_trim_pad_cmd() -> None:
    cmd = _build_audio_trim_pad_cmd(Path("a.mp3"), 12.345, Path("out.wav"))
    assert cmd == ["ffmpeg", "-y", "-i", "a.mp3", "-af", "apad", "-t", "12.345", "out.wav"]


def test_build_audio_topical_cmd_single_narration_uses_anull() -> None:
    cmd = _build_audio_topical_cmd(Path("bed.mp3"), [Path("n0.mp3")], [2.5], 10.0, Path("out.wav"))
    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    assert "[1:a]adelay=2500|2500[n1]" in filter_complex
    assert "[n1]anull[nmix]" in filter_complex
    assert "[0:a]apad[bed_padded]" in filter_complex
    assert "sidechaincompress" in filter_complex
    t_index = cmd.index("-t")
    assert cmd[t_index + 1] == "10.000"


def test_build_audio_topical_cmd_multiple_narration_uses_amix() -> None:
    cmd = _build_audio_topical_cmd(
        Path("bed.mp3"),
        [Path("n0.mp3"), Path("n1.mp3")],
        [0.0, 5.0],
        10.0,
        Path("out.wav"),
    )
    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    assert "[1:a]adelay=0|0[n1]" in filter_complex
    assert "[2:a]adelay=5000|5000[n2]" in filter_complex
    assert "[n1][n2]amix=inputs=2:normalize=0[nmix]" in filter_complex


def test_build_audio_topical_cmd_requires_narration() -> None:
    with pytest.raises(AssertionError):
        _build_audio_topical_cmd(Path("bed.mp3"), [], [], 10.0, Path("out.wav"))


def test_build_loudnorm_cmd() -> None:
    cmd = _build_loudnorm_cmd(Path("in.wav"), Path("out.wav"))
    assert cmd == [
        "ffmpeg",
        "-y",
        "-i",
        "in.wav",
        "-af",
        "loudnorm=I=-14:TP=-1.5:LRA=11",
        "out.wav",
    ]


def test_build_mux_cmd() -> None:
    cmd = _build_mux_cmd(Path("v.mp4"), Path("a.wav"), Path("out.mp4"))
    assert cmd == [
        "ffmpeg",
        "-y",
        "-i",
        "v.mp4",
        "-i",
        "a.wav",
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-shortest",
        "out.mp4",
    ]


def test_build_caption_cmd() -> None:
    cmd = _build_caption_cmd(Path("in.mp4"), Path("cap.ass"), Path("out.mp4"))
    assert cmd == ["ffmpeg", "-y", "-i", "in.mp4", "-vf", "ass=cap.ass", "-c:a", "copy", "out.mp4"]


def test_build_frame_extract_cmd() -> None:
    cmd = build_frame_extract_cmd(Path("v.mp4"), 1.5, Path("frame.jpg"))
    assert cmd == [
        "ffmpeg",
        "-y",
        "-ss",
        "1.500",
        "-i",
        "v.mp4",
        "-frames:v",
        "1",
        "-pix_fmt",
        "yuvj420p",
        "frame.jpg",
    ]


def test_scene_midpoint_timestamp_no_crossfade() -> None:
    ts = _scene_midpoint_timestamp([2.0, 2.0, 2.0], ["crossfade", "cut", "cut"])
    assert ts == pytest.approx(3.0)  # middle clip starts at 2.0, center at 3.0


def test_scene_midpoint_timestamp_with_crossfade_overlap() -> None:
    ts = _scene_midpoint_timestamp([2.0, 2.0, 2.0], ["crossfade", "crossfade", "crossfade"])
    # clip 1 starts at (2.0 - 0.4) = 1.6, center at 1.6 + 1.0 = 2.6
    assert ts == pytest.approx(2.6)
