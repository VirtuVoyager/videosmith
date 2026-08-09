from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw, ImageFont

from storysmith.models import (
    AssetKind,
    AssetRef,
    Mode,
    ProjectStatus,
    QAVerdict,
    VideoProject,
)
from storysmith.settings import Settings
from storysmith.util.hashing import sha256_bytes

if TYPE_CHECKING:
    from storysmith.pipeline import PortBundle

_CROSSFADE_DURATION_S = 0.4
_TARGET_WIDTH = 1080
_TARGET_HEIGHT = 1920
_TARGET_FPS = 24


@dataclass(frozen=True)
class CaptionStyle:
    font_name: str = "Arial"
    font_size: int = 64
    bold: bool = True
    alignment: int = 5  # ASS alignment code: 5 = middle-center
    primary_color: str = "&H00FFFFFF"  # white
    outline_color: str = "&H00000000"  # black
    max_words_per_caption: int = 3


_DEFAULT_CAPTION_STYLE = CaptionStyle()


# ---------------------------------------------------------------------------
# Pure command builders -- no ffmpeg execution, unit-testable without it (§5).
# ---------------------------------------------------------------------------


def _build_normalize_cmd(input_path: Path, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"scale={_TARGET_WIDTH}:{_TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={_TARGET_WIDTH}:{_TARGET_HEIGHT},fps={_TARGET_FPS}",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]


def _build_concat_filter(clip_durations: list[float], transitions: list[str]) -> tuple[str, str]:
    """transitions[i] is the transition INTO clip i; transitions[0] is ignored
    (no previous clip). Returns (filter_complex, output_label) -- output_label
    is a bare stream specifier ("0:v") when no filter_complex is needed."""
    n = len(clip_durations)
    if n == 1:
        return "", "0:v"
    if all(t != "crossfade" for t in transitions[1:]):
        inputs = "".join(f"[{i}:v]" for i in range(n))
        return f"{inputs}concat=n={n}:v=1:a=0[vout]", "vout"

    filters: list[str] = []
    current_label = "0:v"
    cumulative = clip_durations[0]
    for i in range(1, n):
        transition = transitions[i]
        out_label = f"x{i}"
        if transition == "crossfade":
            offset = max(cumulative - _CROSSFADE_DURATION_S, 0.0)
            filters.append(
                f"[{current_label}][{i}:v]xfade=transition=fade:"
                f"duration={_CROSSFADE_DURATION_S}:offset={offset:.3f}[{out_label}]"
            )
            cumulative = cumulative + clip_durations[i] - _CROSSFADE_DURATION_S
        else:
            filters.append(f"[{current_label}][{i}:v]concat=n=2:v=1:a=0[{out_label}]")
            cumulative = cumulative + clip_durations[i]
        current_label = out_label
    return ";".join(filters), current_label


def _build_concat_cmd(
    clip_paths: list[Path],
    clip_durations: list[float],
    transitions: list[str],
    output_path: Path,
) -> list[str]:
    filter_complex, out_label = _build_concat_filter(clip_durations, transitions)
    cmd = ["ffmpeg", "-y"]
    for path in clip_paths:
        cmd += ["-i", str(path)]
    if filter_complex:
        cmd += ["-filter_complex", filter_complex, "-map", f"[{out_label}]"]
    else:
        cmd += ["-map", out_label]
    cmd += ["-pix_fmt", "yuv420p", str(output_path)]
    return cmd


def _build_audio_trim_pad_cmd(
    audio_path: Path, video_duration_s: float, output_path: Path
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-af",
        "apad",
        "-t",
        f"{video_duration_s:.3f}",
        str(output_path),
    ]


def _build_audio_topical_cmd(
    bed_path: Path,
    narration_paths: list[Path],
    narration_offsets_s: list[float],
    video_duration_s: float,
    output_path: Path,
) -> list[str]:
    assert narration_paths, "topical audio mix requires at least one narration segment"
    cmd = ["ffmpeg", "-y", "-i", str(bed_path)]
    for path in narration_paths:
        cmd += ["-i", str(path)]

    filters: list[str] = []
    delayed_labels: list[str] = []
    for idx, offset_s in enumerate(narration_offsets_s, start=1):
        ms = max(round(offset_s * 1000), 0)
        label = f"n{idx}"
        filters.append(f"[{idx}:a]adelay={ms}|{ms}[{label}]")
        delayed_labels.append(f"[{label}]")

    narration_mix_label = "nmix"
    if len(delayed_labels) == 1:
        filters.append(f"{delayed_labels[0]}anull[{narration_mix_label}]")
    else:
        joined = "".join(delayed_labels)
        filters.append(
            f"{joined}amix=inputs={len(delayed_labels)}:normalize=0[{narration_mix_label}]"
        )

    # sidechaincompress's output tracks its main (first) input's duration, so
    # pad the bed first -- otherwise a bed shorter than the narration mix
    # (e.g. every stub audio track is a fixed 1s) truncates the whole ducked
    # mix down to the bed's length regardless of the final -t cap below.
    filters.append("[0:a]apad[bed_padded]")
    filters.append(
        f"[bed_padded][{narration_mix_label}]sidechaincompress=threshold=0.05:ratio=8[bed_ducked]"
    )
    filters.append(f"[bed_ducked][{narration_mix_label}]amix=inputs=2:normalize=0[mixed]")

    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[mixed]",
        "-t",
        f"{video_duration_s:.3f}",
        str(output_path),
    ]
    return cmd


def _build_loudnorm_cmd(input_path: Path, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        "loudnorm=I=-14:TP=-1.5:LRA=11",
        str(output_path),
    ]


def _build_mux_cmd(video_path: Path, audio_path: Path, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-shortest",
        str(output_path),
    ]


def _build_caption_cmd(input_path: Path, ass_path: Path, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"ass={ass_path}",
        "-c:a",
        "copy",
        str(output_path),
    ]


def _build_thumbnail_extract_cmd(
    video_path: Path, timestamp_s: float, output_path: Path
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp_s:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        # yuvj420p (not yuv420p): the mjpeg encoder rejects libx264's
        # full-range yuv420p output under strict standard compliance
        # ("Non full-range YUV is non-standard") without this.
        "-pix_fmt",
        "yuvj420p",
        str(output_path),
    ]


# ---------------------------------------------------------------------------
# ASS caption generation -- pure, unit-testable.
# ---------------------------------------------------------------------------


def _format_ass_timestamp(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def _finalize_chunk(chunk: list[dict[str, str | float]]) -> tuple[float, float, str]:
    start = float(chunk[0]["start"])
    end = float(chunk[-1]["end"])
    text = " ".join(str(word["word"]) for word in chunk)
    return start, end, text


def _group_words(
    words: list[dict[str, str | float]], max_words: int
) -> list[tuple[float, float, str]]:
    groups: list[tuple[float, float, str]] = []
    chunk: list[dict[str, str | float]] = []
    for word in words:
        chunk.append(word)
        if len(chunk) == max_words:
            groups.append(_finalize_chunk(chunk))
            chunk = []
    if chunk:
        groups.append(_finalize_chunk(chunk))
    return groups


def build_ass_subtitles(
    words: list[dict[str, str | float]], style: CaptionStyle = _DEFAULT_CAPTION_STYLE
) -> str:
    groups = _group_words(words, style.max_words_per_caption)
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {_TARGET_WIDTH}\n"
        f"PlayResY: {_TARGET_HEIGHT}\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, Bold, "
        "Alignment, MarginL, MarginR, MarginV\n"
        f"Style: Default,{style.font_name},{style.font_size},{style.primary_color},"
        f"{style.outline_color},{-1 if style.bold else 0},{style.alignment},10,10,10\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Text\n"
    )
    lines = [
        f"Dialogue: 0,{_format_ass_timestamp(start)},{_format_ass_timestamp(end)},Default,{text}"
        for start, end, text in groups
    ]
    return header + "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# Thumbnail overlay (Pillow) -- pure given a frame file, unit-testable.
# ---------------------------------------------------------------------------


def _render_thumbnail(frame_path: Path, title: str, output_path: Path) -> None:
    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        font = ImageFont.truetype("Arial.ttf", 72)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (image.width - text_w) / 2
    y = image.height - text_h - 120
    draw.text((x, y), title, font=font, fill="white", stroke_width=4, stroke_fill="black")
    image.save(output_path, "JPEG", quality=90)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True)  # noqa: S603
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-4000:]
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {' '.join(args)}\n{stderr}")


def _probe_duration(path: Path) -> float:
    result = subprocess.run(  # noqa: S603
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _latest_passing_scene_videos(state: VideoProject) -> list[tuple[int, AssetRef]]:
    passing_indices = {
        r.scene_index
        for r in state.qa_reports
        if r.scene_index is not None and r.verdict == QAVerdict.PASS
    }
    latest: dict[int, AssetRef] = {}
    for asset in state.assets:
        if asset.kind != AssetKind.SCENE_VIDEO or asset.scene_index is None:
            continue
        if asset.scene_index not in passing_indices:
            continue
        current = latest.get(asset.scene_index)
        if current is None or asset.attempt > current.attempt:
            latest[asset.scene_index] = asset
    return sorted(latest.items())


def _bed_and_narration(state: VideoProject) -> tuple[AssetRef | None, list[AssetRef]]:
    audio_assets = [a for a in state.assets if a.kind == AssetKind.AUDIO_MASTER]
    bed = next((a for a in audio_assets if a.meta.get("role") == "bed"), None)
    narration = sorted(
        (a for a in audio_assets if a.meta.get("role") == "narration"),
        key=lambda a: a.scene_index if a.scene_index is not None else 0,
    )
    return bed, narration


async def _load_timing_map(ports: PortBundle, bed_asset: AssetRef) -> dict[int, float]:
    timing_map_uri = bed_asset.meta.get("timing_map_uri")
    if not timing_map_uri:
        return {}
    raw = await ports.storage.get(uri=timing_map_uri)
    entries = json.loads(raw)
    return {int(entry["scene_index"]): float(entry["start_s"]) for entry in entries}


def _scene_midpoint_timestamp(durations: list[float], transitions: list[str]) -> float:
    """Timestamp into the concatenated video at the midpoint clip's center.

    Uses actual measured clip durations (ffprobe'd post-normalization), not
    the manifest's requested Scene.duration_s -- generated clips don't
    necessarily come back at exactly the requested length, and stub clips in
    particular are always ~1s regardless of what was requested. Mirrors
    _build_concat_filter's cumulative-offset math so the timestamp lands
    inside the real concatenated output even after crossfade overlap shrinks
    total duration.
    """
    mid_idx = len(durations) // 2
    elapsed = 0.0
    for i, duration in enumerate(durations):
        is_crossfade = i > 0 and transitions[i] == "crossfade"
        start = elapsed - _CROSSFADE_DURATION_S if is_crossfade else elapsed
        if i == mid_idx:
            return start + duration / 2
        elapsed = start + duration
    return elapsed / 2


async def run(state: VideoProject, *, ports: PortBundle, settings: Settings) -> dict[str, Any]:
    assert state.manifest is not None
    manifest = state.manifest
    scenes_by_index = {scene.index: scene for scene in manifest.scenes}
    scene_videos = _latest_passing_scene_videos(state)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        normalized_paths: list[Path] = []
        durations: list[float] = []
        transitions: list[str] = []
        for idx, asset in scene_videos:
            raw_bytes = await ports.storage.get(uri=asset.uri)
            raw_path = tmp_path / f"raw_{idx}.mp4"
            raw_path.write_bytes(raw_bytes)
            norm_path = tmp_path / f"norm_{idx}.mp4"
            await asyncio.to_thread(_run_ffmpeg, _build_normalize_cmd(raw_path, norm_path))
            normalized_paths.append(norm_path)
            durations.append(await asyncio.to_thread(_probe_duration, norm_path))
            transitions.append(scenes_by_index[idx].transition)

        concat_path = tmp_path / "concat.mp4"
        await asyncio.to_thread(
            _run_ffmpeg, _build_concat_cmd(normalized_paths, durations, transitions, concat_path)
        )
        video_duration = await asyncio.to_thread(_probe_duration, concat_path)

        if state.mode == Mode.RHYME:
            master = next(a for a in state.assets if a.kind == AssetKind.AUDIO_MASTER)
            master_bytes = await ports.storage.get(uri=master.uri)
            master_path = tmp_path / "master.mp3"
            master_path.write_bytes(master_bytes)
            mixed_audio_path = tmp_path / "audio_trimmed.wav"
            await asyncio.to_thread(
                _run_ffmpeg,
                _build_audio_trim_pad_cmd(master_path, video_duration, mixed_audio_path),
            )
        else:
            bed, narration = _bed_and_narration(state)
            assert bed is not None, "topical mode requires a bed AUDIO_MASTER asset"
            bed_bytes = await ports.storage.get(uri=bed.uri)
            bed_path = tmp_path / "bed.mp3"
            bed_path.write_bytes(bed_bytes)
            if not narration:
                mixed_audio_path = tmp_path / "audio_trimmed.wav"
                await asyncio.to_thread(
                    _run_ffmpeg,
                    _build_audio_trim_pad_cmd(bed_path, video_duration, mixed_audio_path),
                )
            else:
                timing_map = await _load_timing_map(ports, bed)
                narration_paths = []
                offsets = []
                for asset in narration:
                    n_bytes = await ports.storage.get(uri=asset.uri)
                    n_path = tmp_path / f"narration_{asset.scene_index}.mp3"
                    n_path.write_bytes(n_bytes)
                    narration_paths.append(n_path)
                    offsets.append(timing_map.get(asset.scene_index or 0, 0.0))
                mixed_audio_path = tmp_path / "audio_mixed.wav"
                await asyncio.to_thread(
                    _run_ffmpeg,
                    _build_audio_topical_cmd(
                        bed_path, narration_paths, offsets, video_duration, mixed_audio_path
                    ),
                )

        loudnorm_path = tmp_path / "audio_final.wav"
        await asyncio.to_thread(_run_ffmpeg, _build_loudnorm_cmd(mixed_audio_path, loudnorm_path))

        muxed_path = tmp_path / "muxed.mp4"
        await asyncio.to_thread(_run_ffmpeg, _build_mux_cmd(concat_path, loudnorm_path, muxed_path))

        final_audio_bytes = loudnorm_path.read_bytes()
        words, _cost = await ports.transcribe.transcribe(audio=final_audio_bytes)
        ass_path = tmp_path / "captions.ass"
        ass_path.write_text(build_ass_subtitles(words), encoding="utf-8")
        captioned_path = tmp_path / "final.mp4"
        await asyncio.to_thread(
            _run_ffmpeg, _build_caption_cmd(muxed_path, ass_path, captioned_path)
        )
        final_bytes = captioned_path.read_bytes()

        midpoint_ts = _scene_midpoint_timestamp(durations, transitions)
        frame_path = tmp_path / "frame.jpg"
        await asyncio.to_thread(
            _run_ffmpeg, _build_thumbnail_extract_cmd(captioned_path, midpoint_ts, frame_path)
        )
        thumb_path = tmp_path / "thumbnail.jpg"
        await asyncio.to_thread(_render_thumbnail, frame_path, manifest.title, thumb_path)
        thumb_bytes = thumb_path.read_bytes()

    final_uri = await ports.storage.put(
        key=f"{state.project_id}/final.mp4", data=final_bytes, content_type="video/mp4"
    )
    thumb_uri = await ports.storage.put(
        key=f"{state.project_id}/thumbnail.jpg", data=thumb_bytes, content_type="image/jpeg"
    )

    assets = [
        AssetRef(kind=AssetKind.FINAL_VIDEO, uri=final_uri, content_hash=sha256_bytes(final_bytes)),
        AssetRef(kind=AssetKind.THUMBNAIL, uri=thumb_uri, content_hash=sha256_bytes(thumb_bytes)),
    ]
    return {"assets": assets, "status": ProjectStatus.EDITING}
