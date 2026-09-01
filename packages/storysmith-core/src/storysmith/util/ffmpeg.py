from __future__ import annotations

import subprocess
from pathlib import Path


def run_ffmpeg(args: list[str]) -> None:
    """Shared ffmpeg subprocess runner -- editor.py and critic.py both shell
    out to ffmpeg and need the same "surface stderr on failure" behavior
    (bare CalledProcessError hides the diagnostic that actually matters,
    as WP5's ffmpeg-6.1-vs-8.1 filtergraph incompatibility demonstrated)."""
    result = subprocess.run(args, capture_output=True)  # noqa: S603
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-4000:]
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {' '.join(args)}\n{stderr}")


def probe_duration(path: Path) -> float:
    result = subprocess.run(  # noqa: S603
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def build_audio_concat_cmd(
    clip_paths: list[Path], clip_durations_s: list[float], gap_s: float, output_path: Path
) -> list[str]:
    """Amendment 02: concatenate clips (one per dialogue turn) into a single
    audio file with `gap_s` of silence between each, via sequential adelay
    offsets + amix -- same filtergraph shape as editor.py's
    _build_audio_topical_cmd (adelay each input to its place, amix with
    normalize=0 so non-overlapping clips don't lose volume), reused here for
    stitching dialogue turns into one clip rather than ducking narration
    against a music bed. music_director.py stores the single output as the
    scene's one narration AssetRef, so nothing downstream ever needs to know
    a scene's narration came from more than one voice."""
    assert clip_paths, "audio concat requires at least one clip"
    assert len(clip_paths) == len(clip_durations_s), "one duration per clip"
    cmd = ["ffmpeg", "-y"]
    for path in clip_paths:
        cmd += ["-i", str(path)]

    filters: list[str] = []
    delayed_labels: list[str] = []
    offset_s = 0.0
    for idx, duration_s in enumerate(clip_durations_s):
        ms = max(round(offset_s * 1000), 0)
        label = f"c{idx}"
        filters.append(f"[{idx}:a]adelay={ms}|{ms}[{label}]")
        delayed_labels.append(f"[{label}]")
        offset_s += duration_s + gap_s

    if len(delayed_labels) == 1:
        filters.append(f"{delayed_labels[0]}anull[out]")
    else:
        joined = "".join(delayed_labels)
        filters.append(f"{joined}amix=inputs={len(delayed_labels)}:normalize=0[out]")

    cmd += ["-filter_complex", ";".join(filters), "-map", "[out]", str(output_path)]
    return cmd


def build_frame_extract_cmd(video_path: Path, timestamp_s: float, output_path: Path) -> list[str]:
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
