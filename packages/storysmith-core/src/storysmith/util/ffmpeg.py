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
