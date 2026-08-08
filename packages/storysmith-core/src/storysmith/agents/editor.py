from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from storysmith.models import AssetKind, AssetRef, ProjectStatus, VideoProject
from storysmith.settings import Settings
from storysmith.util.hashing import sha256_bytes

if TYPE_CHECKING:
    from storysmith.pipeline import PortBundle

# SPEC-GAP: this is the minimal WP1 "produce a fake final video file" pass —
# plain concat, no crossfades, no audio mux/duck/loudnorm, no captions, no
# Pillow thumbnail overlay. WP5 replaces this function body wholesale with
# the full ffmpeg pipeline from §5 (normalize -> xfade -> audio mix -> loudnorm
# -> captions -> thumbnail overlay).


def _latest_scene_videos(state: VideoProject) -> list[tuple[int, AssetRef]]:
    latest: dict[int, AssetRef] = {}
    for asset in state.assets:
        if asset.kind != AssetKind.SCENE_VIDEO or asset.scene_index is None:
            continue
        current = latest.get(asset.scene_index)
        if current is None or asset.attempt > current.attempt:
            latest[asset.scene_index] = asset
    return sorted(latest.items())


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run(args, check=True, capture_output=True)  # noqa: S603


async def run(state: VideoProject, *, ports: PortBundle, settings: Settings) -> dict[str, Any]:
    scene_videos = _latest_scene_videos(state)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        clip_paths: list[Path] = []
        for idx, asset in scene_videos:
            data = await ports.storage.get(uri=asset.uri)
            clip_path = tmp_path / f"scene_{idx}.mp4"
            clip_path.write_bytes(data)
            clip_paths.append(clip_path)

        concat_list = tmp_path / "concat.txt"
        concat_list.write_text("".join(f"file '{p}'\n" for p in clip_paths))

        final_path = tmp_path / "final.mp4"
        await asyncio.to_thread(
            _run_ffmpeg,
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy",
                str(final_path),
            ],
        )
        final_bytes = final_path.read_bytes()

        thumb_path = tmp_path / "thumb.jpg"
        await asyncio.to_thread(
            _run_ffmpeg,
            ["ffmpeg", "-y", "-i", str(final_path), "-frames:v", "1", str(thumb_path)],
        )
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
