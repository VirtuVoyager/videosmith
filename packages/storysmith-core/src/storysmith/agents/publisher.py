from __future__ import annotations

from typing import TYPE_CHECKING, Any

from storysmith.models import AssetKind, ProjectStatus, VideoProject
from storysmith.settings import Settings

if TYPE_CHECKING:
    from storysmith.pipeline import PortBundle

# SPEC-GAP: the graph interrupts before this node (see graph/build.py,
# interrupt_before=["publisher"]), so it only runs once WP7's API approve
# endpoint resumes the checkpoint. VideoProject has no field to persist the
# returned YouTube URL — §12 requires stopping to amend the spec before adding
# one, so WP7 needs a contract decision here (new model field vs. NotifyPort
# message body) before this can do more than mark PUBLISHED.


def _latest_asset(state: VideoProject, kind: AssetKind) -> str | None:
    for asset in reversed(state.assets):
        if asset.kind == kind:
            return asset.uri
    return None


async def run(state: VideoProject, *, ports: PortBundle, settings: Settings) -> dict[str, Any]:
    assert state.manifest is not None
    final_uri = _latest_asset(state, AssetKind.FINAL_VIDEO)
    thumb_uri = _latest_asset(state, AssetKind.THUMBNAIL)
    assert final_uri is not None
    assert thumb_uri is not None

    video_bytes = await ports.storage.get(uri=final_uri)
    thumb_bytes = await ports.storage.get(uri=thumb_uri)
    url = await ports.publish.upload(
        video=video_bytes,
        thumbnail=thumb_bytes,
        title=state.manifest.title,
        description=state.manifest.description,
        tags=state.manifest.tags,
    )
    await ports.notify.send(text=f"Published: {state.manifest.title}", link=url)
    return {"status": ProjectStatus.PUBLISHED}
