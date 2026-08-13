from __future__ import annotations

from storysmith.models import AssetKind, AssetRef


def latest_audio_master(assets: list[AssetRef]) -> AssetRef | None:
    """Latest-attempt AUDIO_MASTER with no role meta (rhyme mode's single track).

    music_director re-generates the whole audio track on every retry (§6), so
    after a retry there can be multiple AUDIO_MASTER assets in state -- always
    take the highest attempt, never just the first match.
    """
    candidates = [a for a in assets if a.kind == AssetKind.AUDIO_MASTER and "role" not in a.meta]
    return max(candidates, key=lambda a: a.attempt, default=None)


def latest_audio_bed(assets: list[AssetRef]) -> AssetRef | None:
    candidates = [
        a for a in assets if a.kind == AssetKind.AUDIO_MASTER and a.meta.get("role") == "bed"
    ]
    return max(candidates, key=lambda a: a.attempt, default=None)


def latest_scene_still(assets: list[AssetRef], scene_index: int) -> AssetRef | None:
    """Latest-attempt SCENE_STILL for one scene (Amendment 01) -- a composition
    retry regenerates the still, so there can be multiple attempts in state."""
    candidates = [
        a for a in assets if a.kind == AssetKind.SCENE_STILL and a.scene_index == scene_index
    ]
    return max(candidates, key=lambda a: a.attempt, default=None)


def latest_narration_assets(assets: list[AssetRef]) -> list[AssetRef]:
    """Latest-attempt narration AUDIO_MASTER per scene index, sorted by index."""
    by_scene: dict[int, AssetRef] = {}
    for asset in assets:
        if (
            asset.kind != AssetKind.AUDIO_MASTER
            or asset.meta.get("role") != "narration"
            or asset.scene_index is None
        ):
            continue
        current = by_scene.get(asset.scene_index)
        if current is None or asset.attempt > current.attempt:
            by_scene[asset.scene_index] = asset
    return [by_scene[idx] for idx in sorted(by_scene)]
