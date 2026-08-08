from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from storysmith.models import CostEntry, ProjectStatus, SceneManifest, VideoProject
from storysmith.settings import Settings

if TYPE_CHECKING:
    from storysmith.pipeline import PortBundle

# SPEC-GAP: real prompt is prompts/director.md (WP2) plus the code-level
# post-validation (duration/scene-count bounds, corrective LLM round) from
# §2.2. This placeholder gets the WP1 graph running end to end against stubs;
# WP2 replaces the prompt and adds the validation loop.
_SYSTEM_PROMPT = (
    "You are the Director. Given a StyleContract and brief, produce a SceneManifest "
    "with 4-7 scenes totaling 30-60 seconds."
)


async def run(state: VideoProject, *, ports: PortBundle, settings: Settings) -> dict[str, Any]:
    assert state.style is not None
    manifest_obj, cost = await ports.llm.complete_structured(
        system=_SYSTEM_PROMPT,
        user=f"style={state.style.model_dump_json()}\nbrief={state.brief}",
        schema=SceneManifest,
        model_tier="standard",
    )
    assert isinstance(manifest_obj, SceneManifest)
    return {
        "manifest": manifest_obj,
        "status": ProjectStatus.SCRIPTED,
        "cost_ledger": [
            CostEntry(at=datetime.now(UTC), item="director:manifest", provider="llm", cost_usd=cost)
        ],
    }
