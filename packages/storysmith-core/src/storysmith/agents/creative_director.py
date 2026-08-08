from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from storysmith.models import CostEntry, ProjectStatus, StyleContract, VideoProject
from storysmith.settings import Settings

if TYPE_CHECKING:
    # Local-only: pipeline.py -> graph.build -> graph.nodes -> agents.* would
    # cycle back here if PortBundle were imported at module load time.
    from storysmith.pipeline import PortBundle

# SPEC-GAP: real prompt lives in prompts/creative_director.md, loaded via the
# prompts.load() helper and merged with configs/style_presets + safety_rules.yaml
# negative_terms per §2.2 — that loader and the preset/safety configs are WP2
# scope. This inline prompt is a placeholder so WP1's graph is runnable end to
# end against stub ports.
_SYSTEM_PROMPT = (
    "You are the Creative Director for a kids' shorts video. "
    "Given a brief, produce a StyleContract with 1-2 characters."
)


async def run(state: VideoProject, *, ports: PortBundle, settings: Settings) -> dict[str, Any]:
    style_obj, cost = await ports.llm.complete_structured(
        system=_SYSTEM_PROMPT,
        user=f"mode={state.mode.value}\nbrief={state.brief}",
        schema=StyleContract,
        model_tier="standard",
    )
    assert isinstance(style_obj, StyleContract)
    return {
        "style": style_obj,
        "status": ProjectStatus.STYLED,
        "cost_ledger": [
            CostEntry(
                at=datetime.now(UTC), item="creative_director:style", provider="llm", cost_usd=cost
            )
        ],
    }
