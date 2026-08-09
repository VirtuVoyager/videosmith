from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from storysmith.models import CostEntry, ProjectStatus, StyleContract, VideoProject
from storysmith.settings import Settings
from storysmith.util import prompts
from storysmith.util.configs import load_safety_negative_terms, load_style_preset_yaml

if TYPE_CHECKING:
    # Local-only: pipeline.py -> graph.build -> graph.nodes -> agents.* would
    # cycle back here if PortBundle were imported at module load time.
    from storysmith.pipeline import PortBundle

_SYSTEM_PROMPT = (
    "You produce structured content contracts for StorySmith, an autonomous "
    "kids' shorts platform. Respond only via the emit tool."
)

# SPEC-GAP: only one preset exists (configs/style_presets/rhyme_soft2d.yaml).
# A topical-mode preset isn't authored yet, so both modes use it for now --
# add a topical preset file and extend this mapping when one exists.
_STYLE_PRESET_BY_MODE = {
    "rhyme": "rhyme_soft2d",
    "topical": "rhyme_soft2d",
}


def _merge_safety_terms(style: StyleContract, base_terms: list[str]) -> StyleContract:
    merged = list(dict.fromkeys([*base_terms, *style.negative_terms]))
    return style.model_copy(update={"negative_terms": merged})


async def run(state: VideoProject, *, ports: PortBundle, settings: Settings) -> dict[str, Any]:
    preset_name = _STYLE_PRESET_BY_MODE[state.mode.value]
    style_preset_yaml = load_style_preset_yaml(settings.configs_dir, preset_name)
    user_prompt = prompts.load(
        "creative_director",
        brief=state.brief,
        mode=state.mode.value,
        style_preset_yaml=style_preset_yaml,
    )

    style_obj, cost = await ports.llm.complete_structured(
        system=_SYSTEM_PROMPT,
        user=user_prompt,
        schema=StyleContract,
        model_tier="standard",
    )
    assert isinstance(style_obj, StyleContract)

    base_terms = load_safety_negative_terms(settings.configs_dir)
    style_obj = _merge_safety_terms(style_obj, base_terms)

    return {
        "style": style_obj,
        "status": ProjectStatus.STYLED,
        "cost_ledger": [
            CostEntry(
                at=datetime.now(UTC), item="creative_director:style", provider="llm", cost_usd=cost
            )
        ],
    }
