from __future__ import annotations

from pathlib import Path
from typing import Any

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load(name: str, **kwargs: Any) -> str:
    """Load prompts/{name}.md and fill in its {placeholder} values.

    Never inline prompts in Python -- templates live as markdown so they can
    be edited/reviewed without touching agent code (§0.1, §2.2).
    """
    path = _PROMPTS_DIR / f"{name}.md"
    template = path.read_text(encoding="utf-8")
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        placeholder = exc.args[0]
        raise KeyError(
            f"prompts/{name}.md references placeholder {{{placeholder}}} with no value provided"
        ) from exc
