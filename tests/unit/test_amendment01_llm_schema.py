from __future__ import annotations

import pytest
from storysmith.models import QAReport, SceneManifest
from storysmith.util.llm_schema import strict_tool_schema

pytestmark = pytest.mark.amendment01


def _find_defaults(node: object) -> list[object]:
    """Recursively collect every "default" value still present in a schema."""
    found: list[object] = []
    if isinstance(node, dict):
        found.extend(v for k, v in node.items() if k == "default")
        for v in node.values():
            found.extend(_find_defaults(v))
    elif isinstance(node, list):
        for item in node:
            found.extend(_find_defaults(item))
    return found


def test_regression_gen_mode_ref_has_no_sibling_default() -> None:
    """Amendment 01: Scene.gen_mode (an enum field with a default) rendered
    as {"$ref": ..., "default": ...} in the raw Pydantic schema -- a $ref
    with a sibling keyword that broke Groq's tool-call schema validation
    intermittently, live, with a well-formed generation rejected as
    "tool_use_failed" for no clear reason. strict_tool_schema must strip it."""
    raw = SceneManifest.model_json_schema()
    gen_mode_schema = raw["$defs"]["Scene"]["properties"]["gen_mode"]
    assert "default" in gen_mode_schema  # confirms the bug is still reproducible upstream

    sanitized = strict_tool_schema(raw)
    assert _find_defaults(sanitized) == []
    assert sanitized["$defs"]["Scene"]["properties"]["gen_mode"] == {"$ref": "#/$defs/SceneGenMode"}


def test_regression_failure_layer_ref_has_no_sibling_default() -> None:
    # Same bug, same shape: QAReport.failure_layer is critic's structured
    # output, not director's -- both LLM adapters go through the same fix.
    raw = QAReport.model_json_schema()
    assert "default" in raw["properties"]["failure_layer"]

    sanitized = strict_tool_schema(raw)
    assert _find_defaults(sanitized) == []


def test_strict_tool_schema_preserves_non_default_content() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string", "default": "x"}, "b": {"type": "integer"}},
        "required": ["b"],
    }
    result = strict_tool_schema(schema)
    assert result == {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["b"],
    }
