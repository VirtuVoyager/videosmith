from __future__ import annotations

from typing import Any


def strict_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip "default" keys from a Pydantic JSON schema before sending it as
    a tool/function-calling parameters schema.

    Pydantic renders any field that is both an enum type and has a default
    as `{"$ref": "#/$defs/...", "default": ...}` -- a `$ref` combined with a
    sibling keyword. Amendment 01 hit this for real: adding
    `Scene.gen_mode: SceneGenMode = SceneGenMode.I2V` (the first
    enum-with-default field ever sent to an LLM's structured-output schema
    in this codebase) made Groq's tool-call schema validation intermittently
    reject otherwise well-formed SceneManifest generations with a 400
    "tool_use_failed", with no clear signal why -- the JSON produced looked
    complete and schema-conformant by eye. Defaults are meaningless for a
    tool call the model must always fully populate anyway, so stripping
    them everywhere in the schema is safe and sidesteps the issue instead
    of chasing which specific provider tolerates which schema shape.
    """
    if isinstance(schema, dict):
        return {k: strict_tool_schema(v) for k, v in schema.items() if k != "default"}
    if isinstance(schema, list):
        return [strict_tool_schema(v) for v in schema]
    return schema
