"""JSON Schema tightening for structured output and tool definitions.

Providers enforce a schema only in *strict* mode, and strict mode has
requirements a stock Pydantic schema does not meet: every object must forbid
additional properties, and every property must be listed as required. A schema
that misses either is accepted by the API and then quietly not enforced — the
worst outcome, because the pipeline looks validated and is not.

This walks the whole schema, `$defs` included, so nested models are tightened
too rather than only the top level.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def _tighten(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _tighten(item)
        return
    if not isinstance(node, dict):
        return

    properties = node.get("properties")
    if isinstance(properties, dict):
        node["additionalProperties"] = False
        # Strict mode requires every property to be required. Optionality is
        # expressed in the type (a nullable field), not by omission.
        node["required"] = sorted(properties)

    for value in node.values():
        _tighten(value)


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A strict-mode-ready JSON Schema for `model`."""
    schema = model.model_json_schema()
    _tighten(schema)
    schema.pop("title", None)
    return schema
