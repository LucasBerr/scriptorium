"""Strict-mode JSON Schema generation.

The failure mode this guards against is silent: a provider accepts a schema that
is not strict-mode-ready and then does not enforce it. The output still looks
structured, and nothing says the guarantee was never in effect.
"""

from __future__ import annotations

from pydantic import BaseModel

from scriptorium.domain import MQMCategory, MQMSeverity, Review
from scriptorium.providers.base import TranslationPayload
from scriptorium.providers.schema import strict_json_schema


class Inner(BaseModel):
    flag: bool = True
    label: str = "x"


class Outer(BaseModel):
    name: str
    inner: Inner
    many: tuple[Inner, ...] = ()
    optional: str | None = None


def test_additional_properties_is_forbidden_at_the_top_level() -> None:
    schema = strict_json_schema(Outer)
    assert schema["additionalProperties"] is False


def test_every_property_is_required_even_when_it_has_a_default() -> None:
    """Strict mode has no notion of an optional property.

    Optionality is expressed in the type — a nullable field — not by omission
    from `required`. A schema that leaves a defaulted field out is rejected.
    """
    schema = strict_json_schema(Outer)
    assert set(schema["required"]) == {"name", "inner", "many", "optional"}


def test_nested_models_are_tightened_through_defs() -> None:
    """The bug this catches: tightening only the top level.

    The top-level object looks strict, the nested one is wide open, and the
    provider rejects the schema — or worse, accepts it.
    """
    schema = strict_json_schema(Outer)
    inner = schema["$defs"]["Inner"]
    assert inner["additionalProperties"] is False
    assert set(inner["required"]) == {"flag", "label"}


def test_the_title_is_dropped() -> None:
    assert "title" not in strict_json_schema(Outer)


def test_an_object_with_no_fields_is_still_closed() -> None:
    """Pydantic emits `properties: {}`, so the walk closes it and requires nothing.

    An empty `required` list is valid, and closing the object is the point: a
    model with no fields should accept no fields.
    """

    class Empty(BaseModel):
        pass

    schema = strict_json_schema(Empty)
    assert schema["additionalProperties"] is False
    assert schema["required"] == []


def test_a_node_that_is_not_an_object_is_untouched() -> None:
    """Only objects carry `properties`; scalars must not grow the keyword."""

    class Scalars(BaseModel):
        count: int
        ratio: float

    schema = strict_json_schema(Scalars)
    assert "additionalProperties" not in schema["properties"]["count"]
    assert "required" not in schema["properties"]["ratio"]


def test_the_review_schema_is_strict_all_the_way_down() -> None:
    """The one that matters: the reviewer's output is parsed, not read."""
    schema = strict_json_schema(Review)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"verdict", "annotations"}

    annotation = schema["$defs"]["MQMAnnotation"]
    assert annotation["additionalProperties"] is False
    assert set(annotation["required"]) == {"category", "severity", "span", "comment"}


def test_the_enums_reach_the_schema_as_closed_sets() -> None:
    """Severity drives routing, so an unexpected value is not a parse detail."""
    schema = strict_json_schema(Review)
    assert set(schema["$defs"]["MQMSeverity"]["enum"]) == {s.value for s in MQMSeverity}
    assert set(schema["$defs"]["MQMCategory"]["enum"]) == {c.value for c in MQMCategory}


def test_the_translation_payload_is_strict() -> None:
    schema = strict_json_schema(TranslationPayload)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
