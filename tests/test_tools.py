"""The tools, their schemas, and the record of what was called."""

from __future__ import annotations

import pytest

from scriptorium.domain import GlossaryTerm, TMEntry
from scriptorium.memory import TranslationMemory
from scriptorium.orchestrator import Toolbox, ToolInputError, UnknownToolError
from scriptorium.terminology import RenderingLedger, TerminologyStore


@pytest.fixture
def toolbox(glossary: list[GlossaryTerm]) -> Toolbox:
    memory = TranslationMemory()
    memory.add(
        TMEntry(
            source_text="Press the button.",
            target_text="Pressione o botão.",
            source_lang="en",
            target_lang="pt-BR",
            provenance="seed",
            approved_by="reviewer",
        )
    )
    store = TerminologyStore(glossary)
    return Toolbox(
        source_lang="en",
        target_lang="pt-BR",
        memory=memory,
        terminology=store,
        ledger=RenderingLedger(store.terms),
    )


# ------------------------------------------------------------------- schemas


def test_all_six_tools_are_defined(toolbox: Toolbox) -> None:
    names = {definition.name for definition in toolbox.definitions()}
    assert names == {
        "lookup_terminology",
        "search_translation_memory",
        "check_consistency",
        "estimate_quality",
        "propose_glossary_entry",
        "flag_for_human",
    }


def test_schemas_are_strict_mode_ready(toolbox: Toolbox) -> None:
    """Strict mode is only enforced when both conditions hold.

    A schema missing either is accepted by the provider and then silently not
    enforced, which is worse than no schema at all because it looks validated.
    """
    for definition in toolbox.definitions():
        schema = definition.input_schema
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_every_tool_has_a_description(toolbox: Toolbox) -> None:
    assert all(len(d.description) > 20 for d in toolbox.definitions())


# ------------------------------------------------------------------ dispatch


def test_unknown_tool_is_refused(toolbox: Toolbox) -> None:
    with pytest.raises(UnknownToolError):
        toolbox.call("nonexistent_tool", {})


def test_bad_arguments_are_refused_before_the_handler_runs(toolbox: Toolbox) -> None:
    with pytest.raises(ToolInputError):
        toolbox.call("search_translation_memory", {"text": "hi", "k": 99})
    assert toolbox.calls == ()


def test_missing_argument_is_refused(toolbox: Toolbox) -> None:
    with pytest.raises(ToolInputError):
        toolbox.call("lookup_terminology", {})


# ------------------------------------------------------------------- handlers


def test_lookup_terminology_returns_only_applicable_terms(toolbox: Toolbox) -> None:
    result = toolbox.call("lookup_terminology", {"text": "Press the button."})
    assert [term["source_term"] for term in result["terms"]] == ["button"]


def test_lookup_terminology_marks_do_not_translate(toolbox: Toolbox) -> None:
    toolbox.terminology.add(
        GlossaryTerm(source_term="SAP", target_term="SAP", do_not_translate=True)
    )
    result = toolbox.call("lookup_terminology", {"text": "Use SAP."})
    assert result["terms"][0]["do_not_translate"] is True


def test_search_translation_memory_reports_the_hit_kind(toolbox: Toolbox) -> None:
    result = toolbox.call("search_translation_memory", {"text": "Press the button.", "k": 3})
    assert result["hits"][0]["kind"] == "exact"
    assert result["hits"][0]["target"] == "Pressione o botão."


def test_check_consistency_is_empty_before_anything_is_seen(toolbox: Toolbox) -> None:
    result = toolbox.call("check_consistency", {"source_term": "button"})
    assert result["established"] is None
    assert result["renderings"] == {}


def test_check_consistency_reports_the_established_form(toolbox: Toolbox) -> None:
    toolbox.ledger.observe("The button.", "O botão.")
    toolbox.ledger.observe("A button.", "Um botão.")
    result = toolbox.call("check_consistency", {"source_term": "button"})
    assert result["established"] == "botão"


def test_estimate_quality_flags_a_terminology_violation(toolbox: Toolbox) -> None:
    good = toolbox.call(
        "estimate_quality",
        {"source": "Press the button.", "translation": "Pressione o botão."},
    )
    bad = toolbox.call(
        "estimate_quality",
        {"source": "Press the button.", "translation": "Pressione a tecla."},
    )
    assert bad["score"] < good["score"]
    assert bad["signals"]["terminology_violations"] == 1
    assert bad["heuristic"] is True


def test_proposals_are_recorded_not_applied(toolbox: Toolbox) -> None:
    """The model may propose; a person disposes."""
    result = toolbox.call(
        "propose_glossary_entry",
        {"source_term": "widget", "target_term": "componente", "rationale": "recurs often"},
    )
    assert result["status"] == "recorded_for_approval"
    assert toolbox.terminology.get("widget") is None
    assert len(toolbox.proposals) == 1


def test_approval_is_not_reachable_from_the_tools(toolbox: Toolbox) -> None:
    """There is no approve tool, deliberately: it is the term base's one checkpoint."""
    assert "approve" not in {d.name for d in toolbox.definitions()}

    toolbox.call(
        "propose_glossary_entry",
        {"source_term": "widget", "target_term": "componente", "rationale": "recurs"},
    )
    term = toolbox.approve_proposal(0, approved_by="lucas")
    assert toolbox.terminology.get("widget") == term
    assert "approved by lucas" in term.notes
    assert toolbox.proposals == []


def test_flag_for_human_accumulates(toolbox: Toolbox) -> None:
    toolbox.call("flag_for_human", {"reason": "ambiguous pronoun"})
    toolbox.call("flag_for_human", {"reason": "unclear register"})
    assert [flag.reason for flag in toolbox.flags] == [
        "ambiguous pronoun",
        "unclear register",
    ]


# -------------------------------------------------------------------- record


def test_calls_are_recorded_with_arguments_and_results(toolbox: Toolbox) -> None:
    """What makes an agent reviewable after the fact instead of a matter of trust."""
    toolbox.call("lookup_terminology", {"text": "Press the button."})
    call = toolbox.calls[0]
    assert call.name == "lookup_terminology"
    assert call.arguments == {"text": "Press the button."}
    assert "terms" in call.result


def test_drain_hands_over_and_clears(toolbox: Toolbox) -> None:
    toolbox.call("flag_for_human", {"reason": "x"})
    assert len(toolbox.drain()) == 1
    assert toolbox.calls == ()
