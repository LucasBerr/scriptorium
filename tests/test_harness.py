"""The harness is the project's instrument; a broken instrument invalidates everything."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scriptorium.domain import GlossaryTerm
from scriptorium.evaluation import Harness, TestCase, load_test_set, render_markdown_table
from scriptorium.providers import EchoProvider, ScriptedProvider
from scriptorium.segmentation.placeholders import protect

CASES = [
    TestCase(source="Press the button.", reference="Pressione o botão."),
    TestCase(source="Delete %d files from {folder}.", reference="Apagar %d arquivos de {folder}."),
    TestCase(source="The button is green.", reference="O botão é verde."),
]


def _oracle() -> dict[str, str]:
    """Recordings for a hypothetical perfect system.

    Keyed and valued in protected space, because that is where a provider lives:
    it is handed text with placeholder tokens and must hand tokens back.
    """
    return {protect(c.source).text: protect(c.reference).text for c in CASES}


def test_load_test_set_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(
        "\n".join(json.dumps({"source": c.source, "reference": c.reference}) for c in CASES),
        encoding="utf-8",
    )
    assert len(load_test_set(path)) == 3


def test_load_test_set_names_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text('{"source": "a", "reference": "b"}\n{"source": "missing ref"}\n', "utf-8")
    with pytest.raises(ValueError, match=":2:"):
        load_test_set(path)


def test_harness_refuses_an_empty_test_set() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Harness([], source_lang="en", target_lang="pt-BR")


def test_a_perfect_arm_scores_perfectly(glossary: list[GlossaryTerm]) -> None:
    """An arm that replays the references must top out on every metric.

    This is the harness's self-check: if the ceiling is not reachable, the
    metrics are wired up wrong and no comparison between arms means anything.
    """
    harness = Harness(CASES, source_lang="en", target_lang="pt-BR", glossary=glossary)
    result = harness.run_arm("perfect", ScriptedProvider(_oracle()), with_context=True)

    assert result.chrf > 99.0
    assert result.ter == 0.0
    assert result.terminology_compliance == 1.0
    assert result.intra_document_consistency == 1.0
    assert result.placeholder_integrity == 1.0


def test_echo_arm_is_the_floor(glossary: list[GlossaryTerm]) -> None:
    harness = Harness(CASES, source_lang="en", target_lang="pt-BR", glossary=glossary)
    result = harness.run_arm("echo", EchoProvider())
    assert result.chrf < 60.0
    # Echoing the source keeps the placeholders, so that metric stays at its ceiling.
    assert result.placeholder_integrity == 1.0
    # But no required target term is ever produced.
    assert result.terminology_compliance == 0.0


def test_context_is_what_distinguishes_arms() -> None:
    """B1 and B2 differ only in what the harness puts into the request."""
    harness = Harness(CASES, source_lang="en", target_lang="pt-BR")
    without = harness._request(1, CASES[1], with_context=False)
    with_context = harness._request(1, CASES[1], with_context=True)
    assert without.context_before == ""
    assert with_context.context_before == CASES[0].source
    assert with_context.context_after == CASES[2].source


def test_cost_per_1k_words_is_reported(glossary: list[GlossaryTerm]) -> None:
    harness = Harness(CASES, source_lang="en", target_lang="pt-BR", glossary=glossary)
    result = harness.run_arm("echo", EchoProvider())
    assert result.cost_usd == 0.0
    assert result.cost_per_1k_words == 0.0
    assert result.segments == 3


def test_comparison_returns_a_bootstrap_result(glossary: list[GlossaryTerm]) -> None:
    harness = Harness(CASES, source_lang="en", target_lang="pt-BR", glossary=glossary)
    floor = harness.run_arm("echo", EchoProvider())
    ceiling = harness.run_arm("perfect", ScriptedProvider(_oracle()))
    comparison = Harness.compare(floor, ceiling)
    assert comparison.mean_difference > 0
    assert comparison.samples == 1000


def test_markdown_table_has_cost_beside_quality(glossary: list[GlossaryTerm]) -> None:
    harness = Harness(CASES, source_lang="en", target_lang="pt-BR", glossary=glossary)
    table = render_markdown_table([harness.run_arm("echo", EchoProvider())])
    assert "chrF++" in table
    assert "Cost/1k words" in table
    assert "Cache" in table


def test_a_term_inside_a_placeholder_is_not_a_terminology_occurrence(
    glossary: list[GlossaryTerm],
) -> None:
    """`{folder}` is a variable name, not the word "folder".

    Regression test for a real bug found while building the harness: measuring
    terminology on unprotected text made the checker demand that a variable name
    be translated, which would have shown up as a permanent 2/3 compliance
    ceiling that no model could ever have fixed.
    """
    cases = [TestCase(source="Delete from {folder}.", reference="Apagar de {folder}.")]
    harness = Harness(cases, source_lang="en", target_lang="pt-BR", glossary=glossary)
    result = harness.run_arm(
        "perfect",
        ScriptedProvider({protect(c.source).text: protect(c.reference).text for c in cases}),
    )
    assert result.terminology_compliance == 1.0
    assert result.placeholder_integrity == 1.0
