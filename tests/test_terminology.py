"""Terminology is the project's central claim, so it is tested hardest."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scriptorium.domain import GlossaryTerm
from scriptorium.terminology import (
    RenderingLedger,
    TerminologyStore,
    count_occurrences,
    enforce,
    mine_terms,
    occurs,
    tokenise,
)
from scriptorium.terminology.enforcement import accepted_forms, applicable_terms, required_form

# ------------------------------------------------------------------- matching


def test_word_boundaries_prevent_substring_matches() -> None:
    assert count_occurrences("A buttonhole.", "button") == 0
    assert count_occurrences("The button.", "button") == 1


def test_matching_is_case_insensitive_by_default() -> None:
    assert occurs("The Button.", "button")
    assert not occurs("The Button.", "button", case_sensitive=True)


def test_terms_with_punctuation_fall_back_to_substring() -> None:
    """`\\b` is meaningless next to punctuation, so such terms match literally."""
    assert occurs("Use C++ here.", "C++")


def test_repeated_occurrences_are_counted() -> None:
    assert count_occurrences("button, button, button", "button") == 3


# ---------------------------------------------------------------- enforcement


def test_a_satisfied_term_produces_no_violation(glossary: list[GlossaryTerm]) -> None:
    assert enforce("Press the button.", "Pressione o botão.", glossary) == []


def test_forbidden_variant_is_a_violation(glossary: list[GlossaryTerm]) -> None:
    violations = enforce("Press the button.", "Pressione a tecla.", glossary)
    assert [v.detail for v in violations] == ["forbidden variant used"]
    assert violations[0].found == "tecla"


def test_absent_required_form_is_a_violation(glossary: list[GlossaryTerm]) -> None:
    violations = enforce("Press the button.", "Pressione o elemento.", glossary)
    assert [v.detail for v in violations] == ["no accepted target form present"]


def test_a_term_that_does_not_occur_is_not_checked(glossary: list[GlossaryTerm]) -> None:
    assert enforce("Nothing relevant.", "Nada relevante.", glossary) == []


def test_do_not_translate_requires_the_source_form() -> None:
    term = GlossaryTerm(source_term="SAP", target_term="SAP", do_not_translate=True)
    assert required_form(term) == "SAP"
    assert enforce("Use SAP here.", "Use SAP aqui.", [term]) == []
    assert enforce("Use SAP here.", "Use seiva aqui.", [term])


def test_applicable_terms_ignores_placeholder_contents() -> None:
    """A variable named `{folder}` is not an occurrence of the term "folder".

    Callers pass protected text; this asserts the consequence that matters.
    """
    folder = GlossaryTerm(source_term="folder", target_term="pasta")
    assert applicable_terms("Delete from ⟦PH0⟧.", [folder]) == []
    assert applicable_terms("Delete from the folder.", [folder]) == [folder]


def test_a_forbidden_variant_wins_over_a_present_required_form(
    glossary: list[GlossaryTerm],
) -> None:
    """Using both forms is still an error: the document would be inconsistent."""
    violations = enforce("Press the button.", "Pressione o botão ou a tecla.", glossary)
    assert violations[0].detail == "forbidden variant used"


# --------------------------------------------------------------------- store


def test_store_loads_the_shipped_glossary() -> None:
    store = TerminologyStore.from_yaml(Path("config/glossary.yaml"))
    assert store.get("button") is not None
    assert store.get("driver") is not None
    assert store.get("driver").do_not_translate is True  # type: ignore[union-attr]


def test_store_add_replaces_an_existing_term() -> None:
    store = TerminologyStore([GlossaryTerm(source_term="button", target_term="botão")])
    store.add(GlossaryTerm(source_term="button", target_term="tecla"))
    assert len(store.terms) == 1
    assert store.get("button").target_term == "tecla"  # type: ignore[union-attr]


def test_store_enforce_matches_the_free_function(glossary: list[GlossaryTerm]) -> None:
    store = TerminologyStore(glossary)
    assert store.enforce("Press the button.", "Pressione a tecla.") == enforce(
        "Press the button.", "Pressione a tecla.", glossary
    )


# -------------------------------------------------------------------- ledger


def test_ledger_establishes_the_modal_rendering(glossary: list[GlossaryTerm]) -> None:
    ledger = RenderingLedger(glossary)
    ledger.observe("The button here.", "O botão aqui.")
    ledger.observe("The button there.", "O botão ali.")
    ledger.observe("The button again.", "A tecla de novo.")
    assert ledger.established("button") == "botão"
    assert ledger.renderings("button") == {"botão": 2, "tecla": 1}


def test_ledger_reports_disagreements(glossary: list[GlossaryTerm]) -> None:
    ledger = RenderingLedger(glossary)
    ledger.observe("The button here.", "O botão aqui.")
    ledger.observe("The button there.", "A tecla ali.")
    assert "button" in ledger.disagreements()


def test_ledger_is_silent_about_unseen_terms(glossary: list[GlossaryTerm]) -> None:
    ledger = RenderingLedger(glossary)
    assert ledger.established("button") is None
    assert ledger.disagreements() == {}


def test_ledger_only_sees_known_forms(glossary: list[GlossaryTerm]) -> None:
    """The documented limitation, asserted so it cannot be forgotten.

    An unanticipated rendering is invisible to the ledger. This is why mining
    feeds the term base, and why word alignment is the phase-3 upgrade.
    """
    ledger = RenderingLedger(glossary)
    ledger.observe("The button here.", "O comando aqui.")
    assert ledger.established("button") is None


# -------------------------------------------------------------------- mining


def test_tokenise_keeps_hyphens_and_apostrophes() -> None:
    assert tokenise("well-known don't 42 x") == ["well-known", "don't", "x"]


def test_mining_requires_recurrence() -> None:
    candidates = mine_terms(["unique phrase here"], min_frequency=3)
    assert candidates == []


def test_mining_finds_a_repeated_multiword_term() -> None:
    segments = [
        "The translation memory stores approved pairs.",
        "Our translation memory grows with every post-edit.",
        "A translation memory is not a glossary.",
    ]
    found = {c.text for c in mine_terms(segments, min_frequency=3)}
    assert "translation memory" in found


def test_mining_rejects_phrases_whose_words_do_not_travel_together() -> None:
    """Collocation test: "of the system" recurs but is not terminology."""
    segments = [
        "The behaviour of the system is odd. The system is slow. A system failure.",
        "Part of the system broke. The system restarted. System logs are long.",
        "Most of the system works. The system reports errors. System time drifts.",
    ]
    found = {c.text for c in mine_terms(segments, min_frequency=3)}
    assert "system" in found
    assert "of the system" not in found


def test_mining_drops_edge_stopwords() -> None:
    segments = ["the report is here"] * 4
    assert all(not c.text.startswith("the ") for c in mine_terms(segments, min_frequency=3))


def test_mining_prefers_capitalised_terms() -> None:
    plain = ["we ship widgets every week", "widgets ship weekly", "more widgets ship"]
    titled = ["we ship Widgets every week", "more Widgets ship", "Widgets ship weekly"]
    plain_score = next(c.score for c in mine_terms(plain, min_frequency=3) if c.text == "widgets")
    titled_score = next(c.score for c in mine_terms(titled, min_frequency=3) if c.text == "widgets")
    assert titled_score > plain_score


def test_mining_returns_proposals_not_terms() -> None:
    """Mining proposes; a human approves. The types enforce the distinction."""
    candidates = mine_terms(["the button is here. the button is green. a button."] * 3)
    assert candidates
    assert not any(isinstance(c, GlossaryTerm) for c in candidates)


def test_mining_respects_max_terms() -> None:
    # Distinct alphabetic words: the tokenizer drops digits, so "term0" and
    # "term1" would both reduce to "term".
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    segments = [f"{word} {word} {word}" for word in words]
    assert len(mine_terms(segments, min_frequency=3, max_terms=5)) == 5


@pytest.mark.parametrize("bad", [0, -1])
def test_mining_handles_degenerate_frequency(bad: int) -> None:
    # A min_frequency below 1 must not crash; everything simply qualifies.
    assert mine_terms(["a word"], min_frequency=bad) != []


# ------------------------------------------- acceptable vs forbidden variants


def test_an_acceptable_variant_passes_enforcement(glossary: list[GlossaryTerm]) -> None:
    """Enforcement must not fail a correct translation.

    Both hyphenations of "e-mail" are correct Brazilian Portuguese. A term base
    that accepted only one would report a violation against a translation a
    reviewer would approve, which would make the compliance number meaningless.
    """
    assert enforce("Check your email.", "Confira seu e-mail.", glossary) == []
    assert enforce("Check your email.", "Confira seu email.", glossary) == []


def test_a_forbidden_variant_still_fails(glossary: list[GlossaryTerm]) -> None:
    violations = enforce("Check your email.", "Confira seu correio eletrônico.", glossary)
    assert [v.detail for v in violations] == ["forbidden variant used"]


def test_a_do_not_translate_term_accepts_only_itself() -> None:
    term = GlossaryTerm(
        source_term="driver",
        target_term="driver",
        acceptable_variants=("motorista",),
        do_not_translate=True,
    )
    assert accepted_forms(term) == ("driver",)
    assert enforce("Install the driver.", "Instale o motorista.", [term]) != []


def test_a_variant_cannot_be_both_acceptable_and_forbidden() -> None:
    with pytest.raises(ValidationError):
        GlossaryTerm(
            source_term="email",
            target_term="e-mail",
            acceptable_variants=("email",),
            forbidden_variants=("email",),
        )


def test_a_term_cannot_forbid_its_own_target_form() -> None:
    with pytest.raises(ValidationError):
        GlossaryTerm(source_term="email", target_term="e-mail", forbidden_variants=("e-mail",))


# --------------------------------------- consistency beyond the term base


def test_the_ledger_records_which_acceptable_form_the_document_used(
    glossary: list[GlossaryTerm],
) -> None:
    """The point of the split: both forms comply, only one is consistent."""
    ledger = RenderingLedger(glossary)
    ledger.observe("Check your email.", "Confira seu e-mail.")
    ledger.observe("Send an email.", "Envie um e-mail.")
    ledger.observe("The email arrived.", "O email chegou.")

    assert ledger.renderings("email") == {"e-mail": 2, "email": 1}
    assert ledger.established("email") == "e-mail"
    assert "email" in ledger.disagreements()


def test_a_document_using_one_form_throughout_has_no_disagreement(
    glossary: list[GlossaryTerm],
) -> None:
    ledger = RenderingLedger(glossary)
    ledger.observe("Check your email.", "Confira seu email.")
    ledger.observe("Send an email.", "Envie um email.")
    assert ledger.disagreements() == {}
    assert ledger.established("email") == "email"
