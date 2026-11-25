"""Inflection rules, and the matching they make possible.

These tests exist because of a measured defect, not a hypothetical one: with
literal matching, the project's own reference translations scored 90.9%
terminology compliance, and every failure was a correct translation that had
been inflected (ADR 0004). The first test in each section is that failure.
"""

from __future__ import annotations

import pytest

from scriptorium.domain import GlossaryTerm
from scriptorium.terminology import enforce, inflections
from scriptorium.terminology.enforcement import accepted_forms, applicable_terms, term_language
from scriptorium.terminology.matching import count_occurrences, occurs
from scriptorium.terminology.store import RenderingLedger

# ------------------------------------------------------------ the rules


def test_portuguese_plurals() -> None:
    assert "arquivos" in inflections("arquivo", "pt-BR")
    assert "pastas" in inflections("pasta", "pt-BR")
    assert "configurações" in inflections("configuração", "pt-BR")
    assert "bancos" in inflections("banco", "pt-BR")
    assert "lugares" in inflections("lugar", "pt-BR")
    assert "homens" in inflections("homem", "pt-BR")
    assert "finais" in inflections("final", "pt-BR")
    assert "papéis" in inflections("papel", "pt-BR")


def test_an_ao_word_gets_all_three_plural_patterns() -> None:
    """Spelling does not say which is right, so the rules do not guess.

    Over-generation is the safe direction: an extra form can only cause a
    violation to be missed, while a missing form fails a correct translation.
    """
    forms = inflections("botão", "pt-BR")
    assert {"botões", "botães", "botãos"} <= forms


def test_a_portuguese_infinitive_yields_the_forms_prose_uses() -> None:
    forms = inflections("entrar", "pt-BR")
    assert {"entre", "entro", "entra", "entrou", "entrando", "entrado"} <= forms


def test_an_already_plural_word_is_not_pluralised_again() -> None:
    assert inflections("configurações", "pt-BR") == frozenset({"configurações"})


def test_english_plurals_and_verb_forms() -> None:
    assert {"files", "filed", "filing"} <= inflections("file", "en")
    assert "libraries" in inflections("library", "en")
    assert "boxes" in inflections("box", "en")
    assert "children" in inflections("child", "en")


def test_an_unsupported_language_yields_the_word_alone() -> None:
    """Applying Portuguese rules to German would produce confident nonsense."""
    assert inflections("Ordner", "de") == frozenset({"Ordner"})


def test_inflection_is_case_folded() -> None:
    assert "arquivos" in inflections("Arquivo", "pt-BR")


# --------------------------------------------------------------- matching


def test_a_plural_in_the_target_counts_as_the_term() -> None:
    """The measured failure: 'Todos os arquivos' is the term 'arquivo'."""
    assert count_occurrences("Todos os arquivos nessa pasta.", "arquivo", language="pt-BR") == 1


def test_a_conjugated_verb_in_the_target_counts_as_the_term() -> None:
    assert occurs("Entre com seu endereço.", "entrar", language="pt-BR")


def test_a_plural_in_the_source_counts_as_the_term() -> None:
    assert count_occurrences("Delete the files now.", "file", language="en") == 1


def test_a_multi_word_term_inflects_its_head() -> None:
    assert (
        count_occurrences("Os bancos de dados estão prontos.", "banco de dados", language="pt-BR")
        == 1
    )


def test_short_function_words_are_not_inflected() -> None:
    """Otherwise 'de' would widen to 'des' and match half the language."""
    assert count_occurrences("Um teste des dados.", "banco de dados", language="pt-BR") == 0


def test_word_boundaries_survive_inflection() -> None:
    assert count_occurrences("buttonhole", "button", language="en") == 0
    assert count_occurrences("arquivologia", "arquivo", language="pt-BR") == 0


def test_literal_matching_is_still_available() -> None:
    """Mining and the forbidden-variant check want the form as written."""
    assert count_occurrences("Todos os arquivos.", "arquivo") == 0


# ------------------------------------------------------------ enforcement


def test_an_inflected_translation_is_compliant() -> None:
    """The defect ADR 0004 records: this used to be a violation."""
    term = GlossaryTerm(source_term="file", target_term="arquivo")
    assert enforce("Every file is copied.", "Todos os arquivos são copiados.", [term]) == []


def test_a_conjugated_translation_is_compliant() -> None:
    term = GlossaryTerm(source_term="sign in", target_term="entrar")
    assert enforce("Sign in with your email.", "Entre com seu e-mail.", [term]) == []


def test_a_wrong_term_is_still_a_violation_after_inflection() -> None:
    term = GlossaryTerm(source_term="file", target_term="arquivo", forbidden_variants=("ficheiro",))
    violations = enforce("Every file is copied.", "Todos os ficheiros são copiados.", [term])
    assert [v.detail for v in violations] == ["no accepted target form present"]


def test_a_forbidden_variant_is_matched_literally_not_inflected() -> None:
    """`logar` must not widen to `logo`, which is an ordinary adverb.

    Inflecting forbidden variants is the one place over-generation would fail
    correct translations, so it is the one place the rules are switched off.
    """
    term = GlossaryTerm(
        source_term="sign in",
        target_term="entrar",
        forbidden_variants=("logar",),
    )
    assert enforce("Sign in here.", "Entre aqui, logo após o início.", [term]) == []


def test_a_do_not_translate_term_is_not_inflected(glossary: list[GlossaryTerm]) -> None:
    del glossary
    term = GlossaryTerm(source_term="driver", target_term="driver", do_not_translate=True)
    assert term_language(term, "pt-BR") is None
    assert accepted_forms(term) == ("driver",)


def test_explicit_forms_replace_the_generated_ones() -> None:
    """An irregular is a person's decision, made once, visible in the term base."""
    term = GlossaryTerm(
        source_term="status",
        target_term="status",
        inflected_forms=("status",),
    )
    assert term_language(term, "pt-BR") is None
    assert accepted_forms(term) == ("status", "status")
    assert enforce("Check the status.", "Verifique o status.", [term]) == []


def test_applicable_terms_sees_an_inflected_source() -> None:
    term = GlossaryTerm(source_term="folder", target_term="pasta")
    assert applicable_terms("Open both folders.", [term]) == [term]


def test_a_term_inside_a_placeholder_is_still_not_an_occurrence() -> None:
    """Inflection must not reopen the hole the protected space closed."""
    term = GlossaryTerm(source_term="folder", target_term="pasta")
    assert applicable_terms("Delete ⟦PH0⟧ from ⟦PH1⟧?", [term]) == []


# --------------------------------------------------------------- ledger


def test_the_ledger_counts_a_plural_as_the_singular_rendering() -> None:
    """Otherwise grammar would register as a consistency failure."""
    term = GlossaryTerm(source_term="file", target_term="arquivo")
    ledger = RenderingLedger([term])
    ledger.observe("The file is here.", "O arquivo está aqui.")
    ledger.observe("Every file is copied.", "Todos os arquivos são copiados.")

    assert ledger.renderings("file") == {"arquivo": 2}
    assert ledger.disagreements() == {}


def test_the_ledger_still_sees_a_genuine_disagreement() -> None:
    term = GlossaryTerm(
        source_term="email",
        target_term="e-mail",
        acceptable_variants=("email",),
    )
    ledger = RenderingLedger([term])
    ledger.observe("Your email.", "Seu e-mail.")
    ledger.observe("Their email.", "O email deles.")
    assert ledger.disagreements() == {"email": {"e-mail": 1, "email": 1}}


@pytest.mark.parametrize("language", ["pt-BR", "pt", "portuguese", "PT-br"])
def test_language_codes_are_accepted_in_the_forms_people_write(language: str) -> None:
    assert "arquivos" in inflections("arquivo", language)
