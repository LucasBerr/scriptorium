"""The heuristic quality estimator.

These tests pin down *ordering*, not values. The constants are hand-set and will
be replaced by fitted ones in phase 3; a test asserting that a particular
translation scores 0.83 would then fail for a good reason, which makes it a bad
test. What must survive the replacement is the direction: a broken translation
must never outscore a sound one.
"""

from __future__ import annotations

import pytest

from scriptorium.domain import GlossaryTerm
from scriptorium.quality.estimation import estimate_quality

SOURCE = "Press the button."
GOOD = "Pressione o botão."


def test_the_estimate_is_labelled_a_heuristic() -> None:
    """A hand-tuned number presented as a measurement is the thing to avoid."""
    assert estimate_quality(SOURCE, GOOD).heuristic is True


def test_an_empty_translation_scores_zero() -> None:
    estimate = estimate_quality(SOURCE, "   ")
    assert estimate.score == 0.0
    assert estimate.signals.is_empty is True


def test_a_sound_translation_scores_well() -> None:
    assert estimate_quality(SOURCE, GOOD).score > 0.9


def test_a_terminology_violation_lowers_the_score(glossary: list[GlossaryTerm]) -> None:
    good = estimate_quality(SOURCE, GOOD, glossary=glossary)
    bad = estimate_quality(SOURCE, "Pressione a tecla.", glossary=glossary)
    assert bad.score < good.score
    assert bad.signals.terminology_violations == 1


def test_violation_penalties_are_capped(glossary: list[GlossaryTerm]) -> None:
    """Two violations are worse than one, but not unboundedly worse.

    Without a cap, a long segment with many flagged terms would score zero and
    be indistinguishable from an empty one, which would make routing useless
    exactly where review time is most valuable.
    """
    one = estimate_quality("Press the button.", "Pressione a tecla.", glossary=glossary)
    two = estimate_quality(
        "Press the button in the folder.",
        "Pressione a tecla no diretório.",
        glossary=glossary,
    )
    assert two.signals.terminology_violations == 2
    assert two.score < one.score
    assert two.score > 0.0


def test_a_dropped_placeholder_lowers_the_score() -> None:
    source = "Delete ⟦PH0⟧ files."
    intact = estimate_quality(source, "Excluir ⟦PH0⟧ arquivos.")
    dropped = estimate_quality(source, "Excluir arquivos.")
    assert dropped.score < intact.score
    assert intact.signals.placeholders_intact is True
    assert dropped.signals.placeholders_intact is False


def test_untranslated_output_is_capped_not_zeroed() -> None:
    """An identical string is usually untranslated, but not always.

    A do-not-translate term or a short UI label can legitimately come back
    unchanged, so this caps the score rather than declaring a failure.
    """
    estimate = estimate_quality(SOURCE, SOURCE)
    assert estimate.signals.identical_to_source is True
    assert 0.0 < estimate.score <= 0.4
    assert estimate.score < estimate_quality(SOURCE, GOOD).score


def test_a_segment_with_no_letters_may_repeat_the_source() -> None:
    """A page number or a code is not an untranslated segment."""
    estimate = estimate_quality("1,024", "1,024")
    assert estimate.signals.identical_to_source is False


def test_gross_length_deviation_lowers_the_score() -> None:
    long_source = "The quick brown fox jumped over the lazy dog near the river."
    reasonable = estimate_quality(
        long_source, "A rápida raposa marrom saltou sobre o cão preguiçoso perto do rio."
    )
    truncated = estimate_quality(long_source, "A raposa.")
    assert truncated.score < reasonable.score
    assert truncated.signals.length_ratio < 0.3


def test_the_score_stays_inside_the_unit_interval() -> None:
    """Every penalty stacked at once must not produce a negative score."""
    source = "Press the button in the folder ⟦PH0⟧."
    estimate = estimate_quality(
        source,
        "x",
        glossary=[
            GlossaryTerm(source_term="button", target_term="botão"),
            GlossaryTerm(source_term="folder", target_term="pasta"),
        ],
    )
    assert 0.0 <= estimate.score <= 1.0


def test_the_signals_are_reported_alongside_the_score() -> None:
    """Phase 3 trains on these, so they are part of the contract, not debug output."""
    estimate = estimate_quality(SOURCE, GOOD)
    assert set(estimate.signals.model_dump()) == {
        "placeholders_intact",
        "terminology_violations",
        "length_ratio",
        "identical_to_source",
        "is_empty",
    }
    assert estimate.signals.length_ratio == pytest.approx(18 / 17, rel=1e-3)
