"""The custom metrics are the ones this project is judged on, so they are tested hardest."""

from __future__ import annotations

from scriptorium.domain import GlossaryTerm
from scriptorium.quality import metrics
from scriptorium.segmentation.placeholders import check_integrity, protect


def test_terminology_compliance_accepts_the_required_form(
    glossary: list[GlossaryTerm],
) -> None:
    report = metrics.terminology_compliance(["Press the button."], ["Pressione o botão."], glossary)
    assert report.applicable == 1
    assert report.rate == 1.0
    assert report.violations == ()


def test_terminology_compliance_catches_a_forbidden_variant(
    glossary: list[GlossaryTerm],
) -> None:
    report = metrics.terminology_compliance(["Press the button."], ["Pressione a tecla."], glossary)
    assert report.rate == 0.0
    assert report.violations[0].detail == "forbidden variant used"
    assert report.violations[0].found == "tecla"


def test_terminology_compliance_catches_an_absent_term(glossary: list[GlossaryTerm]) -> None:
    report = metrics.terminology_compliance(
        ["Press the button."], ["Pressione o elemento."], glossary
    )
    assert report.rate == 0.0
    assert report.violations[0].detail == "no accepted target form present"


def test_terminology_is_only_applicable_when_the_term_occurs(
    glossary: list[GlossaryTerm],
) -> None:
    report = metrics.terminology_compliance(["Nothing relevant."], ["Nada relevante."], glossary)
    assert report.applicable == 0
    assert report.rate == 1.0


def test_term_matching_respects_word_boundaries() -> None:
    """'button' must not match inside 'buttonhole'."""
    term = GlossaryTerm(source_term="button", target_term="botão")
    report = metrics.terminology_compliance(["A buttonhole."], ["Uma casa de botão."], [term])
    assert report.applicable == 0


def test_consistency_is_the_share_in_the_modal_form(glossary: list[GlossaryTerm]) -> None:
    sources = ["The button here.", "The button there.", "The button again."]
    hypotheses = ["O botão aqui.", "O botão ali.", "O botao de novo."]
    report = metrics.intra_document_consistency(sources, hypotheses, glossary)
    assert report.observed == 3
    assert report.in_modal_form == 2
    assert report.rate == 2 / 3


def test_consistency_is_perfect_when_every_rendering_agrees(
    glossary: list[GlossaryTerm],
) -> None:
    sources = ["The button here.", "The button there."]
    hypotheses = ["O botão aqui.", "O botão ali."]
    assert metrics.intra_document_consistency(sources, hypotheses, glossary).rate == 1.0


def test_placeholder_integrity_ignores_segments_without_placeholders() -> None:
    plain = protect("No placeholders here.")
    assert metrics.placeholder_integrity([check_integrity(plain.placeholders, "Nada aqui.")]) == 1.0


def test_placeholder_integrity_counts_only_relevant_segments() -> None:
    with_ph = protect("Delete %d files.")
    good = check_integrity(with_ph.placeholders, with_ph.text)
    bad = check_integrity(with_ph.placeholders, "Apagar arquivos.")
    assert metrics.placeholder_integrity([good, bad]) == 0.5


def test_string_metrics_are_in_range() -> None:
    hyps = ["O gato sentou no tapete."]
    refs = ["O gato sentou no tapete."]
    assert metrics.chrf(hyps, refs) > 99.0
    assert metrics.bleu(hyps, refs) > 99.0
    assert metrics.ter(hyps, refs) == 0.0


def test_paired_bootstrap_finds_a_real_improvement() -> None:
    baseline = [10.0] * 40
    system = [20.0] * 40
    result = metrics.paired_bootstrap(baseline, system, samples=500)
    assert result.mean_difference == 10.0
    assert result.is_significant


def test_paired_bootstrap_rejects_noise() -> None:
    baseline = [10.0, 20.0, 30.0, 40.0] * 10
    system = [20.0, 10.0, 40.0, 30.0] * 10
    result = metrics.paired_bootstrap(baseline, system, samples=500)
    assert not result.is_significant


def test_paired_bootstrap_is_deterministic() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [2.0, 3.0, 4.0, 5.0, 6.0]
    first = metrics.paired_bootstrap(a, b, samples=200)
    second = metrics.paired_bootstrap(a, b, samples=200)
    assert first.p_value == second.p_value
