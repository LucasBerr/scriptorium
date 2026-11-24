"""Metrics.

Three of these are standard (chrF++, BLEU, TER via sacreBLEU) and three are
specific to the problem this project claims to solve. The custom three are the
interesting ones, because a terminology-controlled pipeline that reports only
BLEU has not measured the thing it was built for:

- **terminology compliance** — of the glossary terms that actually occur in a
  source segment, how many are rendered as the term base requires.
- **intra-document consistency** — when a term occurs many times, how often it
  gets the same rendering. This is the page-12-versus-page-280 problem.
- **placeholder integrity** — how often markup and format specifiers survive.

Every comparison between systems goes through :func:`paired_bootstrap`. A score
that moved without a p-value is not a result.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel
from sacrebleu.metrics import BLEU, CHRF, TER

from scriptorium.domain import GlossaryTerm
from scriptorium.segmentation.placeholders import IntegrityReport

_WORD_LIKE = re.compile(r"^[\w\s'-]+$", re.UNICODE)


def chrf(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """chrF++ (character n-grams plus word bigrams). The headline string metric.

    Preferred over BLEU for morphologically richer targets such as Portuguese,
    where BLEU's word-level matching punishes acceptable inflectional variation.
    """
    return float(CHRF(word_order=2).corpus_score(list(hypotheses), [list(references)]).score)


def bleu(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """BLEU. Reported for comparability with published numbers, not as the target."""
    return float(BLEU().corpus_score(list(hypotheses), [list(references)]).score)


def ter(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Translation edit rate. A proxy for post-editing effort; lower is better."""
    return float(TER().corpus_score(list(hypotheses), [list(references)]).score)


def sentence_chrf(hypotheses: Sequence[str], references: Sequence[str]) -> list[float]:
    """Per-segment chrF++, which is what the bootstrap resamples."""
    metric = CHRF(word_order=2)
    return [
        float(metric.sentence_score(hyp, [ref]).score)
        for hyp, ref in zip(hypotheses, references, strict=True)
    ]


def _pattern(term: str, *, case_sensitive: bool) -> re.Pattern[str]:
    """Match a term, with word boundaries when the term looks like words.

    A term containing punctuation or markup cannot rely on `\\b`, so it falls
    back to a plain substring match.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    if _WORD_LIKE.match(term):
        return re.compile(rf"\b{re.escape(term)}\b", flags)
    return re.compile(re.escape(term), flags)


def _count(text: str, term: str, *, case_sensitive: bool) -> int:
    return len(_pattern(term, case_sensitive=case_sensitive).findall(text))


class TerminologyViolation(BaseModel):
    """One place where the term base was not honoured."""

    index: int
    source_term: str
    expected: str
    found: str
    detail: str


class TerminologyReport(BaseModel):
    applicable: int = 0
    compliant: int = 0
    violations: tuple[TerminologyViolation, ...] = ()

    @property
    def rate(self) -> float:
        """Share of applicable term occurrences rendered correctly. 1.0 when none apply."""
        return 1.0 if self.applicable == 0 else self.compliant / self.applicable


def terminology_compliance(
    sources: Sequence[str],
    hypotheses: Sequence[str],
    glossary: Sequence[GlossaryTerm],
) -> TerminologyReport:
    """Check every glossary term that occurs in the source against the translation.

    A term is *applicable* to a segment when it occurs in that segment's source.
    It is compliant when the required target form is present and no forbidden
    variant is. A forbidden variant is what makes this checkable: without one,
    "did it use a synonym?" has no mechanical answer.
    """
    applicable = 0
    compliant = 0
    violations: list[TerminologyViolation] = []

    for index, (source, hypothesis) in enumerate(zip(sources, hypotheses, strict=True)):
        for term in glossary:
            if _count(source, term.source_term, case_sensitive=term.case_sensitive) == 0:
                continue
            applicable += 1
            required = term.source_term if term.do_not_translate else term.target_term

            bad_variant = next(
                (
                    variant
                    for variant in term.forbidden_variants
                    if _count(hypothesis, variant, case_sensitive=term.case_sensitive) > 0
                ),
                None,
            )
            if bad_variant is not None:
                violations.append(
                    TerminologyViolation(
                        index=index,
                        source_term=term.source_term,
                        expected=required,
                        found=bad_variant,
                        detail="forbidden variant used",
                    )
                )
                continue

            if _count(hypothesis, required, case_sensitive=term.case_sensitive) == 0:
                violations.append(
                    TerminologyViolation(
                        index=index,
                        source_term=term.source_term,
                        expected=required,
                        found="",
                        detail="required target form absent",
                    )
                )
                continue

            compliant += 1

    return TerminologyReport(
        applicable=applicable,
        compliant=compliant,
        violations=tuple(violations),
    )


class ConsistencyReport(BaseModel):
    observed: int = 0
    in_modal_form: int = 0
    per_term: dict[str, dict[str, int]] = {}

    @property
    def rate(self) -> float:
        """Share of occurrences rendered in each term's most common form."""
        return 1.0 if self.observed == 0 else self.in_modal_form / self.observed


def intra_document_consistency(
    sources: Sequence[str],
    hypotheses: Sequence[str],
    glossary: Sequence[GlossaryTerm],
) -> ConsistencyReport:
    """Measure whether a repeated term gets the same rendering every time.

    Scope and limitation, stated plainly: consistency is measured over the
    *known* variant set — the required target form plus the forbidden variants in
    the term base. A rendering nobody anticipated is invisible here, which is why
    term mining feeds the term base rather than the other way round. The honest
    version of this metric needs word alignment; this one needs a glossary, and
    says so.
    """
    per_term: dict[str, Counter[str]] = {}

    for source, hypothesis in zip(sources, hypotheses, strict=True):
        for term in glossary:
            if _count(source, term.source_term, case_sensitive=term.case_sensitive) == 0:
                continue
            required = term.source_term if term.do_not_translate else term.target_term
            counts = per_term.setdefault(term.source_term, Counter())
            for form in (required, *term.forbidden_variants):
                found = _count(hypothesis, form, case_sensitive=term.case_sensitive)
                if found:
                    counts[form] += found

    observed = 0
    in_modal = 0
    for counts in per_term.values():
        if not counts:
            continue
        observed += sum(counts.values())
        in_modal += counts.most_common(1)[0][1]

    return ConsistencyReport(
        observed=observed,
        in_modal_form=in_modal,
        per_term={term: dict(counts) for term, counts in per_term.items() if counts},
    )


def placeholder_integrity(reports: Sequence[IntegrityReport]) -> float:
    """Share of segments whose placeholders came back intact.

    Only segments that had placeholders count; a corpus of plain prose would
    otherwise score a meaningless 1.0.
    """
    relevant = [r for r in reports if r.expected]
    if not relevant:
        return 1.0
    return sum(1 for r in relevant if r.is_intact) / len(relevant)


class BootstrapResult(BaseModel):
    """Whether a difference between two systems survives resampling."""

    mean_baseline: float
    mean_system: float
    mean_difference: float
    p_value: float
    ci_low: float
    ci_high: float
    samples: int

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05


def paired_bootstrap(
    baseline_scores: Sequence[float],
    system_scores: Sequence[float],
    *,
    samples: int = 1000,
    seed: int = 20261124,
) -> BootstrapResult:
    """Paired bootstrap resampling over per-segment scores.

    Two systems are run on the same segments, so the pairing is real and the test
    is far more sensitive than comparing two corpus-level numbers. The p-value is
    the share of resamples in which the system does not beat the baseline — a
    one-sided test of "is this actually better".
    """
    if len(baseline_scores) != len(system_scores):
        msg = "paired bootstrap needs the same number of scores for both systems"
        raise ValueError(msg)
    if not baseline_scores:
        msg = "paired bootstrap needs at least one score"
        raise ValueError(msg)

    rng = random.Random(seed)
    size = len(baseline_scores)
    observed = sum(system_scores) / size - sum(baseline_scores) / size

    differences: list[float] = []
    not_better = 0
    for _ in range(samples):
        picks = [rng.randrange(size) for _ in range(size)]
        diff = sum(system_scores[i] - baseline_scores[i] for i in picks) / size
        differences.append(diff)
        if diff <= 0:
            not_better += 1

    differences.sort()
    low = differences[int(0.025 * samples)]
    high = differences[min(samples - 1, int(0.975 * samples))]

    return BootstrapResult(
        mean_baseline=sum(baseline_scores) / size,
        mean_system=sum(system_scores) / size,
        mean_difference=observed,
        p_value=not_better / samples,
        ci_low=low,
        ci_high=high,
        samples=samples,
    )
