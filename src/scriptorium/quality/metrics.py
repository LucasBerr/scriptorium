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
from collections.abc import Sequence

from pydantic import BaseModel
from sacrebleu.metrics import BLEU, CHRF, TER

from scriptorium.domain import GlossaryTerm
from scriptorium.segmentation.placeholders import IntegrityReport
from scriptorium.terminology.enforcement import TerminologyViolation, applicable_terms, enforce
from scriptorium.terminology.store import RenderingLedger


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


class TerminologyReport(BaseModel):
    applicable: int = 0
    compliant: int = 0
    violations: tuple[TerminologyViolation, ...] = ()

    @property
    def rate(self) -> float:
        """Share of applicable term checks that passed. 1.0 when none apply."""
        return 1.0 if self.applicable == 0 else self.compliant / self.applicable


def terminology_compliance(
    sources: Sequence[str],
    hypotheses: Sequence[str],
    glossary: Sequence[GlossaryTerm],
) -> TerminologyReport:
    """Score the term base over a corpus.

    This calls the same :func:`scriptorium.terminology.enforce` the pipeline uses
    to decide whether to ask for a revision. One implementation, so the target
    the pipeline optimises against and the number the report prints cannot drift.

    A term is *applicable* to a segment when it occurs in that segment's source,
    and each (segment, term) pair is one check.
    """
    applicable = 0
    violations: list[TerminologyViolation] = []

    for index, (source, hypothesis) in enumerate(zip(sources, hypotheses, strict=True)):
        applicable += len(applicable_terms(source, glossary))
        for violation in enforce(source, hypothesis, glossary):
            violations.append(violation.model_copy(update={"index": index}))

    return TerminologyReport(
        applicable=applicable,
        compliant=applicable - len(violations),
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

    This is the page-12-versus-page-280 problem, and it is scored with the same
    :class:`RenderingLedger` the orchestrator consults while translating — so the
    metric measures the mechanism rather than a reimplementation of it.

    Limitation, inherited from the ledger: consistency is observed over the known
    forms in the term base. A rendering nobody anticipated is invisible, which is
    why mining feeds the term base and why word alignment is the phase-3 upgrade.
    """
    ledger = RenderingLedger(glossary)
    for source, hypothesis in zip(sources, hypotheses, strict=True):
        ledger.observe(source, hypothesis)

    observed = 0
    in_modal = 0
    per_term: dict[str, dict[str, int]] = {}
    for term in glossary:
        counts = ledger.renderings(term.source_term)
        if not counts:
            continue
        per_term[term.source_term] = counts
        observed += sum(counts.values())
        in_modal += max(counts.values())

    return ConsistencyReport(observed=observed, in_modal_form=in_modal, per_term=per_term)


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
