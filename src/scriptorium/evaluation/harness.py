"""The evaluation harness.

An *arm* is a named system under test. The harness runs arms over the same test
cases, scores them with the same metrics, and reports quality and cost in one
table — because a quality number without its cost is not a result, and a cost
number without its quality is not either.

Arms differ only in what they put into the request. B1 sends the segment alone;
B2 adds the neighbouring text; B3 (phase 2) adds the glossary and retrieved
examples. That is the whole point of keeping context assembly separate from the
provider: an arm is a configuration, not a code path.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from scriptorium.domain import GlossaryTerm, Segment, StructureKind
from scriptorium.providers.base import LLMProvider, TranslationRequest
from scriptorium.quality import metrics
from scriptorium.segmentation.placeholders import check_integrity, protect, restore


class TestCase(BaseModel):
    """One scored item: a source segment and its reference translation."""

    source: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    structure: StructureKind = StructureKind.PARAGRAPH
    note: str = ""

    @property
    def word_count(self) -> int:
        return len(self.source.split())


def load_test_set(path: Path) -> list[TestCase]:
    """Read a JSONL test set. One object per line, `source` and `reference` required."""
    cases: list[TestCase] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cases.append(TestCase.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                msg = f"{path}:{line_number}: {exc}"
                raise ValueError(msg) from exc
    return cases


class ArmResult(BaseModel):
    """What one arm scored, and what it cost to find out."""

    arm: str
    provider: str
    model: str
    segments: int
    chrf: float
    bleu: float
    ter: float
    terminology_compliance: float
    intra_document_consistency: float
    placeholder_integrity: float
    cost_usd: float
    cost_per_1k_words: float
    cache_hit_rate: float
    sentence_scores: tuple[float, ...] = ()
    hypotheses: tuple[str, ...] = ()


class Harness:
    """Run arms over a test set and score them."""

    def __init__(
        self,
        cases: Sequence[TestCase],
        *,
        source_lang: str,
        target_lang: str,
        glossary: Sequence[GlossaryTerm] = (),
    ) -> None:
        if not cases:
            msg = "the harness needs at least one test case"
            raise ValueError(msg)
        self.cases = list(cases)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.glossary = list(glossary)

    def _segment(self, index: int, case: TestCase) -> Segment:
        protected = protect(case.source)
        return Segment(
            id=f"eval:{index:06d}",
            document_id="eval",
            block_key=f"eval#{index}",
            index=index,
            text=protected.text,
            structure=case.structure,
            span_start=0,
            span_end=len(protected.text),
        )

    def _request(self, index: int, case: TestCase, *, with_context: bool) -> TranslationRequest:
        context_before = self.cases[index - 1].source if with_context and index > 0 else ""
        context_after = (
            self.cases[index + 1].source if with_context and index + 1 < len(self.cases) else ""
        )
        return TranslationRequest(
            segment=self._segment(index, case),
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            glossary=tuple(self.glossary) if with_context else (),
            context_before=context_before,
            context_after=context_after,
        )

    def run_arm(
        self,
        arm: str,
        provider: LLMProvider,
        *,
        with_context: bool = False,
    ) -> ArmResult:
        """Run one arm and score it."""
        requests = [
            self._request(index, case, with_context=with_context)
            for index, case in enumerate(self.cases)
        ]
        results = provider.translate_batch(requests)

        # The provider works in protected space: what it returns still carries
        # placeholder tokens. Integrity is therefore checked on the raw response,
        # and placeholders are restored before scoring — otherwise every metric
        # would compare a token against the real `%d` in the reference and report
        # a quality collapse that exists only in the harness.
        protected = [protect(case.source) for case in self.cases]
        integrity = [
            check_integrity(p.placeholders, result.text)
            for p, result in zip(protected, results, strict=True)
        ]
        hypotheses = [
            restore(result.text, p.placeholders)
            for p, result in zip(protected, results, strict=True)
        ]

        references = [c.reference for c in self.cases]

        total_cost = sum(r.cost_usd for r in results)
        total_words = sum(c.word_count for c in self.cases)
        total_input = sum(r.usage.input_tokens for r in results)
        total_cached = sum(r.usage.cached_input_tokens for r in results)

        # Terminology and consistency are measured in *protected* space, where a
        # placeholder is an opaque token. Otherwise a variable named `{folder}`
        # counts as an occurrence of the term "folder" and the checker demands a
        # translation of something that must not be translated. String metrics,
        # by contrast, are computed on restored text, because that is what a
        # reader of the output document actually sees.
        protected_sources = [p.text for p in protected]
        protected_hypotheses = [result.text for result in results]

        terminology = metrics.terminology_compliance(
            protected_sources, protected_hypotheses, self.glossary
        )
        consistency = metrics.intra_document_consistency(
            protected_sources, protected_hypotheses, self.glossary
        )

        return ArmResult(
            arm=arm,
            provider=getattr(provider, "name", "unknown"),
            model=getattr(provider, "model", "unknown"),
            segments=len(self.cases),
            chrf=metrics.chrf(hypotheses, references),
            bleu=metrics.bleu(hypotheses, references),
            ter=metrics.ter(hypotheses, references),
            terminology_compliance=terminology.rate,
            intra_document_consistency=consistency.rate,
            placeholder_integrity=metrics.placeholder_integrity(integrity),
            cost_usd=total_cost,
            cost_per_1k_words=(total_cost / total_words * 1000) if total_words else 0.0,
            cache_hit_rate=(total_cached / total_input) if total_input else 0.0,
            sentence_scores=tuple(metrics.sentence_chrf(hypotheses, references)),
            hypotheses=tuple(hypotheses),
        )

    @staticmethod
    def compare(baseline: ArmResult, system: ArmResult) -> metrics.BootstrapResult:
        """Paired bootstrap on per-segment chrF++ between two arms."""
        return metrics.paired_bootstrap(baseline.sentence_scores, system.sentence_scores)


def render_markdown_table(results: Sequence[ArmResult]) -> str:
    """Render the arm comparison as a Markdown table.

    Cost sits in the same table as quality on purpose: it is the comparison the
    report is for.
    """
    header = (
        "| Arm | Provider | Segments | chrF++ | BLEU | TER | Term. | Consist. | "
        "Placeholders | Cost/1k words | Cache |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [
        f"| {r.arm} | {r.provider} | {r.segments} | {r.chrf:.2f} | {r.bleu:.2f} | "
        f"{r.ter:.2f} | {r.terminology_compliance:.1%} | "
        f"{r.intra_document_consistency:.1%} | {r.placeholder_integrity:.1%} | "
        f"${r.cost_per_1k_words:.4f} | {r.cache_hit_rate:.1%} |"
        for r in results
    ]
    return "\n".join([header, *rows])
