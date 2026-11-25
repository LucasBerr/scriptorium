"""The translate → review → revise loop.

The control flow is written out rather than delegated to a framework, because the
control flow *is* the contribution: which checks run before the model is asked
anything, what counts as a problem worth another pass, and when a segment stops
being the machine's business and becomes a person's.

Order matters, and it is ordered by cost:

1. **Memory first.** An approved exact hit returns immediately, with no model
   call at all. This is the largest cost lever in the pipeline, which is why it
   is the first thing that happens and not an optimisation bolted on later.
2. **Terminology and established renderings next.** Both are deterministic. The
   term base says what a term must be; the ledger says what it has already been
   in this document, and that is passed down as binding. This is how a pipeline
   that translates one segment at a time stays consistent across 300 pages.
3. **Then the model translates**, with retrieved examples and pinned terms.
4. **Then the cheap checks**, before the expensive one: placeholder integrity and
   terminology enforcement are free and mechanical, so they run first and their
   findings are handed to the reviewer rather than rediscovered by it.
5. **Then review and revision**, bounded by a budget so the loop cannot spin.
6. **Then routing.** A segment that still fails a check, or scores below the
   quality threshold, is flagged for a human instead of being shipped quietly.

A revision budget that is exhausted is not a failure to hide: the outcome says
so, and the segment is routed to a person.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from scriptorium.audit.log import AuditLog, AuditRecord
from scriptorium.domain import (
    GlossaryTerm,
    MQMSeverity,
    Review,
    ReviewVerdict,
    TMEntry,
    TokenUsage,
    Translation,
    TranslationStatus,
)
from scriptorium.memory.store import TranslationMemory
from scriptorium.orchestrator.tools import Toolbox, ToolCall
from scriptorium.providers.base import (
    AgenticProvider,
    ProviderResult,
    ReviewRequest,
    RevisionRequest,
    TranslationRequest,
)
from scriptorium.quality.estimation import QualityEstimate, estimate_quality
from scriptorium.segmentation.placeholders import IntegrityReport, check_tokens, tokens_in
from scriptorium.terminology.enforcement import TerminologyViolation
from scriptorium.terminology.store import RenderingLedger, TerminologyStore

# Neutral annotations are the reviewer's preferences, not errors, and acting on
# them would burn a revision on taste.
ACTIONABLE_SEVERITIES = frozenset({MQMSeverity.MINOR, MQMSeverity.MAJOR, MQMSeverity.CRITICAL})


class LoopOutcome(BaseModel):
    """Everything that happened to one segment, and why it ended where it did."""

    segment_id: str
    text: str
    status: TranslationStatus
    quality: QualityEstimate
    integrity: IntegrityReport
    violations: tuple[TerminologyViolation, ...] = ()
    problems: tuple[str, ...] = ()
    review: Review | None = None
    revisions: int = Field(default=0, ge=0)
    from_memory: bool = False
    budget_exhausted: bool = False
    tool_calls: tuple[ToolCall, ...] = ()
    result: ProviderResult

    @property
    def needs_human(self) -> bool:
        return self.status is TranslationStatus.NEEDS_HUMAN


def _merge(total: ProviderResult, part: ProviderResult) -> ProviderResult:
    """Accumulate metering across the calls a single segment required.

    Translation, review and every revision are all billed, so a segment's cost is
    their sum. Reporting only the first call would make the cost column of arm B3
    understate the pipeline by exactly the amount the agentic part adds.
    """
    return ProviderResult(
        text=part.text or total.text,
        model=part.model or total.model,
        usage=TokenUsage(
            input_tokens=total.usage.input_tokens + part.usage.input_tokens,
            cached_input_tokens=total.usage.cached_input_tokens + part.usage.cached_input_tokens,
            output_tokens=total.usage.output_tokens + part.usage.output_tokens,
        ),
        prompt_hash=part.prompt_hash or total.prompt_hash,
        effort=part.effort or total.effort,
        latency_ms=total.latency_ms + part.latency_ms,
        response_id=part.response_id or total.response_id,
        cost_usd=total.cost_usd + part.cost_usd,
    )


class TranslationLoop:
    """Orchestrates one segment at a time, sharing memory and ledger across them."""

    name = "loop"

    def __init__(
        self,
        provider: AgenticProvider,
        *,
        source_lang: str,
        target_lang: str,
        memory: TranslationMemory | None = None,
        terminology: TerminologyStore | None = None,
        max_revisions: int = 2,
        quality_threshold: float = 0.7,
        examples_k: int = 3,
        style_guide: str = "",
        audit: AuditLog | None = None,
    ) -> None:
        self.provider = provider
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.terminology = terminology if terminology is not None else TerminologyStore()
        self.memory = memory if memory is not None else TranslationMemory()
        self.ledger = RenderingLedger(self.terminology.terms)
        self.max_revisions = max_revisions
        self.quality_threshold = quality_threshold
        self.examples_k = examples_k
        self.style_guide = style_guide
        self.audit = audit

        self.toolbox = Toolbox(
            source_lang=source_lang,
            target_lang=target_lang,
            memory=self.memory,
            terminology=self.terminology,
            ledger=self.ledger,
        )

    @property
    def model(self) -> str:
        return str(getattr(self.provider, "model", "unknown"))

    # --------------------------------------------------------------- helpers

    def _findings(
        self,
        integrity: IntegrityReport,
        violations: Sequence[TerminologyViolation],
    ) -> list[str]:
        """Turn mechanical check results into instructions a model can act on."""
        findings: list[str] = []
        if integrity.missing:
            findings.append(
                f"placeholders dropped and must be restored: {', '.join(integrity.missing)}"
            )
        if integrity.duplicated:
            findings.append(
                f"placeholders repeated and must appear once: {', '.join(integrity.duplicated)}"
            )
        if integrity.unknown:
            findings.append(
                f"placeholders invented and must be removed: {', '.join(integrity.unknown)}"
            )
        findings.extend(
            f"terminology: {violation.source_term!r} must be rendered as "
            f"{violation.expected!r} ({violation.detail})"
            for violation in violations
        )
        return findings

    def _pinned_glossary(self, source: str) -> tuple[list[GlossaryTerm], list[tuple[str, str]]]:
        """Applicable terms, with any already-established rendering pinned.

        Where a term has been rendered one way earlier in this document, that
        form replaces the term base's target form for this segment. The term base
        allows a set of acceptable forms; the document gets to have only one.
        """
        applicable = self.toolbox.call("lookup_terminology", {"text": source})["terms"]
        pinned: list[GlossaryTerm] = []
        established: list[tuple[str, str]] = []

        for entry in applicable:
            term = self.terminology.get(str(entry["source_term"]))
            if term is None:
                continue
            consistency = self.toolbox.call("check_consistency", {"source_term": term.source_term})
            previous = consistency["established"]
            if previous and not term.do_not_translate and previous != term.target_term:
                established.append((term.source_term, str(previous)))
                pinned.append(term.model_copy(update={"target_term": str(previous)}))
            else:
                pinned.append(term)
        return pinned, established

    def _memory_hit(self, source: str) -> tuple[str | None, list[TMEntry]]:
        """An approved exact translation if there is one, plus few-shot examples."""
        payload = self.toolbox.call(
            "search_translation_memory", {"text": source, "k": self.examples_k}
        )
        hits = payload["hits"]
        if hits and hits[0]["kind"] == "exact":
            return str(hits[0]["target"]), []

        examples = [
            TMEntry(
                source_text=str(hit["source"]),
                target_text=str(hit["target"]),
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                provenance="translation memory",
                approved_by=hit["approved_by"],
            )
            for hit in hits
        ]
        return None, examples

    # ------------------------------------------------------------------- run

    def run(self, request: TranslationRequest) -> LoopOutcome:
        """Translate one segment, with checks, review and routing."""
        segment = request.segment
        expected_tokens = tokens_in(segment.text)

        reused, examples = self._memory_hit(segment.text)
        if reused is not None:
            self.ledger.observe(segment.text, reused)
            integrity = check_tokens(expected_tokens, reused)
            return LoopOutcome(
                segment_id=segment.id,
                text=reused,
                status=TranslationStatus.HUMAN_APPROVED,
                quality=estimate_quality(segment.text, reused, glossary=self.terminology.terms),
                integrity=integrity,
                from_memory=True,
                tool_calls=self.toolbox.drain(),
                result=ProviderResult(text=reused, model="translation-memory", cost_usd=0.0),
            )

        pinned, established = self._pinned_glossary(segment.text)

        enriched = request.model_copy(
            update={
                "glossary": tuple(pinned),
                "examples": tuple(examples),
                "style_guide": request.style_guide or self.style_guide,
            }
        )
        total = self.provider.translate(enriched)
        draft = total.text

        review: Review | None = None
        revisions = 0
        problems: list[str] = []
        integrity = check_tokens(expected_tokens, draft)
        violations: list[TerminologyViolation] = []

        while True:
            integrity = check_tokens(expected_tokens, draft)
            violations = self.terminology.enforce(segment.text, draft)
            problems = self._findings(integrity, violations)

            review_result = self.provider.review(
                ReviewRequest(
                    segment=segment,
                    draft=draft,
                    source_lang=self.source_lang,
                    target_lang=self.target_lang,
                    glossary=tuple(pinned),
                    automatic_findings=tuple(problems),
                )
            )
            review = review_result.review
            total = _merge(total, review_result.meta.model_copy(update={"text": ""}))

            problems.extend(
                f"{annotation.category.value} ({annotation.severity.value}): "
                f"{annotation.comment} [span: {annotation.span!r}]"
                for annotation in review.annotations
                if annotation.severity in ACTIONABLE_SEVERITIES
            )

            if not problems or revisions >= self.max_revisions:
                break

            revised = self.provider.revise(
                RevisionRequest(
                    segment=segment,
                    draft=draft,
                    source_lang=self.source_lang,
                    target_lang=self.target_lang,
                    problems=tuple(problems),
                    glossary=tuple(pinned),
                    established_renderings=tuple(established),
                )
            )
            total = _merge(total, revised)
            draft = revised.text
            revisions += 1

        quality = estimate_quality(segment.text, draft, glossary=self.terminology.terms)
        budget_exhausted = bool(problems) and revisions >= self.max_revisions

        unresolved = bool(violations) or not integrity.is_intact
        below_bar = quality.score < self.quality_threshold
        if unresolved or below_bar or budget_exhausted:
            status = TranslationStatus.NEEDS_HUMAN
            self.toolbox.call(
                "flag_for_human",
                {
                    "reason": "; ".join(problems)
                    or f"quality estimate {quality.score:.2f} below "
                    f"threshold {self.quality_threshold:.2f}"
                },
            )
        elif review is not None and review.verdict is ReviewVerdict.ACCEPT:
            status = TranslationStatus.REVIEWED
        else:
            status = TranslationStatus.MACHINE

        self.ledger.observe(segment.text, draft)
        total = total.model_copy(update={"text": draft})

        if self.audit is not None:
            self.audit.append(
                AuditRecord.from_translation(
                    Translation(
                        segment_id=segment.id,
                        target_lang=self.target_lang,
                        text=draft,
                        provider=str(getattr(self.provider, "name", "unknown")),
                        model=total.model,
                        prompt_hash=total.prompt_hash,
                        effort=total.effort,
                        usage=total.usage,
                        cost_usd=total.cost_usd,
                        latency_ms=total.latency_ms,
                        status=status,
                        qe_score=quality.score,
                    ),
                    segment_hash=segment.source_hash,
                )
            )

        return LoopOutcome(
            segment_id=segment.id,
            text=draft,
            status=status,
            quality=quality,
            integrity=integrity,
            violations=tuple(violations),
            problems=tuple(problems),
            review=review,
            revisions=revisions,
            budget_exhausted=budget_exhausted,
            tool_calls=self.toolbox.drain(),
            result=total,
        )

    # ------------------------------------------- harness-compatible interface

    def translate(self, request: TranslationRequest) -> ProviderResult:
        return self.run(request).result

    def translate_batch(self, requests: Sequence[TranslationRequest]) -> list[ProviderResult]:
        """Run the loop over a batch, so the loop is an evaluation arm like any other.

        Order matters here in a way it does not for a plain provider: the ledger
        and the memory are built up as the batch proceeds, which is what makes
        arm B3 measure document-level consistency rather than segment quality in
        isolation.
        """
        return [self.run(request).result for request in requests]
