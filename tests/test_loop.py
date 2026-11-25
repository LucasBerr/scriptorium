"""The agentic loop: what it spends, what it fixes, and when it gives up.

Every test here uses a provider that makes no network call, because the point is
the control flow, not the model. The loop's claims are testable precisely
because the decisions are made in code rather than asked for in a prompt.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from scriptorium.audit.log import AuditLog
from scriptorium.domain import (
    GlossaryTerm,
    MQMAnnotation,
    MQMCategory,
    MQMSeverity,
    Review,
    ReviewVerdict,
    Segment,
    StructureKind,
    TMEntry,
    TokenUsage,
    TranslationStatus,
)
from scriptorium.memory import TranslationMemory
from scriptorium.orchestrator import TranslationLoop
from scriptorium.providers.base import (
    ProviderResult,
    ReviewRequest,
    ReviewResult,
    RevisionRequest,
    TranslationRequest,
)
from scriptorium.providers.fake import EchoProvider, ScriptedProvider
from scriptorium.terminology import TerminologyStore

SOURCE = "Press the button."
GOOD = "Pressione o botão."
BAD = "Pressione a tecla."


def request(text: str, index: int = 0) -> TranslationRequest:
    return TranslationRequest(
        segment=Segment(
            id=f"seg:{index:04d}",
            document_id="doc",
            block_key=f"doc#{index}",
            index=index,
            text=text,
            structure=StructureKind.PARAGRAPH,
            span_start=0,
            span_end=len(text),
        ),
        source_lang="en",
        target_lang="pt-BR",
    )


class RecordingProvider:
    """A scripted provider that keeps the requests it was handed.

    Needed because some of the loop's work is not visible in its output: what it
    pins into the prompt is a decision worth asserting on directly.
    """

    name = "recording"

    def __init__(self, inner: ScriptedProvider) -> None:
        self.inner = inner
        self.model = inner.model
        self.translations: list[TranslationRequest] = []
        self.reviews: list[ReviewRequest] = []
        self.revisions: list[RevisionRequest] = []

    def translate(self, req: TranslationRequest) -> ProviderResult:
        self.translations.append(req)
        return self.inner.translate(req)

    def translate_batch(self, reqs: Sequence[TranslationRequest]) -> list[ProviderResult]:
        return [self.translate(req) for req in reqs]

    def review(self, req: ReviewRequest) -> ReviewResult:
        self.reviews.append(req)
        return self.inner.review(req)

    def revise(self, req: RevisionRequest) -> ProviderResult:
        self.revisions.append(req)
        return self.inner.revise(req)


class CostedProvider:
    """Charges a fixed, distinguishable amount per role.

    Distinguishable on purpose: a single total cannot show whether review and
    revision were billed, and understating the agentic part of the bill is the
    specific way a cost column turns into fiction.
    """

    name = "costed"
    model = "costed-v1"

    TRANSLATE = 0.01
    REVIEW = 0.002
    REVISE = 0.005

    def _result(self, text: str, cost: float) -> ProviderResult:
        return ProviderResult(
            text=text,
            model=self.model,
            usage=TokenUsage(input_tokens=100, cached_input_tokens=40, output_tokens=20),
            cost_usd=cost,
            latency_ms=10.0,
        )

    def translate(self, req: TranslationRequest) -> ProviderResult:
        del req
        return self._result(BAD, self.TRANSLATE)

    def translate_batch(self, reqs: Sequence[TranslationRequest]) -> list[ProviderResult]:
        return [self.translate(req) for req in reqs]

    def review(self, req: ReviewRequest) -> ReviewResult:
        del req
        return ReviewResult(
            review=Review(verdict=ReviewVerdict.REVISE),
            meta=self._result("", self.REVIEW),
        )

    def revise(self, req: RevisionRequest) -> ProviderResult:
        del req
        return self._result(GOOD, self.REVISE)


class ForbiddenProvider:
    """Fails loudly if asked to translate. Proves a path made no model call."""

    name = "forbidden"
    model = "none"

    def translate(self, req: TranslationRequest) -> ProviderResult:
        msg = f"the loop called the model for {req.segment.id}, which should not happen"
        raise AssertionError(msg)

    def translate_batch(self, reqs: Sequence[TranslationRequest]) -> list[ProviderResult]:
        return [self.translate(req) for req in reqs]

    def review(self, req: ReviewRequest) -> ReviewResult:
        raise AssertionError(self.translate.__doc__ or "unexpected review call")

    def revise(self, req: RevisionRequest) -> ProviderResult:
        raise AssertionError("unexpected revision call")


@pytest.fixture
def store(glossary: list[GlossaryTerm]) -> TerminologyStore:
    return TerminologyStore(glossary)


# ------------------------------------------------------------ memory first


def test_an_approved_exact_hit_skips_the_model_entirely(store: TerminologyStore) -> None:
    """The pipeline's largest cost lever, so it is the first thing asserted."""
    memory = TranslationMemory()
    memory.add(
        TMEntry(
            source_text=SOURCE,
            target_text=GOOD,
            source_lang="en",
            target_lang="pt-BR",
            provenance="seed",
            approved_by="reviewer",
        )
    )
    loop = TranslationLoop(
        ForbiddenProvider(),
        source_lang="en",
        target_lang="pt-BR",
        memory=memory,
        terminology=store,
    )

    outcome = loop.run(request(SOURCE))

    assert outcome.text == GOOD
    assert outcome.from_memory is True
    assert outcome.status is TranslationStatus.HUMAN_APPROVED
    assert outcome.result.cost_usd == 0.0
    assert outcome.result.usage.input_tokens == 0
    assert outcome.revisions == 0


def test_an_unapproved_entry_is_invisible_to_the_loop(store: TerminologyStore) -> None:
    """Only approved pairs reach the model, as an answer or as an example.

    An unapproved row is usually this pipeline's own earlier output. Feeding it
    back as a few-shot example would teach the model its own guesses and make the
    memory a self-reinforcing loop, so retrieval filters on provenance rather
    than on similarity alone.
    """
    memory = TranslationMemory()
    memory.add(
        TMEntry(
            source_text=SOURCE,
            target_text=GOOD,
            source_lang="en",
            target_lang="pt-BR",
            provenance="machine",
        )
    )
    provider = RecordingProvider(ScriptedProvider({SOURCE: GOOD}))
    loop = TranslationLoop(
        provider,
        source_lang="en",
        target_lang="pt-BR",
        memory=memory,
        terminology=store,
    )

    outcome = loop.run(request(SOURCE))

    assert outcome.from_memory is False
    assert len(provider.translations) == 1
    assert provider.translations[0].examples == ()


def test_an_approved_near_miss_becomes_a_few_shot_example(store: TerminologyStore) -> None:
    """Not the same segment, so not reusable — but worth paying attention to."""
    memory = TranslationMemory()
    memory.add(
        TMEntry(
            source_text="Press the green button.",
            target_text="Pressione o botão verde.",
            source_lang="en",
            target_lang="pt-BR",
            provenance="seed",
            approved_by="reviewer",
        )
    )
    provider = RecordingProvider(ScriptedProvider({SOURCE: GOOD}))
    loop = TranslationLoop(
        provider,
        source_lang="en",
        target_lang="pt-BR",
        memory=memory,
        terminology=store,
    )

    outcome = loop.run(request(SOURCE))

    assert outcome.from_memory is False
    examples = provider.translations[0].examples
    assert [e.target_text for e in examples] == ["Pressione o botão verde."]
    assert all(e.is_approved for e in examples)


# -------------------------------------------------- checks drive revision


def test_a_terminology_violation_is_fixed_by_a_revision(store: TerminologyStore) -> None:
    provider = RecordingProvider(
        ScriptedProvider({SOURCE: BAD}, revisions={BAD: GOOD}),
    )
    loop = TranslationLoop(provider, source_lang="en", target_lang="pt-BR", terminology=store)

    outcome = loop.run(request(SOURCE))

    assert outcome.text == GOOD
    assert outcome.revisions == 1
    assert outcome.violations == ()
    assert outcome.status is TranslationStatus.REVIEWED
    assert outcome.budget_exhausted is False


def test_the_mechanical_findings_reach_the_reviewer(store: TerminologyStore) -> None:
    """The reviewer is told what the free checks already found.

    Otherwise the model is paid to rediscover something a regex knows, and may
    disagree with it.
    """
    provider = RecordingProvider(ScriptedProvider({SOURCE: BAD}, revisions={BAD: GOOD}))
    loop = TranslationLoop(provider, source_lang="en", target_lang="pt-BR", terminology=store)
    loop.run(request(SOURCE))

    findings = provider.reviews[0].automatic_findings
    assert any("terminology" in finding and "botão" in finding for finding in findings)


def test_dropped_placeholders_are_reported_as_a_finding(store: TerminologyStore) -> None:
    source = "Delete ⟦PH0⟧ files."
    provider = RecordingProvider(ScriptedProvider({source: "Excluir arquivos."}))
    loop = TranslationLoop(provider, source_lang="en", target_lang="pt-BR", terminology=store)

    outcome = loop.run(request(source))

    assert outcome.integrity.missing == ("⟦PH0⟧",)
    assert outcome.status is TranslationStatus.NEEDS_HUMAN
    findings = provider.reviews[0].automatic_findings
    assert any("placeholders dropped" in finding for finding in findings)


def test_a_neutral_annotation_does_not_buy_a_revision(store: TerminologyStore) -> None:
    """Neutral is the reviewer's taste, not an error. Acting on it burns a pass."""
    review = Review(
        verdict=ReviewVerdict.ACCEPT,
        annotations=(
            MQMAnnotation(
                category=MQMCategory.REGISTER,
                severity=MQMSeverity.NEUTRAL,
                span="Pressione",
                comment="could be more formal",
            ),
        ),
    )
    provider = RecordingProvider(ScriptedProvider({SOURCE: GOOD}, reviews={GOOD: review}))
    loop = TranslationLoop(provider, source_lang="en", target_lang="pt-BR", terminology=store)

    outcome = loop.run(request(SOURCE))

    assert outcome.revisions == 0
    assert provider.revisions == []
    assert outcome.status is TranslationStatus.REVIEWED


def test_a_minor_annotation_does_buy_one(store: TerminologyStore) -> None:
    review = Review(
        verdict=ReviewVerdict.REVISE,
        annotations=(
            MQMAnnotation(
                category=MQMCategory.GRAMMAR,
                severity=MQMSeverity.MINOR,
                span="Pressione",
                comment="agreement",
            ),
        ),
    )
    other = "Aperte o botão."
    provider = RecordingProvider(
        ScriptedProvider({SOURCE: GOOD}, reviews={GOOD: review}, revisions={GOOD: other}),
    )
    loop = TranslationLoop(
        provider,
        source_lang="en",
        target_lang="pt-BR",
        terminology=store,
        max_revisions=1,
    )

    outcome = loop.run(request(SOURCE))

    assert outcome.revisions == 1
    assert outcome.text == other


# ------------------------------------------------------------- the budget


def test_a_stubborn_provider_exhausts_the_budget_and_routes_to_a_human(
    store: TerminologyStore,
) -> None:
    """The loop must stop, and say why it stopped."""
    loop = TranslationLoop(
        EchoProvider(),
        source_lang="en",
        target_lang="pt-BR",
        terminology=store,
        max_revisions=2,
    )

    outcome = loop.run(request(SOURCE))

    assert outcome.revisions == 2
    assert outcome.budget_exhausted is True
    assert outcome.status is TranslationStatus.NEEDS_HUMAN
    assert outcome.needs_human is True


def test_the_reason_for_routing_is_recorded_as_a_flag(store: TerminologyStore) -> None:
    loop = TranslationLoop(EchoProvider(), source_lang="en", target_lang="pt-BR", terminology=store)
    loop.run(request(SOURCE))

    assert len(loop.toolbox.flags) == 1
    assert "botão" in loop.toolbox.flags[0].reason


def test_a_low_quality_estimate_routes_to_a_human_even_with_no_violation() -> None:
    """An empty translation breaks no term and drops no placeholder."""
    provider = ScriptedProvider({SOURCE: "   "})
    loop = TranslationLoop(
        provider,
        source_lang="en",
        target_lang="pt-BR",
        quality_threshold=0.7,
    )

    outcome = loop.run(request(SOURCE))

    assert outcome.quality.signals.is_empty is True
    assert outcome.status is TranslationStatus.NEEDS_HUMAN


# ------------------------------------------------------------------- cost


def test_cost_is_summed_across_translation_review_and_revision() -> None:
    """One segment, three billed calls. The outcome must report all three."""
    loop = TranslationLoop(
        CostedProvider(),
        source_lang="en",
        target_lang="pt-BR",
        terminology=TerminologyStore([GlossaryTerm(source_term="button", target_term="botão")]),
        max_revisions=1,
    )

    outcome = loop.run(request(SOURCE))

    # translate + review + revise + the second review that follows the revision.
    expected = CostedProvider.TRANSLATE + CostedProvider.REVIEW * 2 + CostedProvider.REVISE
    assert outcome.result.cost_usd == pytest.approx(expected)
    assert outcome.result.usage.input_tokens == 400
    assert outcome.result.usage.cached_input_tokens == 160
    assert outcome.result.latency_ms == pytest.approx(40.0)
    assert outcome.result.text == GOOD


# ----------------------------------------------- consistency across segments


def test_the_established_rendering_is_pinned_into_the_next_prompt(
    store: TerminologyStore,
) -> None:
    """Where the term base allows two forms, the document gets only the first one.

    Enforcement cannot catch this: both "e-mail" and "email" are accepted. So the
    mechanism is the prompt, and what the prompt was told is the thing to assert.
    """
    memory = TranslationMemory()
    memory.add(
        TMEntry(
            source_text="Check your email.",
            target_text="Confira seu email.",
            source_lang="en",
            target_lang="pt-BR",
            provenance="seed",
            approved_by="reviewer",
        )
    )
    provider = RecordingProvider(ScriptedProvider({"Send an email.": "Envie um email."}))
    loop = TranslationLoop(
        provider,
        source_lang="en",
        target_lang="pt-BR",
        memory=memory,
        terminology=store,
    )

    first = loop.run(request("Check your email.", index=0))
    assert first.from_memory is True
    assert loop.ledger.established("email") == "email"

    loop.run(request("Send an email.", index=1))

    pinned = {term.source_term: term.target_term for term in provider.translations[0].glossary}
    assert pinned["email"] == "email", "the document's own form must override the term base"


def test_a_batch_builds_the_ledger_as_it_goes(store: TerminologyStore) -> None:
    provider = ScriptedProvider(
        {
            "Check your email.": "Confira seu email.",
            "Send an email.": "Envie um email.",
        }
    )
    loop = TranslationLoop(provider, source_lang="en", target_lang="pt-BR", terminology=store)

    results = loop.translate_batch([request("Check your email.", 0), request("Send an email.", 1)])

    assert [r.text for r in results] == ["Confira seu email.", "Envie um email."]
    assert loop.ledger.renderings("email") == {"email": 2}
    assert loop.ledger.disagreements() == {}


# ------------------------------------------------------------- transparency


def test_the_tool_calls_are_attached_to_the_outcome(store: TerminologyStore) -> None:
    provider = ScriptedProvider({SOURCE: GOOD})
    loop = TranslationLoop(provider, source_lang="en", target_lang="pt-BR", terminology=store)

    outcome = loop.run(request(SOURCE))

    assert [call.name for call in outcome.tool_calls][:2] == [
        "search_translation_memory",
        "lookup_terminology",
    ]


def test_each_segment_gets_its_own_tool_calls(store: TerminologyStore) -> None:
    """Draining per segment is what keeps the record attributable."""
    provider = ScriptedProvider({SOURCE: GOOD, "Open the folder.": "Abra a pasta."})
    loop = TranslationLoop(provider, source_lang="en", target_lang="pt-BR", terminology=store)

    first = loop.run(request(SOURCE, 0))
    second = loop.run(request("Open the folder.", 1))

    assert first.tool_calls
    assert second.tool_calls
    assert first.tool_calls[0].arguments["text"] == SOURCE
    assert second.tool_calls[0].arguments["text"] == "Open the folder."


def test_the_audit_log_records_the_segment_without_its_text(
    store: TerminologyStore, tmp_path: object
) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    log = AuditLog(tmp_path / "audit.jsonl")
    provider = ScriptedProvider({SOURCE: GOOD})
    loop = TranslationLoop(
        provider,
        source_lang="en",
        target_lang="pt-BR",
        terminology=store,
        audit=log,
    )

    outcome = loop.run(request(SOURCE))
    written = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")

    assert outcome.segment_id in written
    assert SOURCE not in written
    assert GOOD not in written
