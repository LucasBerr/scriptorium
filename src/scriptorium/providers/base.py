"""The provider contract.

A provider does one thing: turn a request into a translation plus the metering
information needed to cost it. Prompt assembly, retrieval, review and routing all
happen above this line, so swapping providers cannot change them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from scriptorium.domain import GlossaryTerm, Review, Segment, TMEntry, TokenUsage


class OfflineError(RuntimeError):
    """Raised when a model call is attempted while offline mode is on.

    CI runs with `SCRIPTORIUM_OFFLINE=1`, so a code path that would spend money
    fails the build instead of quietly billing someone.
    """


class ProviderUnpinnedError(RuntimeError):
    """Raised when a billed call is attempted with placeholder model config."""


class TranslationRequest(BaseModel):
    """Everything a provider needs for one segment.

    How much of this the caller fills in is what distinguishes the evaluation
    arms: B1 sends the segment alone, B2 adds document context, B3 adds the
    glossary and retrieved examples.
    """

    segment: Segment
    source_lang: str
    target_lang: str
    style_guide: str = ""
    glossary: tuple[GlossaryTerm, ...] = ()
    examples: tuple[TMEntry, ...] = ()
    context_before: str = ""
    context_after: str = ""
    document_summary: str = ""


class ProviderResult(BaseModel):
    """A translation and what it cost to get."""

    text: str
    model: str
    usage: TokenUsage = TokenUsage()
    prompt_hash: str = ""
    effort: str | None = None
    latency_ms: int = 0
    response_id: str | None = None
    cost_usd: float = 0.0


@runtime_checkable
class LLMProvider(Protocol):
    """Translate segments. Implementations live beside this file."""

    name: str

    def translate(self, request: TranslationRequest) -> ProviderResult:
        """Translate one segment."""
        ...

    def translate_batch(self, requests: Sequence[TranslationRequest]) -> list[ProviderResult]:
        """Translate several segments.

        A real provider overrides this with its asynchronous batch endpoint,
        which is roughly half price and is how whole documents are translated.
        """
        ...


class TranslationPayload(BaseModel):
    """The structured output every provider must return.

    Asking for JSON against a schema, rather than parsing prose, is what makes
    the pipeline's failure mode a validation error instead of a silent
    mistranslation. The reviewer role uses the same discipline with `Review`.
    """

    translation: str = Field(description="The translated segment, placeholders intact")
    notes: str = Field(default="", max_length=200, description="Only if something was ambiguous")


class ReviewRequest(BaseModel):
    """What the reviewer role is given.

    The reviewer sees the automatic findings too. Hiding them would waste the
    cheaper, deterministic checks that already ran, and would let the model
    "accept" a segment a checker has already failed.
    """

    segment: Segment
    draft: str
    source_lang: str
    target_lang: str
    glossary: tuple[GlossaryTerm, ...] = ()
    automatic_findings: tuple[str, ...] = ()


class RevisionRequest(BaseModel):
    """A draft plus the problems that must be resolved.

    `established_renderings` is how document-wide consistency is enforced on a
    segment translated in isolation: the form a term already took earlier in this
    document is passed down as binding.
    """

    segment: Segment
    draft: str
    source_lang: str
    target_lang: str
    problems: tuple[str, ...]
    glossary: tuple[GlossaryTerm, ...] = ()
    established_renderings: tuple[tuple[str, str], ...] = ()


class ReviewResult(BaseModel):
    """A review and what the review itself cost.

    The metering travels with the review because review calls are part of the
    bill. Returning only the annotations would make arm B3's cost per 1,000
    words understate the pipeline by however much reviewing costs — and that
    column is one of the project's headline claims.
    """

    review: Review
    meta: ProviderResult


@runtime_checkable
class Reviewer(Protocol):
    """Annotate a translation against the MQM typology."""

    def review(self, request: ReviewRequest) -> ReviewResult: ...


@runtime_checkable
class Reviser(Protocol):
    """Correct a draft against a list of findings."""

    def revise(self, request: RevisionRequest) -> ProviderResult: ...


@runtime_checkable
class AgenticProvider(LLMProvider, Reviewer, Reviser, Protocol):
    """Everything :class:`scriptorium.orchestrator.TranslationLoop` needs.

    Python has no intersection type, so the three capabilities are composed into
    one protocol here rather than being spelled out at every call site.
    """
