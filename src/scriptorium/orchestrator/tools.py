"""The tools the orchestrator can call.

Six tools, written for this pipeline rather than taken from a framework. Each one
has a Pydantic input model, so its JSON Schema is generated rather than
hand-maintained and arguments are validated before a handler runs. A tool whose
schema and whose implementation can disagree is a tool that fails in production
and passes in review.

They are exposed two ways on purpose:

- the orchestrator calls them directly, as ordinary Python, which is what makes
  the loop deterministic and testable without a model in the way;
- :meth:`Toolbox.definitions` emits provider-ready tool definitions, so the same
  six can be handed to a model for function calling without reimplementation.

Every call is recorded. The record is what makes an agent's behaviour reviewable
after the fact instead of a matter of trust.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from scriptorium.domain import GlossaryTerm
from scriptorium.memory.store import TranslationMemory
from scriptorium.providers.schema import strict_json_schema
from scriptorium.quality.estimation import estimate_quality
from scriptorium.terminology.store import RenderingLedger, TerminologyStore


class ToolDefinition(BaseModel):
    """A tool as a provider's function-calling API wants to see it."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    """One recorded invocation."""

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class UnknownToolError(KeyError):
    """Raised when a tool name is not registered."""


class ToolInputError(ValueError):
    """Raised when arguments do not match the tool's schema."""


# --- tool inputs -------------------------------------------------------------
# No defaults: strict schema mode requires every property to be required, and a
# default that only exists in Python would be a schema the provider cannot
# enforce.


class LookupTerminologyInput(BaseModel):
    text: str = Field(description="Placeholder-protected source text of the segment")


class SearchMemoryInput(BaseModel):
    text: str = Field(description="Placeholder-protected source text to look up")
    k: int = Field(ge=1, le=10, description="How many entries to return")


class CheckConsistencyInput(BaseModel):
    source_term: str = Field(description="The source-language term to check")


class EstimateQualityInput(BaseModel):
    source: str = Field(description="Placeholder-protected source text")
    translation: str = Field(description="Candidate translation, placeholders intact")


class ProposeGlossaryEntryInput(BaseModel):
    source_term: str
    target_term: str
    rationale: str = Field(description="Why this belongs in the term base")


class FlagForHumanInput(BaseModel):
    reason: str = Field(description="What a human needs to decide")


class GlossaryProposal(BaseModel):
    """A proposed term, awaiting human approval. Never auto-applied."""

    source_term: str
    target_term: str
    rationale: str


class HumanFlag(BaseModel):
    reason: str


class Toolbox:
    """The tools, their schemas, and a record of every call made."""

    def __init__(
        self,
        *,
        source_lang: str,
        target_lang: str,
        memory: TranslationMemory | None = None,
        terminology: TerminologyStore | None = None,
        ledger: RenderingLedger | None = None,
    ) -> None:
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.memory = memory if memory is not None else TranslationMemory()
        self.terminology = terminology if terminology is not None else TerminologyStore()
        self.ledger = ledger if ledger is not None else RenderingLedger()

        self.proposals: list[GlossaryProposal] = []
        self.flags: list[HumanFlag] = []
        self._calls: list[ToolCall] = []

        self._registry: dict[str, tuple[type[BaseModel], Callable[[Any], dict[str, Any]], str]] = {
            "lookup_terminology": (
                LookupTerminologyInput,
                self._lookup_terminology,
                "Return the binding terms that occur in this segment's source text. "
                "Call this before translating; the terms are not suggestions.",
            ),
            "search_translation_memory": (
                SearchMemoryInput,
                self._search_translation_memory,
                "Find approved translations of the same or similar source text. An "
                "exact hit should be reused verbatim rather than retranslated.",
            ),
            "check_consistency": (
                CheckConsistencyInput,
                self._check_consistency,
                "Report how a term has already been rendered earlier in this document, "
                "so a later segment can agree with an earlier one.",
            ),
            "estimate_quality": (
                EstimateQualityInput,
                self._estimate_quality,
                "Score a candidate translation and return the signals behind the score. "
                "Heuristic, not a trained model.",
            ),
            "propose_glossary_entry": (
                ProposeGlossaryEntryInput,
                self._propose_glossary_entry,
                "Propose a term for the term base. The proposal is recorded for human "
                "approval and does not take effect now.",
            ),
            "flag_for_human": (
                FlagForHumanInput,
                self._flag_for_human,
                "Mark this segment as needing a human decision, with the reason.",
            ),
        }

    # ------------------------------------------------------------- dispatch

    def definitions(self) -> list[ToolDefinition]:
        """Provider-ready definitions for all six tools."""
        return [
            ToolDefinition(
                name=name,
                description=description,
                input_schema=strict_json_schema(model),
            )
            for name, (model, _handler, description) in self._registry.items()
        ]

    def call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Validate arguments, run the tool, record the call."""
        try:
            model, handler, _description = self._registry[name]
        except KeyError as exc:
            msg = f"unknown tool {name!r}; known tools: {sorted(self._registry)}"
            raise UnknownToolError(msg) from exc

        try:
            payload = model.model_validate(dict(arguments))
        except ValidationError as exc:
            msg = f"invalid arguments for {name!r}: {exc}"
            raise ToolInputError(msg) from exc

        result = handler(payload)
        self._calls.append(ToolCall(name=name, arguments=dict(arguments), result=result))
        return result

    @property
    def calls(self) -> tuple[ToolCall, ...]:
        return tuple(self._calls)

    def drain(self) -> tuple[ToolCall, ...]:
        """Take the calls recorded so far and clear the log.

        The orchestrator drains once per segment, so each segment's outcome
        carries exactly the calls that produced it.
        """
        calls = tuple(self._calls)
        self._calls.clear()
        return calls

    # ------------------------------------------------------------- handlers

    def _lookup_terminology(self, payload: LookupTerminologyInput) -> dict[str, Any]:
        terms = self.terminology.applicable(payload.text)
        return {
            "terms": [
                {
                    "source_term": term.source_term,
                    "target_term": term.target_term,
                    "do_not_translate": term.do_not_translate,
                    "forbidden_variants": list(term.forbidden_variants),
                    "notes": term.notes,
                }
                for term in terms
            ]
        }

    def _search_translation_memory(self, payload: SearchMemoryInput) -> dict[str, Any]:
        hits = self.memory.search(
            payload.text,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            k=payload.k,
        )
        return {
            "hits": [
                {
                    "source": hit.entry.source_text,
                    "target": hit.entry.target_text,
                    "score": round(hit.score, 4),
                    "kind": hit.kind,
                    "approved_by": hit.entry.approved_by,
                }
                for hit in hits
            ]
        }

    def _check_consistency(self, payload: CheckConsistencyInput) -> dict[str, Any]:
        return {
            "established": self.ledger.established(payload.source_term),
            "renderings": self.ledger.renderings(payload.source_term),
        }

    def _estimate_quality(self, payload: EstimateQualityInput) -> dict[str, Any]:
        estimate = estimate_quality(
            payload.source,
            payload.translation,
            glossary=self.terminology.terms,
        )
        return {
            "score": round(estimate.score, 4),
            "heuristic": estimate.heuristic,
            "signals": estimate.signals.model_dump(),
        }

    def _propose_glossary_entry(self, payload: ProposeGlossaryEntryInput) -> dict[str, Any]:
        proposal = GlossaryProposal(
            source_term=payload.source_term,
            target_term=payload.target_term,
            rationale=payload.rationale,
        )
        self.proposals.append(proposal)
        return {"status": "recorded_for_approval", "pending_proposals": len(self.proposals)}

    def _flag_for_human(self, payload: FlagForHumanInput) -> dict[str, Any]:
        self.flags.append(HumanFlag(reason=payload.reason))
        return {"status": "flagged", "flags": len(self.flags)}

    def approve_proposal(self, index: int, *, approved_by: str) -> GlossaryTerm:
        """Turn a recorded proposal into a real term. Only a human calls this.

        There is no tool for this, deliberately: the model may propose, and a
        person disposes. Making approval reachable from the agent loop would
        remove the one checkpoint the term base has.
        """
        proposal = self.proposals.pop(index)
        term = GlossaryTerm(
            source_term=proposal.source_term,
            target_term=proposal.target_term,
            notes=f"{proposal.rationale} (approved by {approved_by})",
        )
        self.terminology.add(term)
        return term
