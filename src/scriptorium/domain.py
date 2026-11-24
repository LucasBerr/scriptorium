"""Core domain types.

Everything that crosses a module boundary in this project is one of these models.
They are deliberately boring: validation lives here so that the pipeline stages
can assume well-formed input, and so that an invalid state is unrepresentable
rather than merely unlikely.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1"


def content_hash(text: str) -> str:
    """Stable short hash of a piece of text, used for caching and audit records.

    The audit log stores this instead of the text itself: the log is meant to be
    publishable, the corpus is not necessarily.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class DocumentFormat(StrEnum):
    EPUB = "epub"
    PDF = "pdf"
    PO = "po"
    JSON_I18N = "json_i18n"


class StructureKind(StrEnum):
    """What the segment was taken from. Affects register, not just formatting."""

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    QUOTE = "quote"
    CAPTION = "caption"
    TABLE_CELL = "table_cell"
    STRING_ENTRY = "string_entry"


class Placeholder(BaseModel):
    """A span of the source that must survive translation untouched.

    Inline markup, printf/ICU placeholders, HTML entities and shell-style
    variables all become placeholders before the text reaches a model.
    """

    model_config = ConfigDict(frozen=True)

    token: str = Field(description="The sentinel written into the text, e.g. PH0 wrapped in marks")
    kind: str = Field(description="Which pattern matched: tag, printf, brace, entity, ...")
    original: str = Field(description="The exact source text this token stands for")


class SourceDocument(BaseModel):
    """Provenance for an ingested document.

    Ingestion refuses a document that cannot fill in licence and source, which is
    why these fields are required rather than optional with a default.
    """

    id: str
    title: str
    source_uri: str = Field(min_length=1)
    licence: str = Field(min_length=1, description="SPDX identifier or 'public-domain'")
    licence_evidence: str = Field(min_length=1, description="URL or note supporting the claim")
    lang: str
    format: DocumentFormat


class Block(BaseModel):
    """A text-bearing element of the source document, before sentence splitting.

    `protected_text` is the element's inner content with placeholders substituted.
    Segments carry character spans into this string, so reassembly can stitch
    translations back with the original whitespace intact.
    """

    key: str = Field(description="Stable address of the block inside the document")
    item: str = Field(description="Container the block came from, e.g. an EPUB item href")
    index: int = Field(ge=0, description="Position of the block within its container")
    structure: StructureKind
    protected_text: str
    placeholders: tuple[Placeholder, ...] = ()


class Segment(BaseModel):
    """One unit of translation: normally a sentence."""

    id: str
    document_id: str
    block_key: str
    index: int = Field(ge=0, description="Position of the segment within the document")
    text: str = Field(min_length=1, description="Placeholder-protected source text")
    structure: StructureKind
    span_start: int = Field(ge=0, description="Start offset within the block's protected text")
    span_end: int = Field(ge=0)
    source_hash: str = ""

    @model_validator(mode="after")
    def _fill_hash_and_check_span(self) -> Segment:
        if self.span_end < self.span_start:
            msg = f"segment {self.id}: span_end {self.span_end} < span_start {self.span_start}"
            raise ValueError(msg)
        if not self.source_hash:
            object.__setattr__(self, "source_hash", content_hash(self.text))
        return self


class ExtractedDocument(BaseModel):
    """What an adapter returns: provenance, blocks, and segments in reading order."""

    document: SourceDocument
    blocks: tuple[Block, ...]
    segments: tuple[Segment, ...]

    def block(self, key: str) -> Block:
        for block in self.blocks:
            if block.key == key:
                return block
        msg = f"unknown block key: {key}"
        raise KeyError(msg)


class MQMCategory(StrEnum):
    """Error categories, following the MQM core typology.

    This is the vocabulary the industry uses for translation quality, which means
    an annotation here is comparable with a human annotation elsewhere.
    """

    TERMINOLOGY = "terminology"
    MISTRANSLATION = "accuracy/mistranslation"
    OMISSION = "accuracy/omission"
    ADDITION = "accuracy/addition"
    GRAMMAR = "fluency/grammar"
    SPELLING = "fluency/spelling"
    REGISTER = "style/register"
    LOCALE_CONVENTION = "locale/convention"
    MARKUP = "markup"
    OTHER = "other"


class MQMSeverity(StrEnum):
    NEUTRAL = "neutral"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


SEVERITY_WEIGHT: dict[MQMSeverity, float] = {
    MQMSeverity.NEUTRAL: 0.0,
    MQMSeverity.MINOR: 1.0,
    MQMSeverity.MAJOR: 5.0,
    MQMSeverity.CRITICAL: 10.0,
}


class MQMAnnotation(BaseModel):
    model_config = ConfigDict(frozen=True)

    category: MQMCategory
    severity: MQMSeverity
    span: str = Field(description="The exact substring of the translation at fault")
    comment: str = Field(max_length=300)


class ReviewVerdict(StrEnum):
    ACCEPT = "accept"
    REVISE = "revise"


class Review(BaseModel):
    """Structured output of the reviewer role. Never prose to be regex-parsed."""

    verdict: ReviewVerdict
    annotations: tuple[MQMAnnotation, ...] = ()

    def penalty(self) -> float:
        return sum(SEVERITY_WEIGHT[a.severity] for a in self.annotations)

    def score(self, word_count: int) -> float:
        """MQM-style score in [0, 100]: 100 is clean, penalties scale per 100 words."""
        if word_count <= 0:
            return 100.0
        normalised = self.penalty() / word_count * 100.0
        return max(0.0, 100.0 - normalised)


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cache_hit_rate(self) -> float:
        if self.input_tokens <= 0:
            return 0.0
        return self.cached_input_tokens / self.input_tokens


class TranslationStatus(StrEnum):
    MACHINE = "machine"
    REVIEWED = "reviewed"
    HUMAN_APPROVED = "human_approved"
    NEEDS_HUMAN = "needs_human"


class Translation(BaseModel):
    """A translated segment, plus everything needed to re-derive it.

    Sampling parameters are not a reproducibility guarantee on any provider, so
    the record of what was sent is what makes a result reproducible. That is why
    provider, model, prompt hash, schema version and effort are all required.
    """

    segment_id: str
    target_lang: str
    text: str
    provider: str
    model: str
    prompt_hash: str
    schema_version: str = SCHEMA_VERSION
    effort: str | None = None
    usage: TokenUsage = TokenUsage()
    cost_usd: float = 0.0
    latency_ms: int = 0
    status: TranslationStatus = TranslationStatus.MACHINE
    qe_score: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GlossaryTerm(BaseModel):
    """A term whose rendering is enforced, not suggested.

    `forbidden_variants` is what makes enforcement checkable: a translation that
    uses a variant fails the terminology check even if it reads well.
    """

    source_term: str = Field(min_length=1)
    target_term: str = Field(min_length=1)
    forbidden_variants: tuple[str, ...] = ()
    do_not_translate: bool = False
    case_sensitive: bool = False
    notes: str = ""


class TMEntry(BaseModel):
    """A translation memory row. Provenance decides whether it can be trusted."""

    source_text: str
    target_text: str
    source_lang: str
    target_lang: str
    domain: str = "general"
    provenance: str = Field(description="Where this pair came from")
    approved_by: str | None = None

    @property
    def is_approved(self) -> bool:
        return self.approved_by is not None
