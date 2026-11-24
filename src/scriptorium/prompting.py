"""Prompt assembly.

Two things make this a module rather than an f-string at the call site.

First, **layout is a cost decision.** Prompt caching on every candidate provider
works on a stable prefix, so the frozen instructions come first, the glossary and
retrieved examples next, and the segment — the only part that changes per call —
last. Reordering these would silently multiply the bill.

Second, **a prompt is a versioned artifact.** The template lives in `prompts/` as
a file, is hashed, and the hash is written to the audit log with every
translation. A prompt change is then a diff in a pull request rather than an
unrecorded edit, and a result can be traced to the exact text that produced it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from functools import lru_cache
from importlib.resources import files

from pydantic import BaseModel

from scriptorium.domain import GlossaryTerm, Segment, TMEntry

TRANSLATE_TEMPLATE = "translate_v1.md"


class Prompt(BaseModel):
    """A rendered prompt, split at the caching boundary."""

    system: str
    cacheable: str
    volatile: str
    template_hash: str

    @property
    def user(self) -> str:
        """The user turn: stable context first, then the segment."""
        parts = [p for p in (self.cacheable, self.volatile) if p]
        return "\n\n".join(parts)

    @property
    def prompt_hash(self) -> str:
        """Hash of everything that was actually sent, for the audit record."""
        digest = hashlib.sha256()
        digest.update(self.system.encode("utf-8"))
        digest.update(self.user.encode("utf-8"))
        return digest.hexdigest()[:16]


@lru_cache(maxsize=4)
def _template(name: str) -> tuple[str, str]:
    text = files("scriptorium.prompts").joinpath(name).read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _render_glossary(terms: Sequence[GlossaryTerm]) -> str:
    if not terms:
        return ""
    lines = ["TERMINOLOGY (binding):"]
    for term in terms:
        if term.do_not_translate:
            lines.append(f"- {term.source_term} -> DO NOT TRANSLATE")
            continue
        line = f"- {term.source_term} -> {term.target_term}"
        if term.forbidden_variants:
            line += f" (never: {', '.join(term.forbidden_variants)})"
        lines.append(line)
    return "\n".join(lines)


def _render_examples(examples: Sequence[TMEntry]) -> str:
    """Few-shot examples, retrieved per segment rather than frozen in the template.

    These are approved translation-memory pairs for this document or domain, which
    is why few-shot selection is a retrieval problem here and the number of
    examples is a parameter the evaluation sweeps.
    """
    if not examples:
        return ""
    lines = ["APPROVED EXAMPLES (same document or domain):"]
    for entry in examples:
        lines.append(f"- {entry.source_text}\n  -> {entry.target_text}")
    return "\n".join(lines)


def build_translation_prompt(
    segment: Segment,
    *,
    source_lang: str,
    target_lang: str,
    style_guide: str = "",
    glossary: Sequence[GlossaryTerm] = (),
    examples: Sequence[TMEntry] = (),
    context_before: str = "",
    context_after: str = "",
    document_summary: str = "",
) -> Prompt:
    """Assemble the translation prompt for one segment.

    The caller decides how much context to supply; that is what distinguishes
    evaluation arm B1 (segment in isolation) from B2 (with document context).
    """
    template, template_hash = _template(TRANSLATE_TEMPLATE)
    system = template.format(source_lang=source_lang, target_lang=target_lang)
    if style_guide:
        system = f"{system}\n\nSTYLE GUIDE:\n{style_guide}"

    cacheable_parts = [
        _render_glossary(glossary),
        _render_examples(examples),
        f"DOCUMENT SUMMARY:\n{document_summary}" if document_summary else "",
    ]
    cacheable = "\n\n".join(p for p in cacheable_parts if p)

    volatile_parts = [f"STRUCTURAL ROLE: {segment.structure.value}"]
    if context_before:
        volatile_parts.append(f"PRECEDING TEXT:\n{context_before}")
    if context_after:
        volatile_parts.append(f"FOLLOWING TEXT:\n{context_after}")
    volatile_parts.append(f"SEGMENT TO TRANSLATE:\n{segment.text}")

    return Prompt(
        system=system,
        cacheable=cacheable,
        volatile="\n\n".join(volatile_parts),
        template_hash=template_hash,
    )
