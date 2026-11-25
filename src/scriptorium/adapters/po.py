"""Gettext `.po` adapter.

This is the adapter that makes the project's claim concrete: the same pipeline,
the same term base and the same translation memory handle a book and a software
product's UI strings. Terminology consistency across thousands of short strings
is the harder half of that problem, and it is the half a localization team
actually lives with.

Two deliberate differences from the EPUB adapter:

- **No sentence splitting.** A catalog entry is the unit of translation. Splitting
  "Delete %d files?" into clauses would produce fragments a translator cannot
  judge and a `msgstr` nobody can reassemble.
- **Reassembly writes `msgstr`, and leaves entries it has no translation for
  exactly as they were.** For a `.po` file, "untouched" means the existing
  translation state — including an empty `msgstr` — not the source text copied
  into the target.

Parsing goes through `polib` rather than a hand-rolled reader: plural forms,
contexts, flags and comments are a real format with real edge cases, and getting
them subtly wrong would corrupt catalogs silently.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from pathlib import Path

import polib

from scriptorium.adapters.base import Provenance
from scriptorium.domain import (
    Block,
    DocumentFormat,
    ExtractedDocument,
    Segment,
    SourceDocument,
    StructureKind,
)
from scriptorium.segmentation.placeholders import protect, restore

# gettext's own separator between context and message id.
CONTEXT_SEPARATOR = "\x04"

SINGULAR = "singular"
PLURAL = "plural"


def entry_key(entry: polib.POEntry, form: str = SINGULAR) -> str:
    """Stable address for a catalog entry.

    `(msgctxt, msgid)` is what gettext itself uses as a unique key, so it is the
    right address here too — and it survives reordering of the file, which an
    index-based key would not.
    """
    context = entry.msgctxt or ""
    return f"{context}{CONTEXT_SEPARATOR}{entry.msgid}{CONTEXT_SEPARATOR}{form}"


def _translatable(catalogue: polib.POFile) -> Iterator[polib.POEntry]:
    """Entries worth translating: not obsolete, not the header, not empty."""
    for entry in catalogue:
        if entry.obsolete or not entry.msgid:
            continue
        yield entry


class PoAdapter:
    """Extract from and reassemble into a gettext catalog."""

    format: DocumentFormat = DocumentFormat.PO

    def extract(self, path: Path, provenance: Provenance) -> ExtractedDocument:
        catalogue = polib.pofile(str(path))
        document = SourceDocument(
            id=hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12],
            title=provenance.title,
            source_uri=provenance.source_uri,
            licence=provenance.licence,
            licence_evidence=provenance.licence_evidence,
            lang=provenance.lang,
            format=DocumentFormat.PO,
        )

        blocks: list[Block] = []
        segments: list[Segment] = []
        index = 0

        for position, entry in enumerate(_translatable(catalogue)):
            forms = [(SINGULAR, entry.msgid)]
            if entry.msgid_plural:
                forms.append((PLURAL, entry.msgid_plural))

            for form, text in forms:
                protected = protect(text)
                key = entry_key(entry, form)
                blocks.append(
                    Block(
                        key=key,
                        item=path.name,
                        index=position,
                        structure=StructureKind.STRING_ENTRY,
                        protected_text=protected.text,
                        placeholders=protected.placeholders,
                    )
                )
                segments.append(
                    Segment(
                        id=f"{document.id}:{index:06d}",
                        document_id=document.id,
                        block_key=key,
                        index=index,
                        text=protected.text,
                        structure=StructureKind.STRING_ENTRY,
                        span_start=0,
                        span_end=len(protected.text),
                    )
                )
                index += 1

        return ExtractedDocument(
            document=document,
            blocks=tuple(blocks),
            segments=tuple(segments),
        )

    def reassemble(
        self,
        path: Path,
        extracted: ExtractedDocument,
        translations: Mapping[str, str],
        out_path: Path,
    ) -> None:
        """Write translations into `msgstr`, leaving untranslated entries alone.

        Plural handling assumes a two-form target, which is correct for PT-BR and
        wrong for languages such as Polish or Arabic. Respecting the catalog's
        own `Plural-Forms` header is the generalisation, and it waits until there
        is a language in scope that needs it.
        """
        rendered: dict[str, str] = {}
        for segment in extracted.segments:
            translated = translations.get(segment.id)
            if translated is None:
                continue
            block = extracted.block(segment.block_key)
            rendered[segment.block_key] = restore(translated, block.placeholders)

        catalogue = polib.pofile(str(path))
        for entry in _translatable(catalogue):
            singular = rendered.get(entry_key(entry, SINGULAR))
            if entry.msgid_plural:
                plural = rendered.get(entry_key(entry, PLURAL))
                if singular is not None:
                    entry.msgstr_plural[0] = singular
                if plural is not None:
                    entry.msgstr_plural[1] = plural
            elif singular is not None:
                entry.msgstr = singular

        out_path.parent.mkdir(parents=True, exist_ok=True)
        catalogue.save(str(out_path))
