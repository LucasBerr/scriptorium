"""EPUB adapter.

EPUB comes first because its content is already structured XHTML: the effort goes
into language handling rather than into reconstructing paragraphs from glyph
coordinates. PDF arrives later as a bounded adapter (ADR 0001).

The unit of extraction is a *leaf block* — a `<p>`, heading, list item, cell or
caption that contains no other block. Its inner markup is placeholder-protected,
which turns inline tags into opaque tokens, and sentence splitting then runs on
the protected text so a tag can never split a sentence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from ebooklib import ITEM_DOCUMENT, epub

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
from scriptorium.segmentation.splitter import split_sentences

_BLOCK_KINDS: dict[str, StructureKind] = {
    "p": StructureKind.PARAGRAPH,
    "h1": StructureKind.HEADING,
    "h2": StructureKind.HEADING,
    "h3": StructureKind.HEADING,
    "h4": StructureKind.HEADING,
    "h5": StructureKind.HEADING,
    "h6": StructureKind.HEADING,
    "li": StructureKind.LIST_ITEM,
    "blockquote": StructureKind.QUOTE,
    "dd": StructureKind.PARAGRAPH,
    "dt": StructureKind.HEADING,
    "td": StructureKind.TABLE_CELL,
    "th": StructureKind.TABLE_CELL,
    "figcaption": StructureKind.CAPTION,
}
_BLOCK_TAGS = tuple(_BLOCK_KINDS)

# html.parser keeps the document's own markup verbatim in `decode_contents()`,
# which is what the offset-stitching reassembly depends on.
_PARSER = "html.parser"


def _inner_html(element: Tag) -> str:
    return str(element.decode_contents())


def _iter_leaf_blocks(soup: BeautifulSoup) -> Iterator[tuple[int, Tag, StructureKind]]:
    """Yield blocks that contain no nested block, in document order.

    Without the leaf filter a `<blockquote><p>…</p></blockquote>` would be
    extracted twice and translated twice.
    """
    index = 0
    for element in soup.find_all(_BLOCK_TAGS):
        if element.find(_BLOCK_TAGS) is not None:
            continue
        if not _inner_html(element).strip():
            continue
        yield index, element, _BLOCK_KINDS[element.name]
        index += 1


def _document_items(book: epub.EpubBook) -> list[epub.EpubItem]:
    """XHTML items in spine order, falling back to manifest order."""
    by_id = {item.get_id(): item for item in book.get_items_of_type(ITEM_DOCUMENT)}
    ordered: list[epub.EpubItem] = []
    for entry in book.spine:
        idref = entry[0] if isinstance(entry, tuple) else entry
        item = by_id.pop(str(idref), None)
        if item is not None:
            ordered.append(item)
    ordered.extend(by_id.values())
    return ordered


class EpubAdapter:
    """Extract from and reassemble into EPUB."""

    format: DocumentFormat = DocumentFormat.EPUB

    def extract(self, path: Path, provenance: Provenance) -> ExtractedDocument:
        book = epub.read_epub(str(path))
        document = SourceDocument(
            id=hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12],
            title=provenance.title,
            source_uri=provenance.source_uri,
            licence=provenance.licence,
            licence_evidence=provenance.licence_evidence,
            lang=provenance.lang,
            format=DocumentFormat.EPUB,
        )

        blocks: list[Block] = []
        segments: list[Segment] = []
        segment_index = 0

        for item in _document_items(book):
            item_name = str(item.get_name())
            soup = BeautifulSoup(item.get_content(), _PARSER)
            for block_index, element, structure in _iter_leaf_blocks(soup):
                protected = protect(_inner_html(element))
                key = f"{item_name}#{block_index}"
                blocks.append(
                    Block(
                        key=key,
                        item=item_name,
                        index=block_index,
                        structure=structure,
                        protected_text=protected.text,
                        placeholders=protected.placeholders,
                    )
                )
                for span in split_sentences(protected.text, provenance.lang):
                    segments.append(
                        Segment(
                            id=f"{document.id}:{segment_index:06d}",
                            document_id=document.id,
                            block_key=key,
                            index=segment_index,
                            text=span.text,
                            structure=structure,
                            span_start=span.start,
                            span_end=span.end,
                        )
                    )
                    segment_index += 1

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
        rendered = {
            block.key: restore(
                stitch_block(block.protected_text, block_segments, translations),
                block.placeholders,
            )
            for block, block_segments in _group_segments(extracted)
        }

        book = epub.read_epub(str(path))
        for item in _document_items(book):
            item_name = str(item.get_name())
            soup = BeautifulSoup(item.get_content(), _PARSER)
            touched = False
            for block_index, element, _structure in _iter_leaf_blocks(soup):
                fragment = rendered.get(f"{item_name}#{block_index}")
                if fragment is None:
                    continue
                element.clear()
                element.append(BeautifulSoup(fragment, _PARSER))
                touched = True
            if touched:
                item.set_content(str(soup).encode("utf-8"))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        epub.write_epub(str(out_path), book)


def _group_segments(extracted: ExtractedDocument) -> Iterator[tuple[Block, list[Segment]]]:
    by_block: dict[str, list[Segment]] = {}
    for segment in extracted.segments:
        by_block.setdefault(segment.block_key, []).append(segment)
    for block in extracted.blocks:
        yield block, sorted(by_block.get(block.key, ()), key=lambda s: s.span_start)


def stitch_block(
    protected_text: str,
    segments: list[Segment],
    translations: Mapping[str, str],
) -> str:
    """Replace each segment's span in `protected_text` with its translation.

    The text between spans is copied verbatim, so the document's own whitespace
    survives. A segment with no translation keeps its source text, which is what
    makes an untranslated reassembly an exact round trip.
    """
    pieces: list[str] = []
    cursor = 0
    for segment in segments:
        if segment.span_start < cursor:
            msg = f"overlapping spans in block at segment {segment.id}"
            raise ValueError(msg)
        pieces.append(protected_text[cursor : segment.span_start])
        pieces.append(translations.get(segment.id, segment.text))
        cursor = segment.span_end
    pieces.append(protected_text[cursor:])
    return "".join(pieces)
