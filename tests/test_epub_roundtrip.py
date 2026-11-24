"""The round-trip invariant.

Reassembling a document whose segments were never translated must reproduce every
block's markup exactly. If this fails, the pipeline is corrupting documents in a
way no translation-quality metric would ever reveal.
"""

from __future__ import annotations

from pathlib import Path

from scriptorium.adapters import EpubAdapter, Provenance
from scriptorium.adapters.epub import stitch_block
from scriptorium.domain import StructureKind
from tests.conftest import read_block_html


def test_untranslated_reassembly_is_exact(sample_epub: Path, provenance: Provenance) -> None:
    adapter = EpubAdapter()
    extracted = adapter.extract(sample_epub, provenance)

    out = sample_epub.parent / "roundtrip.epub"
    adapter.reassemble(sample_epub, extracted, {}, out)

    before = read_block_html(sample_epub)
    after = read_block_html(out)

    assert set(before) == set(after)
    for key, original in before.items():
        assert after[key] == original, f"block {key} changed on reassembly"


def test_extraction_finds_leaf_blocks_only(sample_epub: Path, provenance: Provenance) -> None:
    """A <blockquote><p> must be extracted once, not twice."""
    extracted = EpubAdapter().extract(sample_epub, provenance)
    quoted = [s for s in extracted.segments if "Delete" in s.text]
    assert len(quoted) == 1
    # The paragraph inside the blockquote is the leaf, so that is what is reported.
    assert quoted[0].structure is StructureKind.PARAGRAPH


def test_structure_kinds_are_assigned(sample_epub: Path, provenance: Provenance) -> None:
    extracted = EpubAdapter().extract(sample_epub, provenance)
    kinds = {s.structure for s in extracted.segments}
    assert StructureKind.HEADING in kinds
    assert StructureKind.PARAGRAPH in kinds
    assert StructureKind.LIST_ITEM in kinds


def test_provenance_is_carried_into_the_document(sample_epub: Path, provenance: Provenance) -> None:
    extracted = EpubAdapter().extract(sample_epub, provenance)
    assert extracted.document.licence == "public-domain"
    assert extracted.document.source_uri == provenance.source_uri


def test_translated_reassembly_keeps_surrounding_whitespace(
    sample_epub: Path, provenance: Provenance
) -> None:
    """Whitespace between sentences is the document's, not something we invent."""
    adapter = EpubAdapter()
    extracted = adapter.extract(sample_epub, provenance)

    block = next(b for b in extracted.blocks if "quick" in b.protected_text)
    segments = sorted(
        (s for s in extracted.segments if s.block_key == block.key),
        key=lambda s: s.span_start,
    )
    assert len(segments) >= 2

    stitched = stitch_block(block.protected_text, segments, {segments[0].id: "TRADUZIDO."})
    assert stitched.startswith("TRADUZIDO.")
    # The gap that followed the first sentence survives untouched.
    gap = block.protected_text[segments[0].span_end : segments[1].span_start]
    assert stitched == f"TRADUZIDO.{gap}{segments[1].text}"
