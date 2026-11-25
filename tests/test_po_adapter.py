"""The `.po` adapter is what makes "the same pipeline for books and software" real."""

from __future__ import annotations

from pathlib import Path

import polib
import pytest

from scriptorium.adapters import PoAdapter, Provenance
from scriptorium.adapters.po import SINGULAR, entry_key
from scriptorium.domain import StructureKind

CATALOGUE = """msgid ""
msgstr ""
"Project-Id-Version: demo 1.0\\n"
"Content-Type: text/plain; charset=UTF-8\\n"
"Plural-Forms: nplurals=2; plural=(n != 1);\\n"

#: ui/main.py:12
msgid "Press the button to continue."
msgstr ""

#: ui/main.py:20
msgctxt "toolbar"
msgid "Save"
msgstr "Salvar"

#: ui/files.py:8
msgid "Delete %d file from {folder}?"
msgid_plural "Delete %d files from {folder}?"
msgstr[0] ""
msgstr[1] ""
"""


@pytest.fixture
def catalogue(tmp_path: Path) -> Path:
    path = tmp_path / "messages.po"
    path.write_text(CATALOGUE, encoding="utf-8")
    return path


@pytest.fixture
def po_provenance() -> Provenance:
    return Provenance(
        title="demo catalogue",
        source_uri="https://example.org/demo",
        licence="MIT",
        licence_evidence="https://example.org/demo/LICENSE",
        lang="en",
    )


def test_every_translatable_entry_becomes_a_segment(
    catalogue: Path, po_provenance: Provenance
) -> None:
    extracted = PoAdapter().extract(catalogue, po_provenance)
    # Three entries, one of which has a plural form, and no header entry.
    assert len(extracted.segments) == 4
    assert all(s.structure is StructureKind.STRING_ENTRY for s in extracted.segments)


def test_entries_are_not_sentence_split(catalogue: Path, po_provenance: Provenance) -> None:
    """A catalog entry is the unit of translation, however many sentences it holds."""
    extracted = PoAdapter().extract(catalogue, po_provenance)
    texts = [s.text for s in extracted.segments]
    assert any("Press the button to continue." in text for text in texts)


def test_placeholders_are_protected(catalogue: Path, po_provenance: Provenance) -> None:
    extracted = PoAdapter().extract(catalogue, po_provenance)
    plural_block = next(b for b in extracted.blocks if "Delete" in b.protected_text)
    assert len(plural_block.placeholders) == 2
    assert "⟦PH0⟧" in plural_block.protected_text


def test_context_is_part_of_the_key(catalogue: Path, po_provenance: Provenance) -> None:
    """gettext keys on (msgctxt, msgid); so does the adapter, so reordering is safe."""
    extracted = PoAdapter().extract(catalogue, po_provenance)
    assert any(key.startswith("toolbar\x04Save") for key in (b.key for b in extracted.blocks))


def test_round_trip_leaves_the_catalogue_unchanged(
    catalogue: Path, po_provenance: Provenance
) -> None:
    """Reassembling with no translations must not touch a single msgstr.

    For a `.po` file "untouched" means the existing translation state, empty
    msgstr included — not the source copied into the target.
    """
    adapter = PoAdapter()
    extracted = adapter.extract(catalogue, po_provenance)
    out = catalogue.parent / "out.po"
    adapter.reassemble(catalogue, extracted, {}, out)

    before = polib.pofile(str(catalogue))
    after = polib.pofile(str(out))
    assert [(e.msgctxt, e.msgid, e.msgstr) for e in before] == [
        (e.msgctxt, e.msgid, e.msgstr) for e in after
    ]
    assert [e.msgstr_plural for e in before] == [e.msgstr_plural for e in after]


def test_translations_are_written_into_msgstr(catalogue: Path, po_provenance: Provenance) -> None:
    adapter = PoAdapter()
    extracted = adapter.extract(catalogue, po_provenance)
    target = next(s for s in extracted.segments if "Press the button" in s.text)

    out = catalogue.parent / "translated.po"
    adapter.reassemble(catalogue, extracted, {target.id: "Pressione o botão para continuar."}, out)

    written = polib.pofile(str(out))
    entry = written.find("Press the button to continue.")
    assert entry is not None
    assert entry.msgstr == "Pressione o botão para continuar."


def test_placeholders_are_restored_on_write(catalogue: Path, po_provenance: Provenance) -> None:
    adapter = PoAdapter()
    extracted = adapter.extract(catalogue, po_provenance)
    singular = next(s for s in extracted.segments if s.text.startswith("Delete"))
    block = extracted.block(singular.block_key)

    translated = singular.text.replace("Delete", "Apagar")
    out = catalogue.parent / "plural.po"
    adapter.reassemble(catalogue, extracted, {singular.id: translated}, out)

    written = polib.pofile(str(out))
    entry = next(e for e in written if e.msgid_plural)
    assert entry.msgstr_plural[0].startswith("Apagar")
    assert "%d" in entry.msgstr_plural[0]
    assert "{folder}" in entry.msgstr_plural[0]
    assert "⟦" not in entry.msgstr_plural[0]
    assert len(block.placeholders) == 2


def test_untranslated_entries_keep_their_existing_target(
    catalogue: Path, po_provenance: Provenance
) -> None:
    adapter = PoAdapter()
    extracted = adapter.extract(catalogue, po_provenance)
    out = catalogue.parent / "partial.po"
    adapter.reassemble(catalogue, extracted, {}, out)

    written = polib.pofile(str(out))
    save_entry = next(e for e in written if e.msgid == "Save")
    assert save_entry.msgstr == "Salvar"


def test_entry_key_is_stable() -> None:
    entry = polib.POEntry(msgid="Save", msgctxt="toolbar")
    assert entry_key(entry, SINGULAR) == "toolbar\x04Save\x04singular"
