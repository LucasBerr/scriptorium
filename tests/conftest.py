"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from bs4 import BeautifulSoup, Tag
from ebooklib import epub

from scriptorium.adapters.base import Provenance
from scriptorium.adapters.epub import _PARSER, _document_items, _inner_html, _iter_leaf_blocks
from scriptorium.domain import GlossaryTerm

CHAPTER_HTML = (
    "<html><body>"
    "<h1>Chapter One</h1>"
    "<p>The <em>quick</em> brown fox jumped. It landed near Dr. Smith, quietly.</p>"
    "<blockquote><p>Delete %d files from {folder}?</p></blockquote>"
    "<ul><li>First item.</li><li>Second item.</li></ul>"
    "<p>Press the button to continue. The button is green.</p>"
    "</body></html>"
)


@pytest.fixture
def provenance() -> Provenance:
    return Provenance(
        title="Test Book",
        source_uri="https://example.org/book",
        licence="public-domain",
        licence_evidence="https://example.org/licence",
        lang="en",
    )


@pytest.fixture
def sample_epub(tmp_path: Path) -> Path:
    book = epub.EpubBook()
    book.set_identifier("test-book-1")
    book.set_title("Test Book")
    book.set_language("en")

    chapter = epub.EpubHtml(title="Chapter One", file_name="ch1.xhtml", lang="en")
    chapter.content = CHAPTER_HTML
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]

    path = tmp_path / "sample.epub"
    epub.write_epub(str(path), book)
    return path


@pytest.fixture
def glossary() -> list[GlossaryTerm]:
    return [
        GlossaryTerm(
            source_term="button",
            target_term="botão",
            forbidden_variants=("botao", "tecla"),
        ),
        GlossaryTerm(source_term="folder", target_term="pasta", forbidden_variants=("diretório",)),
    ]


def read_block_html(path: Path) -> dict[str, str]:
    """Map every leaf block in an EPUB to its inner HTML.

    Used by the round-trip test: the invariant that matters is that each block's
    markup comes back unchanged, which is what proves span stitching and
    placeholder restoration are lossless.
    """
    book = epub.read_epub(str(path))
    blocks: dict[str, str] = {}
    for item in _document_items(book):
        soup = BeautifulSoup(item.get_content(), _PARSER)
        for index, element, _structure in _iter_leaf_blocks(soup):
            assert isinstance(element, Tag)
            blocks[f"{item.get_name()}#{index}"] = _inner_html(element)
    return blocks
