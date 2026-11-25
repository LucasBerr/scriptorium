"""The adapter contract.

An adapter is responsible for two directions and nothing else: turning a source
file into blocks and segments, and putting translated segments back into a file
of the same format. Everything in between is format-agnostic.

Reassembly works by stitching translations into the block's protected text by
character offset, so the whitespace between sentences is the document's own and
not something the pipeline invented.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from scriptorium.domain import DocumentFormat, ExtractedDocument


class LicenceMissingError(ValueError):
    """Raised when a document cannot state where it came from and under what licence.

    Ingestion refuses such a document. This is a policy decision enforced in
    code rather than documented and forgotten — see docs/data-policy.md.
    """


class Provenance(BaseModel):
    """Everything an ingested document must be able to say about itself."""

    title: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    licence: str = Field(min_length=1, description="SPDX identifier or 'public-domain'")
    licence_evidence: str = Field(min_length=1)
    lang: str = Field(min_length=2)


@runtime_checkable
class DocumentAdapter(Protocol):
    """Extract segments from a document, and put translations back."""

    format: DocumentFormat

    def extract(self, path: Path, provenance: Provenance) -> ExtractedDocument:
        """Parse `path` into blocks and segments in reading order."""
        ...

    def reassemble(
        self,
        path: Path,
        extracted: ExtractedDocument,
        translations: Mapping[str, str],
        out_path: Path,
    ) -> None:
        """Write a copy of `path` with each segment replaced by its translation.

        A segment absent from `translations` is left as the source file had it.
        What that means is format-specific and deliberately so: in an EPUB the
        text lives in the document, so the source text stays; in a `.po` catalog
        the target is a separate field, so the existing `msgstr` stays, empty
        included. Either way, a reassembly with no translations reproduces the
        input, which is what makes the round-trip test possible.
        """
        ...
