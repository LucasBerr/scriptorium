"""Sentence splitting.

Rule-based (pysbd) rather than model-based: it is deterministic, fast, needs no
GPU, and its failures are inspectable. Splitting runs on placeholder-protected
text, so inline markup cannot split a sentence.

Spans are returned alongside the sentences because reassembly stitches
translations back into the original block by offset, which is what makes an exact
round trip possible.
"""

from __future__ import annotations

import re
import warnings
from functools import lru_cache

with warnings.catch_warnings():
    # pysbd's own source triggers SyntaxWarning on import under Python 3.12.
    # Upstream's problem, but it would otherwise pollute every CLI run.
    warnings.simplefilter("ignore", SyntaxWarning)
    import pysbd

from pydantic import BaseModel

# pysbd's supported language codes, which are not the same set as the project's
# locale tags. 'pt-BR' is a locale; pysbd wants 'pt'.
_PYSBD_LANGUAGES = frozenset(
    {
        "am",
        "ar",
        "bg",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fa",
        "fr",
        "hi",
        "hy",
        "it",
        "ja",
        "kk",
        "mr",
        "my",
        "nl",
        "pl",
        "ru",
        "sk",
        "ur",
        "zh",
    }
)

_FALLBACK_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?=[^\s])")


class SentenceSpan(BaseModel):
    """A sentence and where it sits in the text it came from."""

    text: str
    start: int
    end: int


def _pysbd_code(lang: str) -> str | None:
    base = lang.split("-")[0].lower()
    return base if base in _PYSBD_LANGUAGES else None


@lru_cache(maxsize=8)
def _segmenter(code: str) -> pysbd.Segmenter:
    # clean=False is required for char_span to be meaningful: cleaning rewrites
    # the text, and offsets into a rewritten string are useless for reassembly.
    return pysbd.Segmenter(language=code, clean=False, char_span=True)


def _fallback_split(text: str) -> list[SentenceSpan]:
    spans: list[SentenceSpan] = []
    cursor = 0
    for match in _FALLBACK_BOUNDARY.finditer(text):
        end = match.start()
        if text[cursor:end].strip():
            spans.append(SentenceSpan(text=text[cursor:end], start=cursor, end=end))
        cursor = match.end()
    if text[cursor:].strip():
        spans.append(SentenceSpan(text=text[cursor:], start=cursor, end=len(text)))
    return spans


def split_sentences(text: str, lang: str = "en") -> list[SentenceSpan]:
    """Split `text` into sentences with character spans into `text`.

    Falls back to a punctuation-based splitter for languages pysbd does not
    support, so an unsupported language degrades instead of failing.
    """
    if not text.strip():
        return []

    code = _pysbd_code(lang)
    if code is None:
        return _fallback_split(text)

    spans: list[SentenceSpan] = []
    for span in _segmenter(code).segment(text):
        sentence = str(span.sent)
        if not sentence.strip():
            continue
        spans.append(SentenceSpan(text=sentence, start=int(span.start), end=int(span.end)))
    return spans or _fallback_split(text)
