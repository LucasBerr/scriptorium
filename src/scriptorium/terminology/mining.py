"""Term candidate mining.

A term base nobody maintains is empty, so candidates are proposed from the
document itself and then approved by a human. Approval is the point: this module
produces *proposals*, never entries.

The method is deliberately simple and explainable — frequency, a length
preference, a capitalisation signal, and a cheap collocation test for multi-word
candidates. No part-of-speech analysis, which means noun-phrase precision is
mediocre; spaCy-based filtering is the phase-3 upgrade, and it earns its place by
being measured against this baseline rather than assumed better.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, Field

_TOKEN = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*", re.UNICODE)

# Small, explicit, English-only. A real stop list is language data and belongs in
# a resource file once a second source language exists.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "without",
        "about",
        "into",
        "over",
        "under",
        "again",
        "further",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "it",
        "its",
        "as",
        "not",
        "no",
        "nor",
        "so",
        "such",
        "only",
        "own",
        "same",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "i",
        "you",
        "he",
        "she",
        "they",
        "we",
        "me",
        "him",
        "her",
        "them",
        "my",
        "your",
        "his",
        "their",
        "our",
        "which",
        "who",
        "whom",
        "what",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "there",
        "here",
    ]
)


class TermCandidate(BaseModel):
    """A proposed term, awaiting human approval."""

    text: str
    frequency: int = Field(ge=1)
    score: float
    words: int = Field(ge=1)
    capitalised: bool = False
    example: str = ""

    @property
    def is_multiword(self) -> bool:
        return self.words > 1


def tokenise(text: str) -> list[str]:
    """Word tokens, shared with the translation memory's inverted index.

    One tokenizer, so index keys and mined candidates cannot disagree about
    what a word is.
    """
    return _TOKEN.findall(text)


def _is_edge_stopword(words: Sequence[str]) -> bool:
    return words[0].lower() in STOPWORDS or words[-1].lower() in STOPWORDS


def mine_terms(
    segments: Iterable[str],
    *,
    min_frequency: int = 3,
    max_ngram: int = 3,
    max_terms: int = 50,
    collocation_threshold: float = 0.6,
) -> list[TermCandidate]:
    """Propose term candidates from a document's segments.

    Args:
        min_frequency: a candidate must recur; one-offs are not terminology.
        max_ngram: longest candidate in words.
        max_terms: how many proposals to return, best first.
        collocation_threshold: for multi-word candidates, the share of its rarest
            constituent word's occurrences that must fall inside the candidate.
            This is the cheap test for "these words mostly travel together" —
            it keeps "of the system" out while letting "translation memory" in.
    """
    texts = list(segments)
    unigram_counts: Counter[str] = Counter()
    ngram_counts: Counter[tuple[str, ...]] = Counter()
    capitalised: set[tuple[str, ...]] = set()
    examples: dict[tuple[str, ...], str] = {}

    for text in texts:
        tokens = tokenise(text)
        lowered = [t.lower() for t in tokens]
        unigram_counts.update(lowered)
        for size in range(1, max_ngram + 1):
            for start in range(len(tokens) - size + 1):
                key = tuple(lowered[start : start + size])
                if _is_edge_stopword(key):
                    continue
                ngram_counts[key] += 1
                examples.setdefault(key, text)
                # Capitalised away from the start of the segment is a signal that
                # this is a name or a product term rather than ordinary prose.
                if start > 0 and tokens[start][:1].isupper():
                    capitalised.add(key)

    candidates: list[TermCandidate] = []
    for key, frequency in ngram_counts.items():
        if frequency < min_frequency:
            continue

        size = len(key)
        if size > 1:
            rarest = min(unigram_counts[word] for word in key)
            if rarest == 0 or frequency / rarest < collocation_threshold:
                continue

        length_preference = 1.0 + 0.5 * (size - 1)
        capital_bonus = 1.5 if key in capitalised else 1.0
        candidates.append(
            TermCandidate(
                text=" ".join(key),
                frequency=frequency,
                score=frequency * length_preference * capital_bonus,
                words=size,
                capitalised=key in capitalised,
                example=examples[key],
            )
        )

    candidates.sort(key=lambda c: (-c.score, -c.frequency, c.text))
    return candidates[:max_terms]
