"""Embeddings for semantic retrieval.

The default embedder is a signed character n-gram hashing vectorizer: no model
download, no GPU, deterministic across processes and machines. It is a real
lexical baseline, not a stub — and having it means the sentence-transformer
implementation that arrives in phase 3 has something to beat. A semantic
retriever adopted without a baseline is an assumption wearing the clothes of a
decision.

Character n-grams rather than words because the target side is Portuguese:
inflection and agreement change word endings constantly, and character n-grams
degrade gracefully where whole-word matching falls off a cliff.

What it cannot do, measured rather than asserted: synonymy. Cosine between
"Press the button to continue." and "Press the button to proceed." is 0.60, but
between "A folder holds files." and "The file was moved to the folder." only
0.32, and between "Documents live inside directories." and that same sentence
0.04 — the meaning is close, the characters are not. Closing that gap is what a
sentence embedder is for, and these numbers are the baseline it has to beat.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    """Turn texts into L2-normalised vectors, so cosine similarity is a dot product."""

    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> Vector:
        """Return a `(len(texts), dim)` float32 matrix of unit vectors."""
        ...


def _char_ngrams(text: str, low: int, high: int) -> list[str]:
    # Padding makes word starts and ends their own features, which is what lets
    # a prefix match contribute even when the suffix inflects.
    padded = f" {text.lower().strip()} "
    grams: list[str] = []
    for size in range(low, high + 1):
        if len(padded) < size:
            continue
        grams.extend(padded[i : i + size] for i in range(len(padded) - size + 1))
    return grams


class HashingEmbedder:
    """Signed hashing of character n-grams into a fixed-width vector.

    Deterministic by construction: `hashlib` rather than Python's `hash`, which is
    randomised per process and would make stored vectors meaningless on reload.
    """

    name = "hashing-char-ngram"

    def __init__(self, dim: int = 256, ngram_range: tuple[int, int] = (3, 5)) -> None:
        if dim <= 0:
            msg = "dim must be positive"
            raise ValueError(msg)
        low, high = ngram_range
        if not 0 < low <= high:
            msg = f"invalid ngram_range: {ngram_range}"
            raise ValueError(msg)
        self.dim = dim
        self._low = low
        self._high = high

    def _bucket_and_sign(self, gram: str) -> tuple[int, float]:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        # The sign comes from a different part of the digest than the bucket, so
        # collisions in one do not systematically correlate with the other.
        return value % self.dim, 1.0 if (value >> 61) & 1 else -1.0

    def encode(self, texts: Sequence[str]) -> Vector:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for gram in _char_ngrams(text, self._low, self._high):
                bucket, sign = self._bucket_and_sign(gram)
                matrix[row, bucket] += sign
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # An empty or whitespace-only text has no features; leave it at zero
        # rather than dividing by zero, and let cosine report 0 similarity.
        np.divide(matrix, norms, out=matrix, where=norms > 0)
        return matrix
