"""The translation memory.

Retrieval is hybrid and ordered by cost, cheapest first:

1. **Exact** — the same source text, already translated and approved. This is the
   one that saves real money, because it skips the model entirely.
2. **Fuzzy** — a near match, ranked by edit similarity. The classic CAT-tool
   behaviour, and the reason a human post-editing one segment makes the next
   hundred cheaper.
3. **Semantic** — a related approved translation, found by cosine over
   embeddings. These become the retrieved few-shot examples in the prompt.

Fuzzy candidates come from a token inverted index rather than a full scan, which
is what keeps the ranking workable. The honest scale limit is stated in ADR 0003:
the embedding matrix lives in memory, so this design holds for corpus sizes in
the hundreds of thousands of segments and not beyond.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from scriptorium.domain import TMEntry, content_hash
from scriptorium.memory.embedding import Embedder, HashingEmbedder, Vector
from scriptorium.terminology.mining import tokenise

HitKind = Literal["exact", "fuzzy", "semantic"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tm_entries (
    id            INTEGER PRIMARY KEY,
    source_hash   TEXT NOT NULL,
    source_text   TEXT NOT NULL,
    target_text   TEXT NOT NULL,
    source_lang   TEXT NOT NULL,
    target_lang   TEXT NOT NULL,
    domain        TEXT NOT NULL DEFAULT 'general',
    provenance    TEXT NOT NULL,
    approved_by   TEXT,
    created_at    TEXT NOT NULL,
    embedding     BLOB
);
CREATE INDEX IF NOT EXISTS idx_tm_lookup
    ON tm_entries (source_hash, source_lang, target_lang);
CREATE INDEX IF NOT EXISTS idx_tm_pair
    ON tm_entries (source_lang, target_lang);

-- Inverted index: narrows fuzzy candidates to entries sharing a word, so
-- ranking does not scan the whole memory.
CREATE TABLE IF NOT EXISTS tm_tokens (
    token    TEXT NOT NULL,
    entry_id INTEGER NOT NULL REFERENCES tm_entries(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_tm_tokens ON tm_tokens (token);
"""


class TMHit(BaseModel):
    """A retrieved translation memory entry and how it was found."""

    entry: TMEntry
    score: float = Field(ge=0.0, le=1.0)
    kind: HitKind

    @property
    def is_exact(self) -> bool:
        return self.kind == "exact"


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


class TranslationMemory:
    """SQLite-backed translation memory with hybrid retrieval."""

    def __init__(self, path: Path | None = None, embedder: Embedder | None = None) -> None:
        self.path = path
        self.embedder: Embedder = embedder or HashingEmbedder()
        self._conn = sqlite3.connect(str(path) if path else ":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._matrix: Vector | None = None
        self._matrix_ids: list[int] = []

    # ---------------------------------------------------------------- writing

    def add(self, entry: TMEntry) -> int:
        """Insert or replace an entry, keyed by source text and language pair."""
        source_hash = content_hash(entry.source_text)
        vector = self.embedder.encode([entry.source_text])[0]

        cursor = self._conn.execute(
            "SELECT id FROM tm_entries WHERE source_hash = ? AND source_lang = ? "
            "AND target_lang = ?",
            (source_hash, entry.source_lang, entry.target_lang),
        )
        existing = cursor.fetchone()
        if existing is not None:
            entry_id = int(existing["id"])
            self._conn.execute(
                "UPDATE tm_entries SET target_text = ?, domain = ?, provenance = ?, "
                "approved_by = ?, embedding = ? WHERE id = ?",
                (
                    entry.target_text,
                    entry.domain,
                    entry.provenance,
                    entry.approved_by,
                    vector.tobytes(),
                    entry_id,
                ),
            )
        else:
            cursor = self._conn.execute(
                "INSERT INTO tm_entries (source_hash, source_text, target_text, source_lang, "
                "target_lang, domain, provenance, approved_by, created_at, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source_hash,
                    entry.source_text,
                    entry.target_text,
                    entry.source_lang,
                    entry.target_lang,
                    entry.domain,
                    entry.provenance,
                    entry.approved_by,
                    datetime.now(UTC).isoformat(),
                    vector.tobytes(),
                ),
            )
            entry_id = int(cursor.lastrowid or 0)

        self._conn.execute("DELETE FROM tm_tokens WHERE entry_id = ?", (entry_id,))
        tokens = {t.lower() for t in tokenise(entry.source_text)}
        self._conn.executemany(
            "INSERT INTO tm_tokens (token, entry_id) VALUES (?, ?)",
            [(token, entry_id) for token in tokens],
        )
        self._conn.commit()
        self._matrix = None
        return entry_id

    def add_many(self, entries: Iterable[TMEntry]) -> list[int]:
        return [self.add(entry) for entry in entries]

    def record_post_edit(
        self,
        source_text: str,
        target_text: str,
        *,
        source_lang: str,
        target_lang: str,
        approved_by: str,
        domain: str = "general",
    ) -> int:
        """Write a human's correction back into the memory.

        This is the loop that makes the system improve rather than merely run: an
        edited segment becomes an approved exact hit, so the same source is never
        paid for twice and later similar segments get it as a fuzzy example.
        """
        return self.add(
            TMEntry(
                source_text=source_text,
                target_text=target_text,
                source_lang=source_lang,
                target_lang=target_lang,
                domain=domain,
                provenance="human post-edit",
                approved_by=approved_by,
            )
        )

    # ---------------------------------------------------------------- reading

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM tm_entries").fetchone()
        return int(row["n"])

    @staticmethod
    def _to_entry(row: sqlite3.Row) -> TMEntry:
        return TMEntry(
            source_text=row["source_text"],
            target_text=row["target_text"],
            source_lang=row["source_lang"],
            target_lang=row["target_lang"],
            domain=row["domain"],
            provenance=row["provenance"],
            approved_by=row["approved_by"],
        )

    def exact(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        approved_only: bool = True,
    ) -> TMEntry | None:
        """The approved translation of exactly this text, if there is one.

        The caller short-circuits on this: an approved exact hit needs no model,
        which is the single largest cost lever in the pipeline.
        """
        query = (
            "SELECT * FROM tm_entries WHERE source_hash = ? AND source_lang = ? AND target_lang = ?"
        )
        params: list[object] = [content_hash(text), source_lang, target_lang]
        if approved_only:
            query += " AND approved_by IS NOT NULL"
        row = self._conn.execute(query, params).fetchone()
        return self._to_entry(row) if row is not None else None

    def _fuzzy(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        approved_only: bool,
        minimum: float,
        limit: int,
        exclude: set[str],
    ) -> list[TMHit]:
        tokens = {t.lower() for t in tokenise(text)}
        if not tokens:
            return []

        placeholders = ",".join("?" for _ in tokens)
        query = (
            f"SELECT DISTINCT e.* FROM tm_entries e "
            f"JOIN tm_tokens t ON t.entry_id = e.id "
            f"WHERE t.token IN ({placeholders}) AND e.source_lang = ? AND e.target_lang = ?"
        )
        params: list[object] = [*tokens, source_lang, target_lang]
        if approved_only:
            query += " AND e.approved_by IS NOT NULL"

        normalised = _normalise(text)
        hits: list[TMHit] = []
        for row in self._conn.execute(query, params):
            if row["source_text"] in exclude:
                continue
            score = SequenceMatcher(None, normalised, _normalise(row["source_text"])).ratio()
            if score >= minimum:
                hits.append(TMHit(entry=self._to_entry(row), score=score, kind="fuzzy"))

        hits.sort(key=lambda hit: -hit.score)
        return hits[:limit]

    def _embedding_matrix(self) -> tuple[Vector, list[int], dict[int, sqlite3.Row]]:
        rows = {
            int(row["id"]): row
            for row in self._conn.execute("SELECT * FROM tm_entries WHERE embedding IS NOT NULL")
        }
        if self._matrix is None or self._matrix_ids != list(rows):
            self._matrix_ids = list(rows)
            if self._matrix_ids:
                self._matrix = np.vstack(
                    [
                        np.frombuffer(rows[entry_id]["embedding"], dtype=np.float32)
                        for entry_id in self._matrix_ids
                    ]
                )
            else:
                self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
        return self._matrix, self._matrix_ids, rows

    def _semantic(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        approved_only: bool,
        minimum: float,
        limit: int,
        exclude: set[str],
    ) -> list[TMHit]:
        matrix, ids, rows = self._embedding_matrix()
        if not ids:
            return []

        query_vector = self.embedder.encode([text])[0]
        # Both sides are L2-normalised, so the dot product is cosine similarity.
        scores = matrix @ query_vector

        hits: list[TMHit] = []
        for entry_id, score in zip(ids, scores, strict=True):
            row = rows[entry_id]
            if row["source_lang"] != source_lang or row["target_lang"] != target_lang:
                continue
            if approved_only and row["approved_by"] is None:
                continue
            if row["source_text"] in exclude:
                continue
            if float(score) < minimum:
                continue
            hits.append(TMHit(entry=self._to_entry(row), score=float(score), kind="semantic"))

        hits.sort(key=lambda hit: -hit.score)
        return hits[:limit]

    def search(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        k: int = 3,
        approved_only: bool = True,
        min_fuzzy: float = 0.6,
        min_semantic: float = 0.5,
    ) -> list[TMHit]:
        """Hybrid retrieval, cheapest and most reliable first.

        An exact hit is returned on its own: once the same source has an approved
        translation, nothing else is worth looking at.
        """
        exact = self.exact(
            text, source_lang=source_lang, target_lang=target_lang, approved_only=approved_only
        )
        if exact is not None:
            return [TMHit(entry=exact, score=1.0, kind="exact")]

        seen: set[str] = set()
        hits = self._fuzzy(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
            approved_only=approved_only,
            minimum=min_fuzzy,
            limit=k,
            exclude=seen,
        )
        seen.update(hit.entry.source_text for hit in hits)

        if len(hits) < k:
            hits.extend(
                self._semantic(
                    text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    approved_only=approved_only,
                    minimum=min_semantic,
                    limit=k - len(hits),
                    exclude=seen,
                )
            )
        return hits[:k]

    def examples(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        k: int = 3,
    ) -> list[TMEntry]:
        """Approved pairs to use as retrieved few-shot examples."""
        return [
            hit.entry
            for hit in self.search(
                text, source_lang=source_lang, target_lang=target_lang, k=k, approved_only=True
            )
        ]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TranslationMemory:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
