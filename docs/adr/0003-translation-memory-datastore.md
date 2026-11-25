# ADR 0003 — SQLite for the translation memory, vectors in process

- Status: accepted
- Date: 2026-11-24
- Supersedes the datastore lean recorded in PROJECT.md §10

## Context

The translation memory needs three kinds of lookup: exact (has this segment been
translated before), fuzzy (is there a near match, the classic CAT-tool behaviour), and semantic (is
there a related approved translation). PROJECT.md leaned towards Postgres with pgvector, on the
grounds that one datastore is easier to defend than two.

Phase 1 changed the weighting. The project's value is its evidence, and evidence that cannot be
reproduced is worth little: `make check` and `make eval` have to run from a fresh clone, and CI
has to run them without provisioning anything.

## Decision

SQLite, one file, with embeddings stored as `float32` blobs and similarity computed in process with
numpy. Retrieval is hybrid and ordered by cost: exact hash lookup, then a token inverted index
narrowing candidates for fuzzy ranking, then cosine over the embedding matrix.

Embeddings go through an `Embedder` protocol. The default implementation is a signed character
n-gram hashing vectorizer — no model download, deterministic, and a legitimate lexical baseline. A
sentence-transformer implementation sits behind the same protocol and arrives with phase 3, where
its contribution can be *measured* against the hashing baseline rather than assumed.

## Rationale

- A reviewer clones the repository and sees a number in under a minute. With Postgres they would
  first have to start a service.
- CI stays fast and hermetic.
- The retrieval interface, not the storage engine, is what the pipeline depends on. Moving to
  pgvector later is a contained change, and by then there would be a measurement to justify it.
- Choosing the embedded option is also the more honest answer to "does this scale?": it does not,
  past a few hundred thousand segments, and saying so with a named next step is better than
  implying production readiness the project has not demonstrated.

## Consequences

- Fuzzy ranking is `O(candidates)` after the token prefilter; the prefilter is what keeps that
  workable, and the scale limit is documented rather than hidden.
- The embedding matrix is held in memory. Fine at corpus sizes this project uses; the first thing
  to break at scale, and therefore the first thing a reviewer should ask about.
- Two embedders means the semantic arm has a baseline to beat, which is more useful than a single
  unexamined choice.
