# Data policy

This project processes third-party texts and sends text to a third-party API. Both facts carry
obligations, and both are enforced in code rather than described here and forgotten.

## Source texts

Every ingested document must record:

- `source_uri` — where it came from
- `licence` — an SPDX identifier, or `public-domain`
- `licence_evidence` — a URL or a note supporting the licence claim

Ingestion **refuses** a document missing any of these. Permitted sources are public-domain works
(Project Gutenberg, Domínio Público) and permissively licensed material such as an open-source
project's translation catalogs. Copyrighted books are out of scope, for legal reasons and because
public-domain works with existing published translations are what make free gold references
possible.

## Outbound text

Any text leaving the process passes a redaction pass first: email addresses, phone numbers,
and configured patterns are replaced with typed placeholders, and a unit test asserts this rather
than trusting it. `SCRIPTORIUM_OFFLINE=1` forbids outbound model calls entirely; the test suite and
CI run with it set.

## What is logged

One audit record per translation: segment hash, provider, model id, prompt hash, schema version,
effort, token counts, cost, latency, and the approving human if any. The log is the reproducibility
mechanism for the project — sampling parameters are not a guarantee on any provider, so the record
of what was sent is what makes a result re-derivable.

Source text is **not** written to the audit log, only its hash. The log is safe to publish; the
corpus is not necessarily.

## What is not in git

Corpora, model checkpoints, and anything in `data/` are tracked with DVC. Secrets live in `.env`,
which is gitignored. A commit containing an API key is a rotation event, not a revert.
