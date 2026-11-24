# ADR 0002 — One LLM provider, reached through an interface

- Status: accepted
- Date: 2026-11-24

## Context

The pipeline needs a frontier-tier model for translation and revision, a mid tier for review, and a
small tier for bulk classification. Provider line-ups and prices change every few months, and the
claim "provider X is cheaper" is routinely made without measurement.

## Decision

OpenAI is the default provider. No module outside `scriptorium.providers` imports a vendor SDK;
everything else depends on the `LLMProvider` protocol. Model ids and prices live in
`config/models.yaml`, pinned from the provider's official pricing page with the date it was read.

A second provider is implemented behind the same protocol and used by evaluation arm **B6**, which
runs the identical pipeline, prompts and schemas and reports quality *and* cost per 1,000 words for
both.

## Rationale

The interface is not speculative abstraction — it is what makes B6 a config change instead of a
rewrite. And the comparison is worth having: at the time of writing, published figures put the cheap
and mid tiers of the two providers at roughly the same rates, with the frontier tiers differing by
2x or more in either direction depending on which source you trusted. That is a question to measure,
not to settle from a blog post.

## Consequences

- One extra indirection on every model call.
- Provider-specific features are only adopted if they can be expressed in the protocol or degraded
  cleanly. The first case of this is prompt caching, which is automatic on a stable prefix for both
  candidate providers and therefore expressible as a prompt-layout rule rather than an API feature.
- The per-token price is explicitly *not* expected to be the main cost lever. Exact TM hits that
  skip the model, cache hit rate on the stable prefix, and the share of segments that routing sends
  to the frontier tier each move the bill more. The cost report measures all three.
