# ADR 0001 — Scope, language pair and the measurement-first rule

- Status: accepted
- Date: 2026-11-24

## Context

Translating long documents with an LLM is easy to demo and hard to verify. The failure mode of a
project like this is a convincing demo with no evidence: output that reads well on the first page,
drifts in terminology by chapter three, and has no number attached to any of it.

## Decision

1. **The evaluation harness is built in phase 1, before the pipeline it is meant to judge.** No
   feature is considered done until an arm in the harness measures it. A phase does not start
   until the previous phase's number exists.
2. **Primary language pair is EN → PT-BR.** Reference translations are plentiful and the author can
   read the output critically. Error analysis in a language you do not speak is not error analysis.
3. **EPUB is the first document format, not PDF.** PDF layout extraction is an unbounded engineering
   problem that teaches nothing about language technology. PDF arrives in phase 2 as a bounded
   adapter for single-column prose.
4. **Only public-domain or permissively licensed source texts.** Enforced at ingestion: a document
   without a recorded licence and source URI is refused. This is also what makes gold references
   available for free.
5. **Terminology is enforced by code, not requested in a prompt.** A prompt instruction is a
   preference; a post-hoc checker that fails the segment is a guarantee. The project claims
   consistency, so the claim has to be mechanical.

## Consequences

- Phase 1 has no user-visible feature and will feel slow. That is the intended trade.
- Some quality improvements will be rejected because they cannot be measured on the test sets.
- The repository carries test sets and reports as first-class artifacts, not afterthoughts.
