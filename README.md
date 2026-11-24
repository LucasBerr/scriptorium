# Scriptorium

**Terminology-controlled machine translation for long documents.**

Translating a 300-page book is not "translate each sentence well". The hard part is keeping a term
rendered the same way on page 12 and on page 280, proving it with a number, and not paying frontier
prices for the 85% of segments that do not need them.

Scriptorium is a translation pipeline that enforces a terminology base in code, retrieves its own
few-shot examples from a translation memory, reviews its output against an MQM error typology, and
reports quality *and* cost per 1,000 words against open baselines.

> **Status: phase 1 in progress.** The evaluation harness is built before the pipeline it judges —
> see [ADR 0001](docs/adr/0001-scope-and-language-pair.md). The harness, the EPUB adapter and the
> metrics are in place and self-checking: `scriptorium eval` runs an echo floor and a
> reference-replaying ceiling over the hard-case set and confirms the ceiling tops out on every
> metric. No model-backed arm has run yet, so there are no headline quality numbers; this line is
> replaced by the metric table when phase 1 closes.

## What is here

| | |
|---|---|
| Plan, architecture and phase exit criteria | [PROJECT.md](PROJECT.md) |
| Why the scope is what it is | [docs/adr/](docs/adr/) |
| Licensing, PII and audit rules | [docs/data-policy.md](docs/data-policy.md) |

## Quick start

```bash
make install        # uv sync --extra dev
make check          # lint + type + test, never makes a billed API call
```

```bash
uv run scriptorium eval            # harness self-check: floor vs ceiling, no API key needed
uv run scriptorium check-config    # are model ids and prices really pinned?
```

To run anything that calls a model, copy `.env.example` to `.env`, fill in a key, and pin real
model ids and prices in `config/models.yaml`. Until those are pinned, `check-config` exits non-zero
and the billed providers refuse to start — a cost report computed from placeholder prices would be
fiction that looks like a result.

## Language pair

EN → PT-BR. Chosen because reference translations are plentiful and the author can judge the output
critically — error analysis in a language you do not speak is not error analysis.
