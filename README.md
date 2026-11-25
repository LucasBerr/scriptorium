# Scriptorium

**Terminology-controlled machine translation for long documents.**

Translating a 300-page book is not "translate each sentence well". The hard part is keeping a term
rendered the same way on page 12 and on page 280, proving it with a number, and not paying frontier
prices for the 85% of segments that do not need them.

Scriptorium is a translation pipeline that enforces a terminology base in code, retrieves its own
few-shot examples from a translation memory, reviews its output against an MQM error typology, and
reports quality *and* cost per 1,000 words against open baselines.

> **Status: phase 2 core complete, no billed run yet.** The pipeline is built and tested end to
> end offline — term base with inflection-aware enforcement, translation memory with hybrid
> retrieval, the translate → review → revise loop with typed tools and MQM output, EPUB and
> gettext adapters. `scriptorium eval` runs two checks that need no API key:
>
> | Arm | chrF++ | Terminology | Consistency | Placeholders |
> |---|---:|---:|---:|---:|
> | floor (echo) | 19.39 | 16.3% | 100.0% | 100.0% |
> | ceiling (oracle) | 100.00 | 100.0% | 100.0% | 100.0% |
> | sloppy provider, no loop | 88.13 | 48.8% | 76.7% | 100.0% |
> | the same provider, with the loop | 100.00 | 100.0% | 100.0% | 100.0% |
>
> 40 hard cases, EN → PT-BR. The first two rows check the *instrument*: a ceiling that does not top
> out would mean no comparison between real systems could be trusted. The last two check the
> *loop*: a provider derived from the references with one accepted term form swapped for a
> forbidden variant in 22 of 40 segments, run with and without the orchestrator. The deterministic
> check catches the defect and one revision fixes it.
>
> **These are not quality numbers.** The scripted providers replay the references, so chrF++ here is
> a ceiling, not a result. Quality and cost against real models need the model ids and prices pinned
> in `config/models.yaml`; until they are, the billed arms refuse to run. See
> [ADR 0001](docs/adr/0001-scope-and-language-pair.md) for why the harness was built first.

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
