# Scriptorium

**Terminology-controlled machine translation for long documents.**

A translation pipeline that keeps a term rendered the same way on page 12 and on page 280,
measures its own output, and gets cheaper as it learns. Books are the first document type;
software localization files are the second.

---

## 1. Problem statement

Translating a 300-page book or a product's UI strings is not "translate each sentence well".
The hard part is **consistency and verifiability at scale**:

- The same source term must map to the same target term across thousands of segments and across
  documents, even when each segment is translated independently.
- Quality must be *measured*, not asserted — per segment, with a number, against a baseline.
- Cost must scale sub-linearly with document size, or the system is unusable in production.
- Human corrections must flow back in, or the system never improves.

These are the same constraints a software-localization platform faces. The deliverable is the
pipeline and its evaluation evidence, not a demo.

## 2. Scope

**In scope**
- Document ingestion: EPUB (primary), PDF (bounded subset), `.po` / `.properties` / flat i18n JSON.
- Segmentation, structure preservation, round-trip reassembly into the source format.
- Translation memory (fuzzy + semantic) and an enforced terminology base.
- Agentic translate → review → revise loop with tool use and structured output.
- A locally fine-tuned MT model and a quality-estimation model, both trained here.
- An evaluation harness with reference-based and reference-free metrics and significance testing.
- A reviewer UI with segment-level confidence and post-editing that writes back to the TM.
- Container, CI, tests, secret handling, audit trail, model card.

**Out of scope**
- Scanned/OCR documents. Layout-faithful PDF regeneration. Right-to-left scripts.
- Real-time/streaming translation. Multi-tenant auth and billing.
- Beating state-of-the-art general MT. The target is *in-domain, terminology-constrained* quality.

## 3. Language pairs and data

Primary pair: **EN → PT-BR**. Chosen because reference translations are plentiful, the author can
judge output quality directly, and error analysis is only credible in a language you actually speak.

Secondary pair (phase 3, optional): **DE → PT-BR**, to demonstrate that nothing in the pipeline is
hardcoded to one pair.

| Dataset | Use | Source |
|---|---|---|
| Public-domain books with published public-domain translations | In-domain test set with gold references | Project Gutenberg, Domínio Público |
| OPUS / Tatoeba / TED subsets, EN-PT | Fine-tuning corpus, general dev set | OPUS collection |
| An OSS project's `.po` catalogs with existing translations | Software-localization test set | A permissively licensed repository |
| Hand-curated hard cases (~200 segments) | Regression set for known failure modes | Written by hand |

**Licensing is a hard constraint, not a footnote.** Every ingested text records its licence and
source URI, and the pipeline refuses to ingest a document without them. The hard-case set covers
terminology traps, gender agreement, ambiguous pronouns, units and measures, named entities, and
strings with placeholders (`%s`, `{count}`) that must survive translation intact.

## 4. Architecture

```
                 ┌──────────────────────────────────────────────────┐
                 │  INGESTION  (adapters: epub | pdf | po | json)   │
                 │  extract → structure-tag → segment → hash        │
                 └────────────────────┬─────────────────────────────┘
                                      │ Segment[]
                 ┌────────────────────▼─────────────────────────────┐
                 │  CONTEXT ASSEMBLY                                │
                 │  TM fuzzy+vector hits · glossary slice ·          │
                 │  rolling document summary · neighbour window      │
                 └────────────────────┬─────────────────────────────┘
                                      │ TranslationRequest
     ┌────────────────────────────────▼─────────────────────────────┐
     │  ORCHESTRATOR (agentic loop, structured output)              │
     │                                                              │
     │   Translator ──► Reviewer (MQM) ──► Reviser ──┐              │
     │        ▲                                      │ until clean  │
     │        └──────────────────────────────────────┘ or budget     │
     │                                                              │
     │   tools: lookup_terminology · search_translation_memory       │
     │          check_consistency · estimate_quality                 │
     │          propose_glossary_entry · flag_for_human              │
     └────────────────────┬─────────────────────────────────────────┘
                          │ Translation + QE score + MQM annotations
     ┌────────────────────▼─────────────────────────────────────────┐
     │  QA & ROUTING                                                │
     │  QE model → confidence → cheap path | expensive path | human │
     └────────────────────┬─────────────────────────────────────────┘
                          │
     ┌────────────────────▼──────────┐   ┌──────────────────────────┐
     │  REASSEMBLY → output document │   │  REVIEWER UI             │
     └───────────────────────────────┘   │  post-edit → TM writeback│
                                         └──────────────────────────┘

     ┌──────────────────────────────────────────────────────────────┐
     │  EVALUATION HARNESS (offline)                                │
     │  baselines B0..B5 · metrics · paired bootstrap · cost/1k words│
     └──────────────────────────────────────────────────────────────┘
```

### 4.1 Ingestion

One `DocumentAdapter` protocol per format. Each adapter yields `Segment` objects and can reassemble
a translated document in the original format. Non-translatable structure (markup, placeholders,
footnote anchors, code spans) is replaced by typed placeholders before translation and restored
after.

The **round-trip invariant**, asserted by test: reassembling a document whose segments were never
translated reproduces every block's inner markup exactly. Per block rather than per file, because
whole-file byte equality additionally depends on the XML serializer — it would be a test of
BeautifulSoup, not of this pipeline. What matters is that span stitching and placeholder
restoration are lossless, and that is exactly what the per-block assertion proves. Segments carry
character offsets into their block, so the whitespace between sentences is the document's own.

Sentence segmentation with `pysbd` (rule-based, language-aware), spaCy only where linguistic
features are needed. Each segment carries a content hash for caching and deduplication.

### 4.2 Translation memory and terminology

- **TM**: `(source, target, lang_pair, domain, provenance, approved_by)` rows with an embedding of
  the source. Retrieval is hybrid: exact hash hit → fuzzy match (token-level edit distance, the
  classic CAT-tool behaviour) → semantic neighbours via vector search. An exact hit above the
  approval threshold short-circuits the LLM entirely, which is both a quality and a cost win.
- **Terminology base**: `(source_term, target_term, forbidden_variants, do_not_translate,
  case_sensitivity, notes, status)`. Terms are *extracted* on a first pass over the document
  (candidate mining over noun phrases + named entities, scored by document frequency and
  distinctiveness), proposed to the human, and then **enforced**: injected into the prompt, and
  verified post-hoc by a deterministic checker that fails the segment if a term is rendered wrong.
  Enforcement is what makes the claim testable; the prompt alone is not enforcement.

### 4.3 Context engineering

The book is never in the prompt. Per segment the orchestrator assembles the smallest sufficient
context: the neighbouring source sentences, top-k TM hits, the glossary entries whose source terms
actually occur in this segment, and a rolling document summary (characters/entities/register)
maintained incrementally. Prompt layout is fixed so the stable part can be cached:

`[tools] → [system: role + style guide + MQM rubric] → [cached glossary/TM block] → [volatile: the segment]`

The TM hits are **retrieved few-shot examples**: approved source/target pairs from this document or
domain, selected per segment rather than frozen in the prompt. Few-shot selection is therefore a
retrieval problem here, and the evaluation treats the number of injected examples as a swept
parameter rather than a constant somebody picked once.

### 4.4 Model layer

The LLM provider is **OpenAI**, reached through a thin internal interface
(`LLMProvider.translate / review / classify`) so that no pipeline code imports a vendor SDK
directly. The interface is not speculative abstraction: it is what makes the provider comparison in
§5 possible, and provider pricing and model names change faster than this document.

| Role | Model tier | Why |
|---|---|---|
| Translator / Reviser | Frontier tier | The main quality path; only the segments that need it get here |
| Reviewer (MQM annotation) | Mid tier | Judging is cheaper than generating, and a different model reduces self-agreement bias |
| Bulk term mining, classification | Small tier | High volume, low judgement |
| Baseline MT | NLLB-200-distilled-600M / OPUS-MT | Open, local, free, the honest floor to beat |
| Domain-adapted MT | LoRA fine-tune of the above (PEFT) | Demonstrates domain adaptation with before/after numbers |
| Quality estimation | Gradient-boosted regressor trained here | Enables cost routing; the features are the interesting part |

Exact model ids and prices are **pinned in `config/models.yaml` from OpenAI's official pricing page
at implementation time**, not copied from this document or from third-party rate-card blogs — the
line-up turns over every few months and the secondary sources disagree with each other.

API specifics that shape the design:

- **Structured output everywhere.** JSON Schema with `strict: true`, both for responses and for
  function/tool definitions. The reviewer returns typed MQM annotations, never prose to be
  regex-parsed. Every schema has a version, and a contract test per schema.
- **Prompt caching is automatic on a stable prefix**, so the prompt layout in §4.3 is a cost
  decision, not only a clarity one: frozen system prompt and rubric first, glossary/TM block next,
  the volatile segment last. Cached-token counts from the `usage` object are logged per call and
  asserted in the cost report — a zero cache-hit rate across repeated calls is a bug, not a detail.
- **Batch API** for whole-book runs: asynchronous, roughly half price, results keyed by request id.
  Full-document translation is a batch job, not a loop of live calls. The interactive path exists
  only for the reviewer UI and for single segments.
- **`reasoning_effort`** is the quality/cost dial per role; the evaluation sweeps it instead of
  guessing a default.
- **Reproducibility comes from the log, not from sampling parameters.** Every translation records
  `(provider, model id, prompt hash, schema version, effort, seed if supported)`. A `seed` is a
  best-effort hint on any provider and is treated as such.
- Token and cost accounting always come from the `usage` object the API returns, never from a
  local tokenizer estimate.
- Long-context surcharges exist on some tiers above a prompt-size threshold. Since §4.3 keeps
  prompts small by construction, the pipeline should never approach it — and a guard asserts that,
  so the design intent is enforced rather than assumed.

**Why the provider choice matters less than it looks.** The per-token sticker price is not the main
cost lever in this pipeline; three architectural decisions each move the bill by more than the gap
between vendors: exact TM hits that skip the model entirely, the cache hit rate on the stable
prefix, and the share of segments that QE routing ever sends to the frontier tier. If routing sends
15% of segments to the expensive model, a 2x difference in that model's price changes the total by
well under 15%. This is stated here so the final cost report measures it instead of asserting it.

### 4.5 Quality estimation and routing

A regressor predicts post-edit effort per segment from features that are cheap to compute:
COMET-Kiwi score, source/target length ratio, TM best-match score, count of glossary violations,
placeholder mismatch, NE count delta, source sentence length, reviewer MQM severity sum.
Trained on segments labelled by human post-edit distance plus COMET as a weak label.

Routing policy: high confidence → accept; medium → second pass with the expensive model;
low → queue for human review. The policy is tuned on a held-out set and reported as a
**cost/quality curve**, which is the point: a number showing how much money a quality point costs.

## 5. Evaluation protocol

This section is the project. Everything else serves it.

**Arms**

| ID | System |
|---|---|
| B0 | Open baseline MT, no context (NLLB/OPUS-MT) |
| B1 | LLM, segment in isolation, no glossary, no TM |
| B2 | LLM with document context, no glossary, no TM |
| B3 | Full pipeline: TM + enforced glossary + reviewer loop |
| B4 | Fine-tuned local model, in-domain |
| B5 | B3 with QE-based routing (the shipped configuration) |
| B6 | B5 with the frontier role served by a second provider, same prompts and schemas |

B6 exists because "provider X is cheaper" is a measurable claim, and this project is built to
measure claims. It reports quality *and* cost per 1,000 words for both providers on the same
in-domain test set, which is a more interesting result than either number alone.

**Metrics**

- Reference-based: `chrF++` and BLEU via sacreBLEU (signature reported), COMET-22, TER.
- Reference-free: COMET-Kiwi.
- Custom, and the ones that matter here:
  - **Terminology compliance rate** — share of glossary terms rendered per the term base.
  - **Intra-document consistency** — share of repeated source terms rendered identically.
  - **Placeholder integrity** — share of segments whose placeholders survive intact (localization).
- LLM-as-judge MQM score, calibrated against a human-annotated subset before being trusted.
- **Human post-edit distance (HTER)** on a 100-segment sample, edited by hand.
- **Cost per 1,000 words** and wall-clock per arm. Reported in the same table as quality.

**Significance**: paired bootstrap resampling between arms; no claim of improvement without it.
**Tracking**: MLflow — every run logs the dataset version, prompt hashes, model ids, effort level,
metrics, and cost. A result that cannot be reproduced from its run record does not count.

**The deliverable** is `reports/evaluation.md`: a report with the table, the cost/quality curve, an
error taxonomy from manual inspection of the worst segments, and an honest section on what the
pipeline still gets wrong.

## 6. Repository layout

```
scriptorium/
├── src/scriptorium/
│   ├── adapters/        # epub, pdf, po, json → Segment[]  (+ reassembly)
│   ├── segmentation/    # sentence splitting, placeholder protection
│   ├── memory/          # TM store, fuzzy + vector retrieval
│   ├── terminology/     # mining, store, enforcement checker
│   ├── context/         # per-segment context assembly, rolling summary
│   ├── orchestrator/    # agent loop, tools, schemas, prompts/
│   ├── models/          # baseline MT, LoRA training, QE model
│   ├── quality/         # metrics, MQM, judge calibration
│   ├── pipeline/        # batch jobs, routing policy
│   ├── api/             # FastAPI
│   └── audit/           # structured audit log, cost accounting
├── ui/                  # reviewer interface
├── data/                # DVC-tracked; nothing large in git
├── eval/                # test sets, harness, baselines
├── reports/             # evaluation.md, model-card.md, data-statement.md
├── docs/adr/            # architecture decision records
├── tests/
└── docker/
```

Prompts live in `orchestrator/prompts/` as versioned files, hashed and logged per call. A prompt
change is a diff in a pull request, not an edit in a notebook.

## 7. Phases

Each phase ends with something runnable and a number, not a checkbox.

### Phase 0 — Foundations (half a day)
Repo, `pyproject.toml`, ruff + mypy + pytest, GitHub Actions, licence and data policy, ADR 001
(scope and language pair). **Done when** CI is green on an empty project.

### Phase 1 — Spine and first measurement (1–2 weeks)
EPUB adapter with round-trip test, segmentation, placeholder protection, single-pass LLM
translation with structured output, the evaluation harness, arms B0/B1/B2 on a 500-segment test set.
**Done when** `make eval` on a fresh clone prints the metric table for B0–B2 with cost per 1k words,
and `reports/evaluation.md` has its first results section.

### Phase 2 — Memory, terminology and the agent loop (1–2 weeks)
TM store with hybrid retrieval, term mining and enforcement, the translate→review→revise loop with
real tools and MQM-typed output, FastAPI, reviewer UI with post-edit writeback, `.po` adapter.
**Done when** terminology compliance ≥ 98% and intra-document consistency ≥ 99% on the hard-case
set, and B3 beats B2 on COMET with paired bootstrap p < 0.05.

### Phase 3 — Trained models (2–3 weeks)
Sentence alignment of the book corpus (embedding + dynamic programming), dataset cleaning and
validation, DVC versioning, LoRA fine-tune of the baseline MT, QE model with feature engineering,
routing policy, MLflow for everything, the effort/model sweep, and the B6 provider comparison.
**Done when** B4 beats B0 by ≥ 2 chrF++ in-domain, the QE model reaches Pearson ≥ 0.5 against human
post-edit effort on held-out data, and B5 cuts cost ≥ 40% versus B3 for ≤ 1 COMET point.

### Phase 4 — Production posture (1 week)
Docker + compose, PII redaction before egress with a test proving it, audit log, secret handling,
eval regression gate in CI against recorded responses, deploy, model card, data statement, final
report, README with the architecture diagram and the headline numbers.
**Done when** a stranger can clone the repo, read the README, and understand both what it does and
how well it does it in under five minutes.

## 8. Non-functional requirements

- **Secrets** never in git; API credentials resolved from the environment, with `.env.example`
  documenting every variable the pipeline reads.
- **Egress control**: a redaction pass strips emails, phone numbers and configured patterns before
  any text leaves the process, with a unit test asserting it. Source documents and their licences
  are recorded; the pipeline refuses undeclared sources.
- **Audit log**: one row per translation — segment hash, model id, prompt hash, schema version,
  effort, token counts, cost, latency, who approved it. This is what makes the system auditable,
  and it is also the training data for the QE model.
- **Testing**: unit tests for segmentation, placeholder round-trip, fuzzy matching and the
  terminology checker; golden-file tests for adapters; schema contract tests for every structured
  output; recorded API responses so CI never spends money or needs network.
- **Observability**: cost and cache-hit-rate counters surfaced per job.
- **Ethical AI**: model card stating intended use, evaluated languages, known failure modes and
  measured bias on a gendered-profession probe set; data statement covering provenance, licence and
  consent for every corpus.

## 9. Risks

| Risk | Mitigation |
|---|---|
| PDF layout work consumes the schedule and teaches nothing | EPUB first; PDF is a bounded adapter in phase 2+, single-column prose only |
| No gold references → no credible evaluation | Reference sets are secured in phase 1, before any modelling work |
| LLM-as-judge is trusted without calibration | Judge is calibrated against human MQM annotations before use; disagreement is reported |
| Fine-tuning turns into a GPU rabbit hole | LoRA on a small model, one short run, fixed budget; the QE model is the fallback contribution |
| Scope creep across four phases | Each phase has a numeric exit criterion; a phase does not start until the previous one's number exists |
| Copyright on source texts | Public-domain and permissively licensed material only, enforced at ingestion |

## 10. Open decisions

- Vector store: `pgvector` (one Postgres for TM + metadata + vectors, simplest ops) vs. LanceDB
  (no server). Leaning Postgres, since one datastore is easier to defend than two.
- Orchestration: a framework's agent loop vs. a hand-written one. Leaning hand-written — the
  control flow is the interesting part and should be readable in the repository, and it keeps the
  provider interface thin enough for B6 to be a config change.
- Secondary language pair: decide at the start of phase 3, only if phase 2's numbers are in.
- EPUB navigation documents are currently extracted like any other content, so the table of
  contents becomes segments. For a translated book that is correct — a localized EPUB wants a
  localized TOC — but it mixes navigation text into a corpus meant to measure prose quality.
  Decision for phase 2: tag segments with the role of their container (`content` vs `navigation`)
  and let the evaluation filter, rather than dropping the nav outright.
