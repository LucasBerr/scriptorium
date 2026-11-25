# ADR 0004: Inflection-aware term matching

**Status:** accepted
**Date:** 2026-11-24

## Context

Terminology compliance was measured by looking for the term base's stored target
form in the translation. The first run of the evaluation harness against a
terminology-heavy test set exposed what that costs in an inflected language: the
*reference translations* scored 90.9% compliance. Four of forty-four checks
failed, every one of them on a correct translation:

| source | reference | stored form | why it failed |
|---|---|---|---|
| Every **file** in that folder… | Todos os **arquivos**… | `arquivo` | plural |
| **Sign in** with your email… | **Entre** com seu… | `entrar` | conjugated |

This is a defect in the pipeline, not only in the report. The orchestrator's
deterministic check would have demanded a revision of a correct segment, burned
the revision budget on it, and then routed it to a human — turning roughly one
correct segment in eleven into review work for nothing. It would also have made
phase 2's exit criterion (compliance ≥ 98%) unreachable by any correct system,
which is the clearest possible sign that the instrument, not the system, was
wrong.

## Options

1. **Keep literal matching and accept a 91% ceiling.** Rejected: the number
   would measure Portuguese morphology rather than terminology, and the loop
   would act on it.
2. **Write references that avoid inflection.** Rejected outright. Tailoring the
   test set to the tool's blind spot is how an evaluation stops meaning
   anything.
3. **Stem both sides (Snowball).** Tried and rejected. It fixes plurals, but the
   English stemmer maps *settings* to *set*, which then matches "set the value"
   and reports compliance that is not there; and it leaves *entrar* / *Entre*
   unmatched anyway, because the Portuguese stemmer's minimum-stem rule declines
   to touch a word that short. Half the problem solved, a new class of false
   positive introduced.
4. **Generate the surface forms a term may take, by rule, per language.**
   Accepted.

## Decision

`terminology/morphology.py` expands a term into its legitimate surface forms:
English plurals and the `-s/-ed/-ing` verb forms, Portuguese plurals (including
the three `-ão` patterns) and, for infinitives, the present, preterite,
imperfect, imperative, gerund and participle. Matching accepts any of them.

Three properties of the design are deliberate:

* **Generated, not learned.** Every accepted form can be printed and argued
  with. `GlossaryTerm.inflected_forms` overrides the rules for words they get
  wrong, and setting it turns generation off for that term, so a person decides
  once and the decision is visible in the term base.
* **Over-generates rather than under-generates.** `botão` yields `botães` and
  `botãos` as well as `botões`, because spelling alone does not say which is
  right. A form that is not a word costs nothing unless it is a *different*
  word; a missing form causes a false violation. The error direction is
  therefore towards missing a violation, never towards inventing one.
* **Forbidden variants are matched literally.** This is the one place
  over-generation would hurt: the rules turn the verb `logar` into `logo`, an
  ordinary Portuguese adverb, and the check would fail correct translations. A
  variant that slips past the literal check still fails the accepted-form check,
  so the violation is reported either way — only its `detail` differs.

## Consequences

* The ceiling arm scores 100% compliance and 100% consistency, so phase 2's exit
  criteria are now measurable against an instrument that a correct translation
  can satisfy.
* The rendering ledger records occurrences under the term base's spelling rather
  than the inflection observed, so a plural is no longer counted as a second,
  disagreeing rendering of the same term.
* Compliance is now a slightly generous measure. The honest statement in any
  report is that it under-counts rather than over-counts violations, and the
  ADR is where the size of that concession is argued.
* This remains surface matching. A rendering nobody anticipated — a synonym
  outside the term base — is still invisible, and the fix for that is word
  alignment, which is phase 3's work and not a morphology problem.
