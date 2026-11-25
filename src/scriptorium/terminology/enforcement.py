"""Terminology enforcement.

A prompt instruction is a preference. This is the guarantee: a deterministic
check that can fail a segment, run after the model has answered.

The same function backs the `terminology_compliance` metric, so what the pipeline
is told to satisfy and what the report measures cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from scriptorium.domain import GlossaryTerm
from scriptorium.terminology.matching import count_occurrences

# The declared language pair (ADR 0001). They are defaults rather than constants
# because the matcher needs to know which inflection rules apply, and a caller
# working on another pair must be able to say so instead of silently getting
# Portuguese morphology.
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "pt-BR"


class TerminologyViolation(BaseModel):
    """One place where the term base was not honoured."""

    source_term: str
    expected: str
    found: str
    detail: str
    index: int = 0
    """Segment position. Left at 0 by the enforcer; filled in by the metric."""


def required_form(term: GlossaryTerm) -> str:
    """The preferred form: what the model is told to use."""
    return term.source_term if term.do_not_translate else term.target_term


def term_language(term: GlossaryTerm, target_language: str | None) -> str | None:
    """Which inflection rules apply to a term's target forms.

    A do-not-translate term keeps its source spelling, so inflecting it with the
    target language's rules would be wrong; `inflected_forms` means a person has
    already said exactly which forms count, so the rules must not add more.
    """
    if term.do_not_translate or term.inflected_forms:
        return None
    return target_language


def accepted_forms(term: GlossaryTerm) -> tuple[str, ...]:
    """Every form that passes enforcement, preferred one first.

    A do-not-translate term accepts only itself: the point of marking it is that
    there is no alternative.
    """
    if term.do_not_translate:
        return (term.source_term,)
    if term.inflected_forms:
        return (term.target_term, *term.acceptable_variants, *term.inflected_forms)
    return (term.target_term, *term.acceptable_variants)


def applicable_terms(
    source: str,
    terms: Sequence[GlossaryTerm],
    *,
    language: str | None = SOURCE_LANGUAGE,
) -> list[GlossaryTerm]:
    """The terms that actually occur in this source segment.

    Callers pass *placeholder-protected* source text. A variable named `{folder}`
    is then an opaque token and does not count as an occurrence of "folder" —
    enforcing a translation of a variable name would be a permanent, unfixable
    violation.
    """
    return [
        term
        for term in terms
        if count_occurrences(
            source,
            term.source_term,
            case_sensitive=term.case_sensitive,
            language=language,
        )
        > 0
    ]


def enforce(
    source: str,
    target: str,
    terms: Sequence[GlossaryTerm],
    *,
    source_language: str | None = SOURCE_LANGUAGE,
    target_language: str | None = TARGET_LANGUAGE,
) -> list[TerminologyViolation]:
    """Check one translated segment against the term base.

    A term is checked only when it occurs in the source, in any inflected form.
    It fails if a forbidden variant appears literally, or if none of its accepted
    forms appears in any inflection. The forbidden
    variants are what make this mechanical: without something to rule out, "did
    it quietly use a synonym?" has no answer a machine can give.
    """
    violations: list[TerminologyViolation] = []

    for term in applicable_terms(source, terms, language=source_language):
        required = required_form(term)

        bad_variant = next(
            (
                variant
                for variant in term.forbidden_variants
                # Literal, not inflected. Inflecting a forbidden variant is the
                # one place over-generation would hurt: the rules turn the verb
                # "logar" into "logo", which is an ordinary Portuguese adverb,
                # and the check would then fail correct translations. A variant
                # that slips past this still fails the accepted-form check
                # below, so the violation is reported either way — only the
                # `detail` differs.
                if count_occurrences(target, variant, case_sensitive=term.case_sensitive) > 0
            ),
            None,
        )
        if bad_variant is not None:
            violations.append(
                TerminologyViolation(
                    source_term=term.source_term,
                    expected=required,
                    found=bad_variant,
                    detail="forbidden variant used",
                )
            )
            continue

        # Any accepted form satisfies enforcement. Narrowing the document to one
        # of them is consistency's job, not enforcement's, and keeping the two
        # separate is what lets the evaluation report them as separate numbers.
        if not any(
            count_occurrences(
                target,
                form,
                case_sensitive=term.case_sensitive,
                language=term_language(term, target_language),
            )
            > 0
            for form in accepted_forms(term)
        ):
            violations.append(
                TerminologyViolation(
                    source_term=term.source_term,
                    expected=required,
                    found="",
                    detail="no accepted target form present",
                )
            )

    return violations
