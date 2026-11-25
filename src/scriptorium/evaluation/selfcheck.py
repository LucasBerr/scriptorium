"""An offline self-check for the agentic arm.

Arms B1-B6 need a billed provider to say anything about a real model. This
module makes one claim that does not: that the *loop* recovers terminology
errors a provider makes, and that the harness can see the difference.

The construction is deliberate. A sloppy provider is derived from the test set
itself by swapping each required term form for a forbidden variant, so its
output is a correct translation with exactly one class of defect — the class the
term base exists to catch. Two arms then run over that same provider:

* **sloppy (no loop)** sends the defect straight through.
* **loop** applies the deterministic check, hands the finding to the reviewer,
  and spends one revision fixing it.

If terminology compliance does not rise between those two arms, the loop does
not do the thing this project claims it does, and that is worth knowing without
spending a cent. What this does *not* show is translation quality: the provider
is replaying references, so the string metrics are a ceiling, not a result.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from scriptorium.domain import GlossaryTerm
from scriptorium.evaluation.harness import TestCase
from scriptorium.segmentation.placeholders import protect
from scriptorium.terminology.enforcement import (
    TARGET_LANGUAGE,
    accepted_forms,
    applicable_terms,
    term_language,
)
from scriptorium.terminology.matching import count_occurrences, term_pattern


def _spoil(reference: str, terms: Sequence[GlossaryTerm]) -> str:
    """Replace one accepted term form with a forbidden variant, if we can."""
    for term in terms:
        if term.do_not_translate or not term.forbidden_variants:
            continue
        language = term_language(term, TARGET_LANGUAGE)
        for form in accepted_forms(term):
            if count_occurrences(
                reference, form, case_sensitive=term.case_sensitive, language=language
            ):
                pattern = term_pattern(form, case_sensitive=term.case_sensitive, language=language)
                return pattern.sub(term.forbidden_variants[0], reference, count=1)
    return reference


class SloppyRecordings(BaseModel):
    """Recordings for a provider that breaks terminology and can fix it on request.

    `spoiled` is reported rather than assumed: a test set whose references never
    contain a term with a forbidden variant cannot demonstrate anything here, and
    a self-check that quietly compares two identical arms is worse than none.
    """

    translations: dict[str, str]
    revisions: dict[str, str]
    spoiled: int


def sloppy_recordings(
    cases: Sequence[TestCase],
    glossary: Sequence[GlossaryTerm],
) -> SloppyRecordings:
    """Build recordings in protected space, keyed as the providers key them."""
    translations: dict[str, str] = {}
    revisions: dict[str, str] = {}
    spoiled = 0

    for case in cases:
        source = protect(case.source).text
        reference = protect(case.reference).text
        sloppy = _spoil(reference, applicable_terms(source, glossary))
        translations[source] = sloppy
        if sloppy != reference:
            revisions[sloppy] = reference
            spoiled += 1

    return SloppyRecordings(translations=translations, revisions=revisions, spoiled=spoiled)
