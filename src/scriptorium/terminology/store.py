"""The term base and the rendering ledger."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import yaml

from scriptorium.domain import GlossaryTerm
from scriptorium.terminology.enforcement import (
    SOURCE_LANGUAGE,
    TARGET_LANGUAGE,
    TerminologyViolation,
    accepted_forms,
    applicable_terms,
    enforce,
    term_language,
)
from scriptorium.terminology.matching import count_occurrences


class TerminologyStore:
    """The approved term base for one language pair."""

    def __init__(self, terms: Sequence[GlossaryTerm] = ()) -> None:
        self._terms: list[GlossaryTerm] = list(terms)

    @classmethod
    def from_yaml(cls, path: Path) -> TerminologyStore:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        return cls([GlossaryTerm.model_validate(entry) for entry in raw])

    @property
    def terms(self) -> tuple[GlossaryTerm, ...]:
        return tuple(self._terms)

    def add(self, term: GlossaryTerm) -> None:
        self._terms = [t for t in self._terms if t.source_term != term.source_term]
        self._terms.append(term)

    def get(self, source_term: str) -> GlossaryTerm | None:
        return next((t for t in self._terms if t.source_term == source_term), None)

    def applicable(self, source: str) -> list[GlossaryTerm]:
        """Terms occurring in this (placeholder-protected) source segment."""
        return applicable_terms(source, self._terms)

    def enforce(self, source: str, target: str) -> list[TerminologyViolation]:
        return enforce(source, target, self._terms)


class RenderingLedger:
    """What each term has actually been rendered as, so far, in this document.

    This is the mechanism behind the consistency claim. The term base says what a
    term *should* be; the ledger records what it *has been*, which is what lets
    the orchestrator notice that segment 400 is about to disagree with segment 12
    and ask for a revision before the document is assembled.

    This is also why `GlossaryTerm` separates acceptable from forbidden variants.
    Enforcement accepts any acceptable form; the ledger then holds the document
    to whichever one it used first. Without that split, consistency would have
    nothing to add over enforcement, because every term would have exactly one
    legal rendering.

    Scope, stated plainly: renderings are detected from the known forms — the
    accepted ones plus the recorded forbidden variants. A rendering nobody
    anticipated is invisible here. That is why mining feeds the term base, and
    why the honest version of this needs word alignment (phase 3).
    """

    def __init__(
        self,
        terms: Sequence[GlossaryTerm] = (),
        *,
        source_language: str | None = SOURCE_LANGUAGE,
        target_language: str | None = TARGET_LANGUAGE,
    ) -> None:
        self._terms = list(terms)
        self._source_language = source_language
        self._target_language = target_language
        self._seen: dict[str, Counter[str]] = {}

    def observe(self, source: str, target: str) -> None:
        """Record which known form each applicable term took in this segment.

        Recorded under the term base's spelling of the form, not the inflection
        that appeared: *arquivos* is an occurrence of the rendering *arquivo*.
        Otherwise a plural would look like a second, disagreeing rendering and
        the consistency metric would report drift that is only grammar.
        """
        for term in applicable_terms(source, self._terms, language=self._source_language):
            counts = self._seen.setdefault(term.source_term, Counter())
            for form in accepted_forms(term):
                found = count_occurrences(
                    target,
                    form,
                    case_sensitive=term.case_sensitive,
                    language=term_language(term, self._target_language),
                )
                if found:
                    counts[form] += found
            for variant in term.forbidden_variants:
                found = count_occurrences(target, variant, case_sensitive=term.case_sensitive)
                if found:
                    counts[variant] += found

    def renderings(self, source_term: str) -> dict[str, int]:
        return dict(self._seen.get(source_term, Counter()))

    def established(self, source_term: str) -> str | None:
        """The form this term has most often taken, if it has been seen at all.

        The orchestrator uses this to keep later segments agreeing with earlier
        ones even when the term base allows more than one acceptable form.
        """
        counts = self._seen.get(source_term)
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    def disagreements(self) -> dict[str, dict[str, int]]:
        """Terms that have been rendered more than one way."""
        return {term: dict(counts) for term, counts in self._seen.items() if len(counts) > 1}
