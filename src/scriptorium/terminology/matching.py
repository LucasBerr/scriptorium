"""Term matching primitives.

One implementation, used by the enforcer, the metric and the agent tools. If
these diverged, the pipeline would be optimising against a different target from
the one it is scored on — which is the quiet way a project ends up reporting
numbers that do not mean what they appear to.

Matching is optionally *inflection-aware*: given a language, a term matches any
of the surface forms `morphology.inflections` allows. Without a language it
matches literally. Both modes exist because they are needed in different places:
mining and the ledger work with forms already observed in text, while the
enforcer has to recognise *arquivos* as the term *arquivo*.
"""

from __future__ import annotations

import re
from functools import lru_cache

from scriptorium.terminology.morphology import inflections

# `\b` only behaves usefully when the term is made of word characters. A term
# containing markup or punctuation falls back to a substring match.
_WORD_LIKE = re.compile(r"^[\w\s'-]+$", re.UNICODE)

# Below this length, a token is left alone: the inflection rules would turn "de"
# into "des" and "os" into "oses", and short function words carry no term weight.
_MIN_INFLECTED_LENGTH = 3


def _token_alternation(token: str, language: str) -> str:
    forms = inflections(token, language)
    if len(forms) == 1:
        return re.escape(token)
    # Longest first, so "arquivos" is preferred over "arquivo" at the same
    # position and the count is one occurrence rather than a partial match.
    ordered = sorted(forms, key=lambda form: (-len(form), form))
    return "(?:" + "|".join(re.escape(form) for form in ordered) + ")"


@lru_cache(maxsize=4096)
def term_pattern(
    term: str,
    *,
    case_sensitive: bool = False,
    language: str | None = None,
) -> re.Pattern[str]:
    """Compile a matcher for `term`.

    Word boundaries matter: without them "button" matches inside "buttonhole" and
    the compliance metric reports occurrences that are not occurrences.

    With a `language`, each token of the term is widened to its inflected forms
    and the tokens are joined by flexible whitespace, so a multi-word term
    survives a line break.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    if not _WORD_LIKE.match(term):
        return re.compile(re.escape(term), flags)

    tokens = term.split()
    if language is None or not tokens:
        return re.compile(rf"\b{re.escape(term)}\b", flags)

    parts = [
        _token_alternation(token, language)
        if len(token) >= _MIN_INFLECTED_LENGTH
        else re.escape(token)
        for token in tokens
    ]
    return re.compile(rf"\b{r'\s+'.join(parts)}\b", flags)


def count_occurrences(
    text: str,
    term: str,
    *,
    case_sensitive: bool = False,
    language: str | None = None,
) -> int:
    pattern = term_pattern(term, case_sensitive=case_sensitive, language=language)
    return len(pattern.findall(text))


def occurs(
    text: str,
    term: str,
    *,
    case_sensitive: bool = False,
    language: str | None = None,
) -> bool:
    pattern = term_pattern(term, case_sensitive=case_sensitive, language=language)
    return pattern.search(text) is not None
