"""Surface inflection, so a term base can be checked against real prose.

The problem this solves is the reason naive terminology checking does not work
for Portuguese. The term base says *arquivo*; a correct translation says
*arquivos*. It says *entrar*; the translation says *Entre*. Matching the stored
form literally reports violations against translations a reviewer would approve,
and a compliance metric whose ceiling is 91% is not a metric — it is a pipeline
that sends one correct segment in eleven to a human for no reason.

So each term is expanded to the surface forms it can legitimately take, and the
check accepts any of them. The expansion is rule-based and deliberately plain:

* It is **generated, not learned**, so every accepted form can be printed and
  argued with. `GlossaryTerm.inflected_forms` overrides the rules for the words
  they get wrong, which is how irregulars are handled — by a person, once.
* It **over-generates rather than under-generates**. A form that is not really a
  word costs nothing unless it happens to be a different word; a form that is
  missing causes a false violation. The error direction is therefore towards
  missing a violation, not towards inventing one, and that is the trade this
  module makes on purpose. `docs/adr/0004-terminology-morphology.md` records it.
* It stops at **nouns, adjectives and verbs**, because those are what a term base
  contains. There is no attempt at a full morphological analyser.

Multi-word terms are expanded head-wise: every token is expanded, and the match
is a token sequence (see `matching.py`). "banco de dados" therefore also matches
"bancos de dados".
"""

from __future__ import annotations

from functools import lru_cache

# ----------------------------------------------------------------- English

_EN_IRREGULAR_PLURALS = {
    "child": "children",
    "foot": "feet",
    "man": "men",
    "person": "people",
    "tooth": "teeth",
    "woman": "women",
}

_EN_VOWELS = "aeiou"


def _english_noun_plural(word: str) -> set[str]:
    if word in _EN_IRREGULAR_PLURALS:
        return {_EN_IRREGULAR_PLURALS[word]}
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return {word + "es"}
    if word.endswith("y") and len(word) > 1 and word[-2] not in _EN_VOWELS:
        return {word[:-1] + "ies"}
    if word.endswith("f"):
        return {word + "s", word[:-1] + "ves"}
    if word.endswith("fe"):
        return {word + "s", word[:-2] + "ves"}
    return {word + "s"}


def _english_verb_forms(word: str) -> set[str]:
    """Third person, past and gerund. Enough for imperative-heavy UI text."""
    forms = set(_english_noun_plural(word))  # the -s/-es rules coincide
    if word.endswith("e"):
        forms |= {word + "d", word[:-1] + "ing"}
    elif word.endswith("y") and len(word) > 1 and word[-2] not in _EN_VOWELS:
        forms |= {word[:-1] + "ied", word + "ing"}
    else:
        forms |= {word + "ed", word + "ing"}
    return forms


def _english(word: str) -> set[str]:
    return {word} | _english_noun_plural(word) | _english_verb_forms(word)


# -------------------------------------------------------------- Portuguese

_PT_VERB_ENDINGS = {
    # Infinitive ending -> the finite and participial endings that replace it.
    # Not the full paradigm: the forms a product text actually uses, which is
    # present, preterite, imperfect, the imperative, the gerund and the
    # participle (inflected for gender and number, because it agrees).
    "ar": (
        "o",
        "a",
        "amos",
        "am",
        "ei",
        "ou",
        "amos",
        "aram",
        "ava",
        "avam",
        "e",
        "em",
        "ando",
        "ado",
        "ada",
        "ados",
        "adas",
        "ar",
    ),
    "er": (
        "o",
        "e",
        "emos",
        "em",
        "i",
        "eu",
        "emos",
        "eram",
        "ia",
        "iam",
        "a",
        "am",
        "endo",
        "ido",
        "ida",
        "idos",
        "idas",
        "er",
    ),
    "ir": (
        "o",
        "e",
        "imos",
        "em",
        "i",
        "iu",
        "imos",
        "iram",
        "ia",
        "iam",
        "a",
        "am",
        "indo",
        "ido",
        "ida",
        "idos",
        "idas",
        "ir",
    ),
}


def _portuguese_plural(word: str) -> set[str]:
    if word.endswith("ão"):
        # Three real patterns (-ões, -ães, -ãos) and no way to pick from the
        # spelling alone, so all three are generated. `inflected_forms` is how a
        # term that needs exactly one of them says so.
        stem = word[:-2]
        return {stem + "ões", stem + "ães", stem + "ãos"}
    if word.endswith(("r", "z")):
        return {word + "es"}
    if word.endswith("m"):
        return {word[:-1] + "ns"}
    if word.endswith("al"):
        return {word[:-2] + "ais"}
    if word.endswith("el"):
        return {word[:-2] + "éis"}
    if word.endswith("il"):
        return {word[:-2] + "is"}
    if word.endswith("ol"):
        return {word[:-2] + "óis"}
    if word.endswith("ul"):
        return {word[:-2] + "uis"}
    if word.endswith("s"):
        return {word}
    return {word + "s"}


def _portuguese(word: str) -> set[str]:
    forms = {word} | _portuguese_plural(word)
    for ending, suffixes in _PT_VERB_ENDINGS.items():
        if word.endswith(ending) and len(word) > len(ending) + 1:
            stem = word[: -len(ending)]
            forms |= {stem + suffix for suffix in suffixes}
            break
    return forms


_LANGUAGES = {
    "en": _english,
    "english": _english,
    "pt": _portuguese,
    "pt-br": _portuguese,
    "portuguese": _portuguese,
}


@lru_cache(maxsize=8192)
def inflections(word: str, language: str) -> frozenset[str]:
    """Every surface form `word` is allowed to take in `language`.

    An unsupported language yields the word itself. That is the honest default:
    this project claims EN -> PT-BR, and silently applying Portuguese rules to
    German would produce confident nonsense.
    """
    expand = _LANGUAGES.get(language.lower())
    if expand is None or not word:
        return frozenset({word})
    return frozenset(expand(word.lower()))
