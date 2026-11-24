"""Placeholder protection: keeping non-translatable spans intact.

Inline markup, printf and ICU placeholders, HTML entities and shell-style
variables must come out of a translation exactly as they went in. Asking a model
nicely is not a guarantee, so they are substituted for sentinel tokens before the
text is sent and restored afterwards. What the model sees is prose with opaque
tokens in it; what it cannot do is reorder a tag's attributes or translate `%s`.

This also has a useful side effect: because sentence splitting runs on the
protected text, a tag can never split a sentence.

The integrity report produced by :func:`check_integrity` is the raw material for
the `placeholder_integrity` evaluation metric.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from pydantic import BaseModel

from scriptorium.domain import Placeholder

OPEN_MARK = "⟦"
CLOSE_MARK = "⟧"

# Order matters: the first pattern to match at a position wins, so the more
# specific forms come first. `%(name)s` must be tried before bare printf.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("python_named", re.compile(r"%\([A-Za-z_][A-Za-z0-9_]*\)[0-9.\-+ #]*[sdifroxXeEgGc]")),
    # A space is a legal printf flag ("% d"), but allowing it here turns ordinary
    # prose into a false positive: "50% off" would match "% o" as octal. Prose is
    # the common case, so the space flag is deliberately not supported. The cost
    # of this trade-off is visible in the placeholder_integrity metric and in the
    # hard-case test set, which is where it belongs.
    (
        "printf",
        re.compile(r"%(?:\d+\$)?[-+#0]*\d*(?:\.\d+)?(?:hh|h|ll|l|L|z|j|t)?[diouxXeEfFgGaAcspn%]"),
    ),
    ("brace", re.compile(r"\{\{[^{}]*\}\}|\{[^{}]*\}")),
    ("tag", re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*?)?/?>")),
    ("entity", re.compile(r"&(?:[A-Za-z][A-Za-z0-9]{1,30}|#\d{1,7}|#[xX][0-9A-Fa-f]{1,6});")),
    ("shell_var", re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*")),
)


class PlaceholderCollisionError(ValueError):
    """Raised when the source text already contains the sentinel marks.

    Rather than escape them and complicate every downstream consumer, protection
    refuses the input. Real prose does not contain these code points; a document
    that does needs a different sentinel, chosen deliberately.
    """


class ProtectedText(BaseModel):
    """Text with non-translatable spans replaced by sentinel tokens."""

    text: str
    placeholders: tuple[Placeholder, ...] = ()

    def tokens(self) -> tuple[str, ...]:
        return tuple(p.token for p in self.placeholders)


class IntegrityReport(BaseModel):
    """What happened to the placeholders on the way through a model."""

    expected: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    duplicated: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()

    @property
    def is_intact(self) -> bool:
        return not (self.missing or self.duplicated or self.unknown)


def _token(index: int) -> str:
    return f"{OPEN_MARK}PH{index}{CLOSE_MARK}"


def _find_spans(text: str) -> list[tuple[int, int, str]]:
    """Locate non-translatable spans, left to right, without overlaps."""
    spans: list[tuple[int, int, str]] = []
    position = 0
    length = len(text)
    while position < length:
        for kind, pattern in _PATTERNS:
            match = pattern.match(text, position)
            if match is not None and match.end() > match.start():
                spans.append((match.start(), match.end(), kind))
                position = match.end()
                break
        else:
            position += 1
    return spans


def protect(text: str) -> ProtectedText:
    """Replace every non-translatable span in `text` with a sentinel token.

    Raises:
        PlaceholderCollisionError: if `text` already contains the sentinel marks.
    """
    if OPEN_MARK in text or CLOSE_MARK in text:
        msg = "source text already contains the placeholder sentinel marks"
        raise PlaceholderCollisionError(msg)

    spans = _find_spans(text)
    if not spans:
        return ProtectedText(text=text)

    pieces: list[str] = []
    placeholders: list[Placeholder] = []
    cursor = 0
    for index, (start, end, kind) in enumerate(spans):
        pieces.append(text[cursor:start])
        token = _token(index)
        pieces.append(token)
        placeholders.append(Placeholder(token=token, kind=kind, original=text[start:end]))
        cursor = end
    pieces.append(text[cursor:])
    return ProtectedText(text="".join(pieces), placeholders=tuple(placeholders))


def restore(text: str, placeholders: Iterable[Placeholder]) -> str:
    """Put the original spans back.

    Tokens absent from `text` are simply not restored; detecting that is the job
    of :func:`check_integrity`, which reports it rather than guessing a repair.
    """
    result = text
    for placeholder in placeholders:
        result = result.replace(placeholder.token, placeholder.original)
    return result


def check_integrity(expected: Sequence[Placeholder], translated: str) -> IntegrityReport:
    """Compare the tokens a translation should carry with the ones it does."""
    expected_tokens = tuple(p.token for p in expected)
    missing = tuple(t for t in expected_tokens if t not in translated)
    duplicated = tuple(t for t in expected_tokens if translated.count(t) > 1)

    found = set(re.findall(rf"{OPEN_MARK}PH\d+{CLOSE_MARK}", translated))
    unknown = tuple(sorted(found - set(expected_tokens)))

    return IntegrityReport(
        expected=expected_tokens,
        missing=missing,
        duplicated=duplicated,
        unknown=unknown,
    )
