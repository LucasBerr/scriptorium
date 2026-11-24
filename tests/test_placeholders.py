"""Placeholder protection must be lossless, or reassembly silently corrupts documents."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scriptorium.segmentation.placeholders import (
    CLOSE_MARK,
    OPEN_MARK,
    PlaceholderCollisionError,
    check_integrity,
    protect,
    restore,
)


@pytest.mark.parametrize(
    ("text", "kinds"),
    [
        ('Click <a href="/x">here</a>.', ["tag", "tag"]),
        ("Delete %d files.", ["printf"]),
        ("Delete %1$s now.", ["printf"]),
        ("Hello %(name)s!", ["python_named"]),
        ("Move to {folder} or {{other}}.", ["brace", "brace"]),
        ("Caf&eacute; &#233; bom.", ["entity", "entity"]),
        ("Set $HOME and ${PATH}.", ["shell_var", "shell_var"]),
    ],
)
def test_protect_recognises_each_kind(text: str, kinds: list[str]) -> None:
    protected = protect(text)
    assert [p.kind for p in protected.placeholders] == kinds
    assert restore(protected.text, protected.placeholders) == text


def test_prose_percentages_are_not_mistaken_for_format_specifiers() -> None:
    """'50% off' must not be read as the octal conversion '% o'.

    The space printf flag is deliberately unsupported for exactly this reason.
    """
    text = "Almost 50% off, and 100% of readers agree."
    protected = protect(text)
    assert protected.placeholders == ()
    assert protected.text == text


def test_inline_markup_cannot_split_a_sentence() -> None:
    protected = protect("The <em>quick</em> fox. Next one.")
    assert OPEN_MARK in protected.text
    assert protected.text.count(".") == 2


def test_collision_is_refused_rather_than_escaped() -> None:
    with pytest.raises(PlaceholderCollisionError):
        protect(f"already has {OPEN_MARK}PH0{CLOSE_MARK} in it")


@settings(max_examples=300)
@given(
    st.text(
        alphabet=st.characters(blacklist_characters=OPEN_MARK + CLOSE_MARK),
        max_size=200,
    )
)
def test_round_trip_is_lossless_for_arbitrary_text(text: str) -> None:
    protected = protect(text)
    assert restore(protected.text, protected.placeholders) == text


def test_integrity_reports_a_dropped_placeholder() -> None:
    protected = protect("Delete %d files from {folder}.")
    translated = protected.text.replace(protected.placeholders[0].token, "")
    report = check_integrity(protected.placeholders, translated)
    assert not report.is_intact
    assert report.missing == (protected.placeholders[0].token,)


def test_integrity_reports_duplication_and_invention() -> None:
    protected = protect("Delete %d files.")
    token = protected.placeholders[0].token
    report = check_integrity(protected.placeholders, f"{token} {token} {OPEN_MARK}PH9{CLOSE_MARK}")
    assert report.duplicated == (token,)
    assert report.unknown == (f"{OPEN_MARK}PH9{CLOSE_MARK}",)


def test_intact_translation_passes() -> None:
    protected = protect("Delete %d files from {folder}.")
    assert check_integrity(protected.placeholders, protected.text).is_intact
