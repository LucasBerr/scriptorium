"""Sentence spans must index back into the text they came from."""

from __future__ import annotations

from itertools import pairwise

from scriptorium.segmentation.splitter import split_sentences


def test_spans_slice_back_to_the_source() -> None:
    text = "Dr. Smith arrived. He said hello! Then he left."
    for span in split_sentences(text, "en"):
        assert text[span.start : span.end] == span.text


def test_spans_are_contiguous_and_cover_the_text() -> None:
    text = "One sentence. Two sentences.  Three."
    spans = split_sentences(text, "en")
    assert spans[0].start == 0
    for earlier, later in pairwise(spans):
        assert earlier.end == later.start


def test_abbreviations_do_not_end_a_sentence() -> None:
    spans = split_sentences("Dr. Smith met Mr. Jones in the U.S. yesterday.", "en")
    assert len(spans) == 1


def test_unsupported_language_falls_back_instead_of_failing() -> None:
    # Portuguese is not in pysbd's language set; the fallback splitter handles it.
    spans = split_sentences("Primeira frase. Segunda frase!", "pt-BR")
    assert len(spans) == 2


def test_empty_text_yields_no_segments() -> None:
    assert split_sentences("   \n  ", "en") == []
