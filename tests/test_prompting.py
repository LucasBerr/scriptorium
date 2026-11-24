"""Prompt layout is a cost decision, so it is asserted rather than assumed."""

from __future__ import annotations

import pytest

from scriptorium.domain import GlossaryTerm, Segment, StructureKind, TMEntry
from scriptorium.prompting import build_translation_prompt


@pytest.fixture
def segment() -> Segment:
    return Segment(
        id="s1",
        document_id="d1",
        block_key="b1",
        index=0,
        text="Press the button.",
        structure=StructureKind.PARAGRAPH,
        span_start=0,
        span_end=17,
    )


def _prompt(segment: Segment, **kwargs: object) -> object:
    return build_translation_prompt(
        segment,
        source_lang="en",
        target_lang="pt-BR",
        **kwargs,  # type: ignore[arg-type]
    )


def test_the_segment_comes_last(segment: Segment, glossary: list[GlossaryTerm]) -> None:
    """Everything stable must precede the only part that changes per call."""
    prompt = build_translation_prompt(
        segment,
        source_lang="en",
        target_lang="pt-BR",
        glossary=glossary,
        document_summary="A manual.",
    )
    user = prompt.user
    assert user.index("TERMINOLOGY") < user.index("SEGMENT TO TRANSLATE")
    assert user.index("DOCUMENT SUMMARY") < user.index("SEGMENT TO TRANSLATE")
    assert user.rstrip().endswith(segment.text)


def test_glossary_is_in_the_cacheable_part(segment: Segment, glossary: list[GlossaryTerm]) -> None:
    prompt = build_translation_prompt(
        segment, source_lang="en", target_lang="pt-BR", glossary=glossary
    )
    assert "TERMINOLOGY" in prompt.cacheable
    assert "TERMINOLOGY" not in prompt.volatile


def test_forbidden_variants_reach_the_model(segment: Segment, glossary: list[GlossaryTerm]) -> None:
    prompt = build_translation_prompt(
        segment, source_lang="en", target_lang="pt-BR", glossary=glossary
    )
    assert "never: botao, tecla" in prompt.cacheable


def test_do_not_translate_is_rendered_distinctly(segment: Segment) -> None:
    term = GlossaryTerm(source_term="SAP", target_term="SAP", do_not_translate=True)
    prompt = build_translation_prompt(
        segment, source_lang="en", target_lang="pt-BR", glossary=[term]
    )
    assert "SAP -> DO NOT TRANSLATE" in prompt.cacheable


def test_examples_are_retrieved_few_shot(segment: Segment) -> None:
    entry = TMEntry(
        source_text="Press the key.",
        target_text="Pressione a tecla.",
        source_lang="en",
        target_lang="pt-BR",
        provenance="test",
    )
    prompt = build_translation_prompt(
        segment, source_lang="en", target_lang="pt-BR", examples=[entry]
    )
    assert "APPROVED EXAMPLES" in prompt.cacheable
    assert "Pressione a tecla." in prompt.cacheable


def test_language_pair_reaches_the_system_prompt(segment: Segment) -> None:
    prompt = build_translation_prompt(segment, source_lang="en", target_lang="pt-BR")
    assert "en" in prompt.system
    assert "pt-BR" in prompt.system


def test_template_hash_is_stable_but_prompt_hash_is_not(segment: Segment) -> None:
    """The template is versioned; the rendered prompt is per segment."""
    other = segment.model_copy(update={"text": "Press the other button."})
    first = build_translation_prompt(segment, source_lang="en", target_lang="pt-BR")
    second = build_translation_prompt(other, source_lang="en", target_lang="pt-BR")
    assert first.template_hash == second.template_hash
    assert first.prompt_hash != second.prompt_hash


def test_prompt_hash_is_reproducible(segment: Segment) -> None:
    first = build_translation_prompt(segment, source_lang="en", target_lang="pt-BR")
    second = build_translation_prompt(segment, source_lang="en", target_lang="pt-BR")
    assert first.prompt_hash == second.prompt_hash
