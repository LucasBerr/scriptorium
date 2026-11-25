"""The translation memory: reuse, consistency, and most of the savings."""

from __future__ import annotations

from pathlib import Path

import pytest

from scriptorium.domain import TMEntry
from scriptorium.memory import HashingEmbedder, TranslationMemory

PAIRS = [
    ("Press the button to continue.", "Pressione o botão para continuar."),
    ("Press the button to cancel.", "Pressione o botão para cancelar."),
    ("The file was moved to the folder.", "O arquivo foi movido para a pasta."),
]


@pytest.fixture
def memory() -> TranslationMemory:
    tm = TranslationMemory()
    tm.add_many(
        TMEntry(
            source_text=source,
            target_text=target,
            source_lang="en",
            target_lang="pt-BR",
            provenance="seed",
            approved_by="reviewer",
        )
        for source, target in PAIRS
    )
    return tm


# ------------------------------------------------------------------ embedder


def test_embedder_returns_unit_vectors() -> None:
    vectors = HashingEmbedder(dim=64).encode(["hello", "world"])
    assert vectors.shape == (2, 64)
    for vector in vectors:
        assert pytest.approx(float((vector**2).sum()), abs=1e-5) == 1.0


def test_embedder_is_deterministic_across_instances() -> None:
    """Stored vectors are reloaded later, so hashing must not be process-seeded."""
    first = HashingEmbedder(dim=64).encode(["consistency matters"])
    second = HashingEmbedder(dim=64).encode(["consistency matters"])
    assert (first == second).all()


def test_embedder_ranks_related_text_higher() -> None:
    embedder = HashingEmbedder()
    vectors = embedder.encode(
        [
            "Press the button to continue.",
            "Press the button to proceed.",
            "The weather is cold today.",
        ]
    )
    related = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])
    assert related > unrelated


def test_embedder_handles_empty_text_without_dividing_by_zero() -> None:
    vector = HashingEmbedder(dim=32).encode(["   "])[0]
    assert float((vector**2).sum()) == 0.0


@pytest.mark.parametrize(("dim", "ngrams"), [(0, (3, 5)), (16, (0, 3)), (16, (5, 3))])
def test_embedder_rejects_nonsense_configuration(dim: int, ngrams: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match=r"must be positive|invalid ngram_range"):
        HashingEmbedder(dim=dim, ngram_range=ngrams)


# --------------------------------------------------------------------- store


def test_exact_hit_short_circuits(memory: TranslationMemory) -> None:
    """The single largest cost lever: an approved exact hit needs no model."""
    hits = memory.search("Press the button to continue.", source_lang="en", target_lang="pt-BR")
    assert len(hits) == 1
    assert hits[0].is_exact
    assert hits[0].score == 1.0


def test_fuzzy_ranks_the_closer_neighbour_first(memory: TranslationMemory) -> None:
    hits = memory.search("Press the button to proceed.", source_lang="en", target_lang="pt-BR")
    assert [hit.kind for hit in hits] == ["fuzzy", "fuzzy"]
    assert hits[0].score > hits[1].score
    assert "cancel" in hits[0].entry.source_text


def test_unapproved_entries_are_excluded_by_default() -> None:
    tm = TranslationMemory()
    tm.add(
        TMEntry(
            source_text="Not approved.",
            target_text="Não aprovado.",
            source_lang="en",
            target_lang="pt-BR",
            provenance="machine",
        )
    )
    assert tm.search("Not approved.", source_lang="en", target_lang="pt-BR") == []
    assert tm.search("Not approved.", source_lang="en", target_lang="pt-BR", approved_only=False)


def test_the_wrong_language_pair_is_not_a_hit(memory: TranslationMemory) -> None:
    assert memory.search("Press the button to continue.", source_lang="en", target_lang="de") == []


def test_semantic_retrieval_runs_when_no_token_is_shared(memory: TranslationMemory) -> None:
    """No shared word, so fuzzy cannot fire; the embedding path must."""
    hits = memory.search(
        "Documents live inside directories.",
        source_lang="en",
        target_lang="pt-BR",
        min_semantic=0.01,
    )
    assert hits
    assert {hit.kind for hit in hits} == {"semantic"}
    # Ranked, best first.
    assert [hit.score for hit in hits] == sorted((h.score for h in hits), reverse=True)


def test_semantic_threshold_filters_distant_text(memory: TranslationMemory) -> None:
    """The documented weakness of a lexical embedder, asserted.

    "documents/directories" and "file/folder" are synonyms with almost no shared
    characters, so a character n-gram embedder scores them far apart. This is the
    gap a sentence embedder closes in phase 3.
    """
    assert (
        memory.search(
            "Documents live inside directories.",
            source_lang="en",
            target_lang="pt-BR",
            min_semantic=0.5,
        )
        == []
    )


def test_adding_the_same_source_replaces_rather_than_duplicates(
    memory: TranslationMemory,
) -> None:
    before = len(memory)
    memory.add(
        TMEntry(
            source_text="Press the button to continue.",
            target_text="Aperte o botão para continuar.",
            source_lang="en",
            target_lang="pt-BR",
            provenance="revised",
            approved_by="reviewer",
        )
    )
    assert len(memory) == before
    hit = memory.search("Press the button to continue.", source_lang="en", target_lang="pt-BR")[0]
    assert hit.entry.target_text == "Aperte o botão para continuar."


def test_post_edit_writeback_creates_an_approved_exact_hit() -> None:
    """The loop that makes the system improve rather than merely run."""
    tm = TranslationMemory()
    tm.record_post_edit(
        "Press the button.",
        "Pressione o botão.",
        source_lang="en",
        target_lang="pt-BR",
        approved_by="lucas",
    )
    hit = tm.search("Press the button.", source_lang="en", target_lang="pt-BR")[0]
    assert hit.is_exact
    assert hit.entry.approved_by == "lucas"
    assert hit.entry.provenance == "human post-edit"


def test_examples_returns_entries_for_few_shot(memory: TranslationMemory) -> None:
    examples = memory.examples(
        "Press the button to proceed.", source_lang="en", target_lang="pt-BR", k=2
    )
    assert len(examples) == 2
    assert all(entry.is_approved for entry in examples)


def test_k_limits_the_result_count(memory: TranslationMemory) -> None:
    hits = memory.search("Press the button to proceed.", source_lang="en", target_lang="pt-BR", k=1)
    assert len(hits) == 1


def test_memory_persists_to_disk_and_reloads(tmp_path: Path) -> None:
    path = tmp_path / "tm.sqlite"
    with TranslationMemory(path) as tm:
        tm.record_post_edit(
            "Save the file.",
            "Salve o arquivo.",
            source_lang="en",
            target_lang="pt-BR",
            approved_by="lucas",
        )

    with TranslationMemory(path) as reopened:
        assert len(reopened) == 1
        hit = reopened.search("Save the file.", source_lang="en", target_lang="pt-BR")[0]
        assert hit.entry.target_text == "Salve o arquivo."
        # Semantic retrieval must still work from vectors read back off disk.
        assert reopened.search(
            "Store the document.", source_lang="en", target_lang="pt-BR", min_semantic=0.01
        )


def test_empty_memory_returns_nothing() -> None:
    tm = TranslationMemory()
    assert tm.search("anything", source_lang="en", target_lang="pt-BR") == []
    assert len(tm) == 0
