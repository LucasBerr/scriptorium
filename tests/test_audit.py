"""The audit log is the reproducibility mechanism, so its contents are asserted."""

from __future__ import annotations

from pathlib import Path

from scriptorium.audit import AuditLog, AuditRecord
from scriptorium.domain import TokenUsage, Translation, content_hash


def _translation(cost: float = 0.01, cached: int = 0) -> Translation:
    return Translation(
        segment_id="s1",
        target_lang="pt-BR",
        text="Pressione o botão.",
        provider="openai",
        model="a-model",
        prompt_hash="abc123",
        effort="medium",
        usage=TokenUsage(input_tokens=1000, cached_input_tokens=cached, output_tokens=50),
        cost_usd=cost,
        latency_ms=420,
    )


def test_append_and_read_round_trip(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    record = AuditRecord.from_translation(_translation(), segment_hash=content_hash("source"))
    log.append(record)

    read_back = list(log.read())
    assert len(read_back) == 1
    assert read_back[0].prompt_hash == "abc123"
    assert read_back[0].model == "a-model"


def test_source_text_is_never_written(tmp_path: Path) -> None:
    """The log must be publishable even when the corpus is not."""
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append(AuditRecord.from_translation(_translation(), segment_hash=content_hash("secret")))

    contents = path.read_text(encoding="utf-8")
    assert "secret" not in contents
    assert "Pressione" not in contents
    assert content_hash("secret") in contents


def test_total_cost_accumulates(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    for _ in range(3):
        log.append(AuditRecord.from_translation(_translation(cost=0.02), segment_hash="h"))
    assert log.total_cost_usd() == 0.06


def test_cache_hit_rate_is_aggregated(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(AuditRecord.from_translation(_translation(cached=900), segment_hash="h"))
    log.append(AuditRecord.from_translation(_translation(cached=100), segment_hash="h"))
    assert log.cache_hit_rate() == 0.5


def test_missing_log_reads_as_empty(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "nothing.jsonl")
    assert list(log.read()) == []
    assert log.total_cost_usd() == 0.0
    assert log.cache_hit_rate() == 0.0
