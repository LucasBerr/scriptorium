"""The audit log.

One JSON object per translation, appended. It serves three purposes at once,
which is why it is a first-class module and not a logging call:

1. **Reproducibility.** Sampling parameters are a weak guarantee on any provider,
   so what makes a result re-derivable is the record of exactly what was sent:
   provider, model, prompt hash, schema version, effort.
2. **Cost accounting.** Token counts and cache hit rate per call, which is what
   the evaluation report's cost column is computed from.
3. **Training data.** The quality-estimation model in phase 3 learns from these
   rows joined with human post-edits.

Source text is never written, only its hash. The log is therefore safe to
publish; the corpus is not necessarily (see docs/data-policy.md).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from scriptorium.domain import SCHEMA_VERSION, Translation


class AuditRecord(BaseModel):
    """What is known about one translation call."""

    segment_id: str
    segment_hash: str
    target_lang: str
    provider: str
    model: str
    prompt_hash: str
    schema_version: str = SCHEMA_VERSION
    effort: str | None = None
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    status: str = "machine"
    approved_by: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def cache_hit_rate(self) -> float:
        if self.input_tokens <= 0:
            return 0.0
        return self.cached_input_tokens / self.input_tokens

    @classmethod
    def from_translation(
        cls,
        translation: Translation,
        *,
        segment_hash: str,
        approved_by: str | None = None,
    ) -> AuditRecord:
        return cls(
            segment_id=translation.segment_id,
            segment_hash=segment_hash,
            target_lang=translation.target_lang,
            provider=translation.provider,
            model=translation.model,
            prompt_hash=translation.prompt_hash,
            schema_version=translation.schema_version,
            effort=translation.effort,
            input_tokens=translation.usage.input_tokens,
            cached_input_tokens=translation.usage.cached_input_tokens,
            output_tokens=translation.usage.output_tokens,
            cost_usd=translation.cost_usd,
            latency_ms=translation.latency_ms,
            status=translation.status.value,
            approved_by=approved_by,
            created_at=translation.created_at,
        )


class AuditLog:
    """Append-only JSONL audit log."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: AuditRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def read(self) -> Iterator[AuditRecord]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield AuditRecord.model_validate(json.loads(line))

    def total_cost_usd(self) -> float:
        return sum(record.cost_usd for record in self.read())

    def cache_hit_rate(self) -> float:
        """Aggregate cache hit rate.

        Zero across a run of repeated prompts means the stable prefix is being
        invalidated somewhere — a bug, not a detail, since the prompt layout in
        prompting.py exists precisely to avoid it.
        """
        records = list(self.read())
        total_input = sum(r.input_tokens for r in records)
        if total_input <= 0:
            return 0.0
        return sum(r.cached_input_tokens for r in records) / total_input
