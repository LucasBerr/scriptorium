"""Providers that make no network call.

`ScriptedProvider` is how CI exercises the pipeline end to end: it replays
recorded responses keyed by segment hash, so `make check` runs the real code
paths without a key and without spending money. An unknown segment is an error
rather than a plausible guess, which keeps a stale recording from passing as a
result.

`EchoProvider` returns the source unchanged. It is a degenerate arm, useful for
testing reassembly and for establishing the floor of every metric.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from scriptorium.domain import TokenUsage, content_hash
from scriptorium.providers.base import ProviderResult, TranslationRequest


class EchoProvider:
    """Returns the source text as the translation."""

    name = "echo"

    def __init__(self, model: str = "echo-v1") -> None:
        self.model = model

    def translate(self, request: TranslationRequest) -> ProviderResult:
        return ProviderResult(
            text=request.segment.text,
            model=self.model,
            usage=TokenUsage(),
            prompt_hash="echo",
        )

    def translate_batch(self, requests: Sequence[TranslationRequest]) -> list[ProviderResult]:
        return [self.translate(request) for request in requests]


class MissingRecordingError(KeyError):
    """Raised when a scripted provider has no recording for a segment."""


class ScriptedProvider:
    """Replays recorded translations, keyed by the hash of the source text."""

    name = "scripted"

    def __init__(self, recordings: Mapping[str, str], model: str = "scripted-v1") -> None:
        # Keyed by content hash rather than raw text so a recording file stays
        # readable and does not duplicate the corpus.
        self._by_hash = {content_hash(source): target for source, target in recordings.items()}
        self.model = model

    def translate(self, request: TranslationRequest) -> ProviderResult:
        key = content_hash(request.segment.text)
        try:
            text = self._by_hash[key]
        except KeyError as exc:
            msg = (
                f"no recording for segment {request.segment.id} (hash {key}). "
                "Re-record the fixture rather than relaxing this check."
            )
            raise MissingRecordingError(msg) from exc
        return ProviderResult(
            text=text,
            model=self.model,
            usage=TokenUsage(input_tokens=0, output_tokens=0),
            prompt_hash="scripted",
        )

    def translate_batch(self, requests: Sequence[TranslationRequest]) -> list[ProviderResult]:
        return [self.translate(request) for request in requests]
