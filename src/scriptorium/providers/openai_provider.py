"""OpenAI provider.

Structured output is requested against a JSON Schema with `strict`, so a malformed
response is a validation error rather than a mistranslation that happens to parse.

Two deliberate choices:

- **Chat Completions, not the newer Responses surface.** The schema and usage
  shapes here are stable and well understood; moving to Responses is a contained
  follow-up behind this same class, and ADR 0002 is what makes that safe.
- **`translate_batch` is sequential for now.** The asynchronous Batch endpoint is
  roughly half price and is how whole documents are meant to be translated
  (PROJECT.md §4.4); wiring it is phase 1's last task, and the signature here is
  already the one it will implement, so nothing above this line changes.

This module has **not** been exercised against the live API in this commit: there
is no key in the development environment, and CI runs offline by design. The
integration test that covers it is marked `network` and skipped by default. Treat
the request shape as reviewed-but-unverified until that test has run once.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from scriptorium.config import ModelSpec, Settings
from scriptorium.domain import Review, TokenUsage
from scriptorium.prompting import (
    Prompt,
    build_review_prompt,
    build_revision_prompt,
    build_translation_prompt,
)
from scriptorium.providers.base import (
    OfflineError,
    ProviderResult,
    ProviderUnpinnedError,
    ReviewRequest,
    ReviewResult,
    RevisionRequest,
    TranslationPayload,
    TranslationRequest,
)
from scriptorium.providers.schema import strict_json_schema

# Generated from the Pydantic models rather than hand-written, so the schema the
# provider enforces and the model the code validates against cannot drift apart.
_TRANSLATION_SCHEMA: dict[str, Any] = strict_json_schema(TranslationPayload)
_REVIEW_SCHEMA: dict[str, Any] = strict_json_schema(Review)


class MissingCredentialError(RuntimeError):
    """Raised when no API key is available for a billed call."""


class OpenAIProvider:
    """Translate segments through OpenAI."""

    name = "openai"

    def __init__(self, spec: ModelSpec, settings: Settings) -> None:
        if settings.offline:
            msg = "offline mode is on; refusing to construct a billed provider"
            raise OfflineError(msg)
        if not spec.is_pinned:
            msg = (
                f"model id {spec.model!r} is still a placeholder. Pin real ids and prices in "
                "config/models.yaml from the provider's official pricing page first."
            )
            raise ProviderUnpinnedError(msg)
        if not settings.openai_api_key:
            msg = "OPENAI_API_KEY is not set"
            raise MissingCredentialError(msg)

        self._spec = spec
        self._client = OpenAI(api_key=settings.openai_api_key)

    def _complete(
        self,
        prompt: Prompt,
        *,
        schema: dict[str, Any],
        schema_name: str,
    ) -> tuple[dict[str, Any], ProviderResult]:
        """One structured-output request. Shared by all three roles.

        Returns the parsed payload and a :class:`ProviderResult` carrying the
        metering, so each role only has to say what it wanted from the payload.
        """
        extra: dict[str, Any] = {}
        if self._spec.reasoning_effort:
            extra["reasoning_effort"] = self._spec.reasoning_effort

        started = time.monotonic()
        response = self._client.chat.completions.create(
            model=self._spec.model,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
            **extra,
        )
        latency_ms = int((time.monotonic() - started) * 1000)

        content = response.choices[0].message.content or ""
        payload: dict[str, Any] = json.loads(content)
        usage = self._usage(response)

        meta = ProviderResult(
            text="",
            model=str(response.model),
            usage=usage,
            prompt_hash=prompt.prompt_hash,
            effort=self._spec.reasoning_effort,
            latency_ms=latency_ms,
            response_id=str(response.id),
            cost_usd=self._spec.cost(usage),
        )
        return payload, meta

    def translate(self, request: TranslationRequest) -> ProviderResult:
        prompt = build_translation_prompt(
            request.segment,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            style_guide=request.style_guide,
            glossary=request.glossary,
            examples=request.examples,
            context_before=request.context_before,
            context_after=request.context_after,
            document_summary=request.document_summary,
        )
        payload, meta = self._complete(
            prompt, schema=_TRANSLATION_SCHEMA, schema_name="translation"
        )
        parsed = TranslationPayload.model_validate(payload)
        return meta.model_copy(update={"text": parsed.translation})

    def review(self, request: ReviewRequest) -> ReviewResult:
        """Annotate a draft against the MQM typology."""
        prompt = build_review_prompt(
            request.segment,
            request.draft,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            glossary=request.glossary,
            automatic_findings=request.automatic_findings,
        )
        payload, meta = self._complete(prompt, schema=_REVIEW_SCHEMA, schema_name="review")
        return ReviewResult(review=Review.model_validate(payload), meta=meta)

    def revise(self, request: RevisionRequest) -> ProviderResult:
        prompt = build_revision_prompt(
            request.segment,
            request.draft,
            request.problems,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            glossary=request.glossary,
            established_renderings=request.established_renderings,
        )
        payload, meta = self._complete(
            prompt, schema=_TRANSLATION_SCHEMA, schema_name="translation"
        )
        parsed = TranslationPayload.model_validate(payload)
        return meta.model_copy(update={"text": parsed.translation})

    @staticmethod
    def _usage(response: Any) -> TokenUsage:
        """Read metering off the response.

        Cached input tokens are read explicitly: a cache hit rate of zero across
        repeated calls means the stable prefix is being invalidated somewhere, and
        that is a bug worth failing a cost report over.
        """
        raw = getattr(response, "usage", None)
        if raw is None:
            return TokenUsage()
        details = getattr(raw, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0)
        return TokenUsage(
            input_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
            cached_input_tokens=cached,
            output_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
        )

    def translate_batch(self, requests: Sequence[TranslationRequest]) -> list[ProviderResult]:
        return [self.translate(request) for request in requests]
