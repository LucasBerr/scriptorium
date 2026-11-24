"""Provider contract and the guards that keep CI from spending money."""

from __future__ import annotations

import pytest

from scriptorium.config import ModelSpec, Settings
from scriptorium.domain import Segment, StructureKind
from scriptorium.providers import EchoProvider, ScriptedProvider
from scriptorium.providers.base import (
    LLMProvider,
    OfflineError,
    ProviderUnpinnedError,
    TranslationPayload,
    TranslationRequest,
)
from scriptorium.providers.fake import MissingRecordingError
from scriptorium.providers.openai_provider import OpenAIProvider


def _request(text: str = "Press the button.") -> TranslationRequest:
    return TranslationRequest(
        segment=Segment(
            id="s1",
            document_id="d1",
            block_key="b1",
            index=0,
            text=text,
            structure=StructureKind.PARAGRAPH,
            span_start=0,
            span_end=len(text),
        ),
        source_lang="en",
        target_lang="pt-BR",
    )


def test_fakes_satisfy_the_protocol() -> None:
    assert isinstance(EchoProvider(), LLMProvider)
    assert isinstance(ScriptedProvider({}), LLMProvider)


def test_echo_returns_the_source() -> None:
    result = EchoProvider().translate(_request())
    assert result.text == "Press the button."
    assert result.cost_usd == 0.0


def test_scripted_replays_a_recording() -> None:
    provider = ScriptedProvider({"Press the button.": "Pressione o botão."})
    assert provider.translate(_request()).text == "Pressione o botão."


def test_scripted_refuses_to_guess() -> None:
    """A stale recording must fail loudly, not pass as a result."""
    provider = ScriptedProvider({"Something else.": "Outra coisa."})
    with pytest.raises(MissingRecordingError):
        provider.translate(_request())


def test_batch_preserves_order() -> None:
    provider = ScriptedProvider({"A.": "A-pt", "B.": "B-pt"})
    results = provider.translate_batch([_request("A."), _request("B.")])
    assert [r.text for r in results] == ["A-pt", "B-pt"]


def test_offline_mode_refuses_to_build_a_billed_provider() -> None:
    with pytest.raises(OfflineError):
        OpenAIProvider(ModelSpec(model="a-real-model"), Settings(offline=True))


def test_placeholder_model_id_refuses_to_build() -> None:
    with pytest.raises(ProviderUnpinnedError):
        OpenAIProvider(
            ModelSpec(model="TODO-frontier-model-id"),
            Settings(offline=False, openai_api_key="sk-test"),
        )


def test_structured_payload_rejects_a_missing_translation() -> None:
    with pytest.raises(ValueError, match="translation"):
        TranslationPayload.model_validate({"notes": "no translation field"})


def test_structured_payload_accepts_the_expected_shape() -> None:
    payload = TranslationPayload.model_validate({"translation": "Olá.", "notes": ""})
    assert payload.translation == "Olá."
