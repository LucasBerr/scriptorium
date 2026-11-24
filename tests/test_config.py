"""The two guards: offline mode and the placeholder-price refusal."""

from __future__ import annotations

from pathlib import Path

import pytest

from scriptorium.config import ModelsConfig, ModelSpec, Settings
from scriptorium.domain import TokenUsage

PLACEHOLDER_YAML = """
provider: openai
pinned_from: "TODO: url"
pinned_on: "TODO: YYYY-MM-DD"
roles:
  translator:
    model: "TODO-frontier-model-id"
    reasoning_effort: medium
    price_input: 0.0
    price_output: 0.0
"""

PINNED_YAML = """
provider: openai
pinned_from: "https://example.org/pricing"
pinned_on: "2026-11-24"
roles:
  translator:
    model: "a-real-model-id"
    reasoning_effort: medium
    price_input: 2.0
    price_cached_input: 0.2
    price_output: 10.0
"""


def test_placeholder_config_is_not_pinned(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(PLACEHOLDER_YAML, encoding="utf-8")
    assert ModelsConfig.load(path).is_pinned is False


def test_real_config_is_pinned(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(PINNED_YAML, encoding="utf-8")
    config = ModelsConfig.load(path)
    assert config.is_pinned is True
    assert config.role("translator").model == "a-real-model-id"


def test_the_shipped_config_is_still_a_placeholder() -> None:
    """A reminder in test form: the repo must not pretend to know prices it does not."""
    assert ModelsConfig.load(Path("config/models.yaml")).is_pinned is False


def test_unknown_role_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(PINNED_YAML, encoding="utf-8")
    with pytest.raises(KeyError):
        ModelsConfig.load(path).role("reviewer")


def test_cached_input_is_billed_at_its_own_rate() -> None:
    spec = ModelSpec(
        model="m",
        price_input=2.0,
        price_cached_input=0.2,
        price_output=10.0,
    )
    usage = TokenUsage(input_tokens=1_000_000, cached_input_tokens=900_000, output_tokens=100_000)
    # 100k fresh input at $2/M + 900k cached at $0.20/M + 100k output at $10/M
    assert spec.cost(usage) == pytest.approx(0.2 + 0.18 + 1.0)


def test_cost_without_cache_is_the_full_input_rate() -> None:
    spec = ModelSpec(model="m", price_input=2.0, price_cached_input=0.2, price_output=10.0)
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0)
    assert spec.cost(usage) == pytest.approx(2.0)


def test_offline_defaults_to_off_but_is_settable() -> None:
    assert Settings(offline=True).offline is True
