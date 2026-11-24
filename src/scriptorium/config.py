"""Settings and the model/price pin file.

Two guards live here, and both exist because a plausible-looking number is worse
than a missing one:

- `SCRIPTORIUM_OFFLINE=1` forbids outbound model calls. CI and the test suite run
  with it set, so a test that would spend money fails loudly instead.
- A billed run refuses to start while `config/models.yaml` still contains `TODO`
  placeholders, because a cost report computed from placeholder prices is a lie
  that looks like a result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from scriptorium.domain import GlossaryTerm, TokenUsage

TODO_MARKER = "TODO"


class Settings(BaseSettings):
    """Runtime configuration, from the environment or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    models_config: Path = Field(
        default=Path("config/models.yaml"), alias="SCRIPTORIUM_MODELS_CONFIG"
    )
    audit_log: Path = Field(default=Path("outputs/audit.jsonl"), alias="SCRIPTORIUM_AUDIT_LOG")
    offline: bool = Field(default=False, alias="SCRIPTORIUM_OFFLINE")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")


class ModelSpec(BaseModel):
    """One role's model and its prices, in USD per million tokens."""

    model: str
    reasoning_effort: str | None = None
    price_input: float = 0.0
    price_cached_input: float = 0.0
    price_output: float = 0.0

    @property
    def is_pinned(self) -> bool:
        return TODO_MARKER not in self.model

    def cost(self, usage: TokenUsage) -> float:
        """Cost of one call. Cached input is billed at its own, lower rate."""
        fresh_input = max(0, usage.input_tokens - usage.cached_input_tokens)
        return (
            fresh_input * self.price_input
            + usage.cached_input_tokens * self.price_cached_input
            + usage.output_tokens * self.price_output
        ) / 1_000_000


class ModelsConfig(BaseModel):
    """The contents of `config/models.yaml`."""

    provider: str
    pinned_from: str = ""
    pinned_on: str = ""
    roles: dict[str, ModelSpec]

    @property
    def is_pinned(self) -> bool:
        """True when every role has a real model id and the pin is dated."""
        if TODO_MARKER in self.pinned_from or TODO_MARKER in self.pinned_on:
            return False
        return all(spec.is_pinned for spec in self.roles.values())

    def role(self, name: str) -> ModelSpec:
        try:
            return self.roles[name]
        except KeyError as exc:
            msg = f"no model configured for role {name!r}; known roles: {sorted(self.roles)}"
            raise KeyError(msg) from exc

    @classmethod
    def load(cls, path: Path) -> ModelsConfig:
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(
            {
                "provider": raw["provider"],
                "pinned_from": raw.get("pinned_from", ""),
                "pinned_on": raw.get("pinned_on", ""),
                "roles": raw["roles"],
            }
        )


def load_glossary(path: Path) -> list[GlossaryTerm]:
    """Load the term base from YAML.

    Phase 2 replaces this with a store that supports mining and approval
    workflow; the shape of a term does not change, so nothing above it does.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [GlossaryTerm.model_validate(entry) for entry in raw]
