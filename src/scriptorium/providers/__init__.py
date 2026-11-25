"""LLM providers, behind one protocol.

No module outside this package imports a vendor SDK. That is what makes the
provider comparison in evaluation arm B6 a configuration change rather than a
rewrite — see ADR 0002.
"""

from scriptorium.providers.base import (
    AgenticProvider,
    LLMProvider,
    OfflineError,
    ProviderResult,
    Reviewer,
    ReviewRequest,
    ReviewResult,
    Reviser,
    RevisionRequest,
    TranslationRequest,
)
from scriptorium.providers.fake import EchoProvider, ScriptedProvider

__all__ = [
    "AgenticProvider",
    "EchoProvider",
    "LLMProvider",
    "OfflineError",
    "ProviderResult",
    "ReviewRequest",
    "ReviewResult",
    "Reviewer",
    "Reviser",
    "RevisionRequest",
    "ScriptedProvider",
    "TranslationRequest",
]
