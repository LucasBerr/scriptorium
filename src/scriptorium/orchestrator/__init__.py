"""The agentic translation loop and the tools it calls."""

from scriptorium.orchestrator.loop import LoopOutcome, TranslationLoop
from scriptorium.orchestrator.tools import (
    GlossaryProposal,
    Toolbox,
    ToolCall,
    ToolDefinition,
    ToolInputError,
    UnknownToolError,
)

__all__ = [
    "GlossaryProposal",
    "LoopOutcome",
    "ToolCall",
    "ToolDefinition",
    "ToolInputError",
    "Toolbox",
    "TranslationLoop",
    "UnknownToolError",
]
