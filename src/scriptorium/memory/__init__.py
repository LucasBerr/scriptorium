"""Translation memory: the mechanism behind reuse, consistency and most of the savings."""

from scriptorium.memory.embedding import Embedder, HashingEmbedder
from scriptorium.memory.store import TMHit, TranslationMemory

__all__ = ["Embedder", "HashingEmbedder", "TMHit", "TranslationMemory"]
