"""
Adapter interfaces for pluggable providers (embeddings, vector index, tagging).
These keep the core system decoupled from specific backends (local or remote).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class EmbeddingProvider(ABC):
    """Interface for embedding providers."""

    name: str
    dimension: int

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of texts."""

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text."""
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else np.zeros(self.dimension, dtype=np.float32)


class VectorIndex(ABC):
    """Interface for vector indexes."""

    @abstractmethod
    def upsert(self, item_id: str, embedding: np.ndarray) -> None:
        """Insert or update an embedding."""

    @abstractmethod
    def delete(self, item_id: str) -> None:
        """Remove an embedding."""

    @abstractmethod
    def search(self, embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Return (id, score) pairs."""


class Tagger(ABC):
    """Interface for automated tagging providers."""

    name: str

    @abstractmethod
    def suggest_tags(self, title: str, content: str, max_tags: int = 8) -> List[str]:
        """Return a list of suggested tags for a note."""


class InMemoryVectorIndex(VectorIndex):
    """Simple in-memory vector index (brute-force cosine)."""

    def __init__(self):
        self._embeddings: dict[str, np.ndarray] = {}

    def upsert(self, item_id: str, embedding: np.ndarray) -> None:
        if embedding is None:
            return
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        self._embeddings[item_id] = embedding.astype(np.float32)

    def delete(self, item_id: str) -> None:
        self._embeddings.pop(item_id, None)

    def search(self, embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        if embedding is None or not self._embeddings:
            return []
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        scores = []
        for item_id, vec in self._embeddings.items():
            score = float(np.dot(embedding, vec))
            scores.append((item_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
