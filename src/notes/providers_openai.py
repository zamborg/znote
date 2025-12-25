"""
OpenAI-backed providers for embeddings and tagging.
Designed to be lightweight and swappable via the adapter interfaces.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import requests

from .adapters import EmbeddingProvider, Tagger


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using OpenAI's embedding endpoint."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.name = f"openai:{model}"
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.dimension = 1536  # text-embedding-3-small output size

    def _headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIEmbeddingProvider")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []

        payload = {"model": self.model, "input": texts}
        resp = requests.post(f"{self.api_base}/embeddings", headers=self._headers(), json=payload)
        resp.raise_for_status()
        data = resp.json()

        embeddings = []
        for item in data.get("data", []):
            vec = np.array(item["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)

        return embeddings


class OpenAITagger(Tagger):
    """Tagging provider that prompts OpenAI for concise tags."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.name = f"openai:{model}"
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

    def _headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAITagger")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def suggest_tags(self, title: str, content: str, max_tags: int = 8) -> List[str]:
        prompt = (
            "You are a concise tag generator for a research notes system. "
            "Return a comma-separated list of lowercase tags (1-3 words each), "
            "no explanations. Max tags: {max_tags}.\n\n"
            f"Title: {title}\nContent:\n{content[:2000]}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You generate tags."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 64,
        }

        resp = requests.post(f"{self.api_base}/chat/completions", headers=self._headers(), json=payload)
        resp.raise_for_status()
        data = resp.json()
        content_text = data["choices"][0]["message"]["content"]
        tags = [t.strip().lower() for t in content_text.split(",")]
        tags = [t for t in tags if t]
        return tags[:max_tags]
