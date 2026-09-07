"""Text embedding models — abstract base and concrete implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class Embedder(ABC):
    """Abstract base class for text embedding models."""

    dimension: int

    @abstractmethod
    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into vectors.

        Args:
            texts: List of text strings.

        Returns:
            ndarray of shape (len(texts), dimension).
        """
        raise NotImplementedError

    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            query: Query text.

        Returns:
            ndarray of shape (dimension,).
        """
        raise NotImplementedError


class OpenAIEmbedder(Embedder):
    """OpenAI text-embedding-3-small (1536d) or text-embedding-3-large (3072d)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 100,
    ):
        import os

        from openai import OpenAI

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.dimension = 1536 if "3-small" in model else 3072

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using OpenAI's embedding API."""
        import asyncio

        texts = [t[: 8000] for t in texts]  # Truncate very long texts
        vectors = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_vectors = [item.embedding for item in response.data]
            vectors.extend(batch_vectors)
            if i + self.batch_size < len(texts):
                await asyncio.sleep(0.1)  # Rate limiting

        return np.array(vectors, dtype=np.float32)

    async def embed_query(self, query: str) -> np.ndarray:
        vecs = await self.embed([query])
        return vecs[0]


class LocalEmbedder(Embedder):
    """Local TF-IDF-style embedder using hashed word vectors.

    Serves as a fallback when no API key is available. Produces
    fixed-dimension vectors based on word hashes — useful for testing
    but not competitive for production quality retrieval.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using hashed word-frequency vectors."""
        vectors = []
        for text in texts:
            words = text.lower().split()
            vec = np.zeros(self.dimension, dtype=np.float32)

            word_counts: dict[str, float] = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            # TF-IDF-like: hash word to dimension, weight by count
            for word, count in word_counts.items():
                idx = hash(word) % self.dimension
                vec[idx] += count / max(len(words), 1)

            # L2-normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm

            vectors.append(vec)

        return np.array(vectors, dtype=np.float32)

    async def embed_query(self, query: str) -> np.ndarray:
        vecs = await self.embed([query])
        return vecs[0]


def get_embedder(
    provider: str = "local",
    **kwargs,
) -> Embedder:
    """Factory for embedder providers."""
    providers = {
        "openai": OpenAIEmbedder,
        "local": LocalEmbedder,
    }
    cls = providers.get(provider)
    if cls is None:
        raise ValueError(f"Unknown embedder provider: {provider}")
    return cls(**kwargs)
