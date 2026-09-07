"""In-memory vector store with cosine similarity search."""
from __future__ import annotations

import numpy as np

from src.rag.chunker import TextChunk


class VectorStore:
    """Simple in-memory vector store.

    Stores chunks and their embeddings, supports cosine similarity search.
    Uses float32 for memory efficiency.
    """

    def __init__(self):
        self.chunks: list[TextChunk] = []
        self.vectors: np.ndarray | None = None

    def add(self, chunks: list[TextChunk], vectors: np.ndarray) -> None:
        """Add chunks and their embeddings.

        Args:
            chunks: List of TextChunk objects.
            vectors: ndarray of shape (len(chunks), dimension).
        """
        assert len(chunks) == len(vectors), (
            f"Chunk count ({len(chunks)}) != vector count ({len(vectors)})"
        )

        self.chunks.extend(chunks)

        if self.vectors is None:
            self.vectors = vectors.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[TextChunk, float]]:
        """Find the top-k most similar chunks to a query vector.

        Args:
            query_vector: Query embedding (dimension,).
            top_k: Number of results to return.

        Returns:
            List of (TextChunk, similarity_score) tuples, descending by score.
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []

        similarities = self._cosine_similarity(
            query_vector.astype(np.float32),
            self.vectors,
        )

        # Descending order
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (self.chunks[i], float(similarities[i]))
            for i in top_indices
            if similarities[i] >= 0
        ]

    def _cosine_similarity(
        self,
        query: np.ndarray,
        vectors: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between query and all vectors."""
        norm_q = np.linalg.norm(query)
        norm_v = np.linalg.norm(vectors, axis=1)

        # Guard against zero vectors
        if norm_q == 0 or np.any(norm_v == 0):
            return np.zeros(len(vectors))

        return np.dot(vectors, query) / (norm_v * norm_q)

    def __len__(self) -> int:
        return len(self.chunks)
