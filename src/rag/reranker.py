"""Document re-ranking for RAG.

Re-rankers improve over initial vector retrieval by considering:
- Keyword overlap with the query
- Length normalization (penalize very short/long chunks)
- Position bias (sometimes earlier chunks are more authoritative)
"""
from __future__ import annotations

import re

from src.rag.chunker import TextChunk


class Reranker:
    """Lightweight lexical re-ranker.

    Computes a combined score from:
    1. Base similarity score (from vector search)
    2. BM25-style keyword overlap
    3. Length normalization
    """

    def __init__(
        self,
        keyword_weight: float = 0.3,
        length_weight: float = 0.1,
    ):
        self.keyword_weight = keyword_weight
        self.length_weight = length_weight

    def rerank(
        self,
        query: str,
        results: list[tuple[TextChunk, float]],
        top_k: int = 3,
    ) -> list[tuple[TextChunk, float]]:
        """Re-rank retrieved chunks.

        Args:
            query: Original query.
            results: List of (chunk, base_score) from vector search.
            top_k: Number of final results to return.

        Returns:
            Re-ranked list of (chunk, new_score) tuples.
        """
        if not results:
            return []

        query_tokens = self._tokenize(query)
        query_token_set = set(query_tokens)

        scored: list[tuple[TextChunk, float]] = []
        for chunk, base_score in results:
            keyword_score = self._keyword_overlap(query_token_set, chunk.text)
            length_score = self._length_score(chunk.text)

            combined = (
                (1 - self.keyword_weight - self.length_weight) * base_score
                + self.keyword_weight * keyword_score
                + self.length_weight * length_score
            )
            scored.append((chunk, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]

    def _keyword_overlap(self, query_tokens: set[str], chunk_text: str) -> float:
        """Fraction of query tokens that appear in the chunk."""
        if not query_tokens:
            return 0.0
        chunk_tokens = set(self._tokenize(chunk_text))
        matches = query_tokens & chunk_tokens
        return len(matches) / len(query_tokens)

    def _length_score(self, text: str) -> float:
        """Prefer chunks of moderate length (200-1000 chars)."""
        n = len(text)
        if n < 50:
            return 0.2
        if 200 <= n <= 1000:
            return 1.0
        if n < 200:
            return 0.5 + (n - 50) / 150 * 0.5
        # Penalize long chunks
        return max(0.3, 1.0 - (n - 1000) / 2000)
