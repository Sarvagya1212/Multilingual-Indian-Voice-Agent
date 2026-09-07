"""RAG retriever — orchestrates embedding, search, filtering, and re-ranking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.logger import setup_logger
from src.rag.chunker import TextChunk, TextChunker
from src.rag.config import RAGConfig
from src.rag.embedder import Embedder, get_embedder
from src.rag.reranker import Reranker
from src.rag.vector_store import VectorStore

logger = setup_logger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its relevance score and rank."""

    chunk: TextChunk
    score: float
    rank: int


class Retriever:
    """Combines chunking, embedding, vector search, and re-ranking.

    Pipeline:
    1. Index documents (chunk → embed → store)
    2. Retrieve (embed query → vector search → filter → rerank)
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        chunker: TextChunker | None = None,
        reranker: Reranker | None = None,
        config: RAGConfig | None = None,
    ):
        self.config = config or RAGConfig()
        self.embedder = embedder or get_embedder(
            provider=self.config.embedder_provider,
            dimension=self.config.embedder_dimension,
        )
        self.chunker = chunker or TextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size,
        )
        self.reranker = reranker or Reranker()
        self.vector_store = VectorStore()
        self._indexed = False

    async def index_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> None:
        """Index documents for retrieval.

        Args:
            documents: List of dicts with "content", "source", and optional "metadata".
        """
        all_chunks: list[TextChunk] = []

        for doc in documents:
            chunks = self.chunker.chunk_text(
                text=doc["content"],
                metadata=doc.get("metadata", {}),
                source=doc.get("source", "unknown"),
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("[RAG] No chunks to index")
            return

        # Batch embed
        texts = [chunk.text for chunk in all_chunks]
        vectors = await self.embedder.embed(texts)

        self.vector_store.add(all_chunks, vectors)
        self._indexed = True
        logger.info(f"[RAG] Indexed {len(all_chunks)} chunks from {len(documents)} docs")

    async def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query.

        Args:
            query: User query string.
            filters: Metadata filters to apply.
            top_k: Override default retrieval count.

        Returns:
            List of RetrievedChunk objects, descending by score.
        """
        if not self._indexed:
            logger.warning("[RAG] No documents indexed — returning empty results")
            return []

        k = top_k or self.config.top_k

        # Embed query
        query_vector = await self.embedder.embed_query(query)

        # Vector search
        raw_results = self.vector_store.search(query_vector, top_k=k)

        # Apply filters and min-score threshold
        filtered: list[tuple[TextChunk, float]] = []
        for chunk, score in raw_results:
            if filters:
                skip = False
                for key, value in filters.items():
                    if chunk.metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            if score < self.config.min_similarity_score:
                continue
            filtered.append((chunk, score))

        # Build RetrievedChunk list
        retrieved: list[RetrievedChunk] = []
        for rank, (chunk, score) in enumerate(filtered, start=1):
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                )
            )

        return retrieved

    async def retrieve_with_rerank(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        final_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve with re-ranking.

        Args:
            query: User query.
            filters: Metadata filters.
            top_k: How many to retrieve before reranking.
            final_k: How many to return after reranking.

        Returns:
            Re-ranked list of RetrievedChunk objects.
        """
        initial_k = top_k or self.config.rerank_top_k
        final = final_k or self.config.final_k

        # Initial retrieval (get more than final_k to allow reranking)
        raw = await self.retrieve(query, filters=filters, top_k=initial_k)

        if not raw:
            return []

        # Rerank
        raw_tuples = [(r.chunk, r.score) for r in raw]
        reranked = self.reranker.rerank(query, raw_tuples, top_k=final)

        return [
            RetrievedChunk(chunk=chunk, score=score, rank=i + 1)
            for i, (chunk, score) in enumerate(reranked)
        ]

    def build_context(
        self,
        chunks: list[RetrievedChunk],
        max_chars: int = 3000,
    ) -> str:
        """Build a context string from retrieved chunks.

        Args:
            chunks: Retrieved chunks.
            max_chars: Maximum total characters.

        Returns:
            Formatted context string with source attribution.
        """
        parts: list[str] = []
        total = 0

        for rc in chunks:
            chunk_text = rc.chunk.text.strip()
            source = rc.chunk.metadata.get("source", "unknown")

            if total + len(chunk_text) + 10 > max_chars:
                break

            parts.append(f"[Source: {source}]\n{chunk_text}")
            total += len(chunk_text) + 12

        return "\n\n---\n\n".join(parts) if parts else ""
