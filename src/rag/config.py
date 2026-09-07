"""RAG configuration."""
from pydantic import BaseModel


class RAGConfig(BaseModel):
    """Configuration for the RAG pipeline."""

    embedder_provider: str = "local"  # local, openai
    embedder_model: str = "text-embedding-3-small"
    embedder_dimension: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 50  # Below this, chunks are dropped
    top_k: int = 5
    rerank_top_k: int = 10
    final_k: int = 3
    min_similarity_score: float = 0.0  # 0 = accept all results; re-ranking handles quality
