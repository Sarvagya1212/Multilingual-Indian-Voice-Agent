"""RAG evaluation metrics — retrieval recall, context relevance, groundedness."""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class RAGMetrics:
    """Metrics for a single RAG evaluation."""

    query: str
    retrieval_recall: float  # 0-1: fraction of relevant docs retrieved
    context_relevance: float  # 0-1: avg relevance score of retrieved chunks
    answer_groundedness: float  # 0-1: how much of the answer is in the context
    hallucination_score: float  # 0-1: fraction of unsupported claims
    num_retrieved: int
    num_relevant: int
    latency_ms: float
    has_answer: bool

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "retrieval_recall": self.retrieval_recall,
            "context_relevance": self.context_relevance,
            "answer_groundedness": self.answer_groundedness,
            "hallucination_score": self.hallucination_score,
            "num_retrieved": self.num_retrieved,
            "num_relevant": self.num_relevant,
            "latency_ms": self.latency_ms,
            "has_answer": self.has_answer,
        }


class RAGEvaluator:
    """Aggregates RAG evaluation results."""

    def __init__(self):
        self.results: list[RAGMetrics] = []

    def evaluate(
        self,
        query: str,
        retrieved_chunks: list,  # List of RetrievedChunk
        ground_truth_chunks: list[str] | None = None,
        generated_answer: str | None = None,
        relevance_scores: list[float] | None = None,
        latency_ms: float = 0.0,
    ) -> RAGMetrics:
        """Evaluate a single RAG query.

        Args:
            query: User query.
            retrieved_chunks: List of RetrievedChunk objects.
            ground_truth_chunks: Optional list of relevant source IDs.
            generated_answer: Optional LLM-generated answer for groundedness.
            relevance_scores: Optional pre-computed relevance scores.
            latency_ms: Total RAG latency in ms.

        Returns:
            RAGMetrics for this query.
        """
        # Retrieval recall: how many ground truth chunks were retrieved
        recall = 0.0
        num_relevant = 0
        if ground_truth_chunks:
            retrieved_sources = {rc.chunk.metadata.get("source", "") for rc in retrieved_chunks}
            matched = sum(1 for gt in ground_truth_chunks if gt in retrieved_sources)
            num_relevant = len(ground_truth_chunks)
            recall = matched / num_relevant if num_relevant > 0 else 0.0

        # Context relevance: average score
        if relevance_scores:
            context_relevance = sum(relevance_scores) / len(relevance_scores)
        elif retrieved_chunks:
            context_relevance = sum(rc.score for rc in retrieved_chunks) / len(retrieved_chunks)
        else:
            context_relevance = 0.0

        # Groundedness / hallucination
        groundedness = 0.0
        hallucination = 0.0
        has_answer = False
        if generated_answer is not None and retrieved_chunks:
            has_answer = len(generated_answer.strip()) > 0
            groundedness, hallucination = self._compute_groundedness(
                generated_answer, retrieved_chunks
            )
        elif generated_answer is not None:
            # No context retrieved — high hallucination risk
            has_answer = len(generated_answer.strip()) > 0
            groundedness = 0.0
            hallucination = 1.0

        metrics = RAGMetrics(
            query=query,
            retrieval_recall=recall,
            context_relevance=context_relevance,
            answer_groundedness=groundedness,
            hallucination_score=hallucination,
            num_retrieved=len(retrieved_chunks),
            num_relevant=num_relevant,
            latency_ms=latency_ms,
            has_answer=has_answer,
        )
        self.results.append(metrics)
        return metrics

    def _compute_groundedness(
        self,
        answer: str,
        retrieved_chunks: list,
    ) -> tuple[float, float]:
        """Estimate groundedness — fraction of answer tokens that appear in context."""
        if not answer.strip():
            return 0.0, 1.0

        context_text = " ".join(rc.chunk.text for rc in retrieved_chunks).lower()
        context_words = set(re.findall(r"\w+", context_text))

        answer_words = re.findall(r"\w+", answer.lower())
        if not answer_words:
            return 0.0, 1.0

        # Only count content words (length > 2)
        content_words = [w for w in answer_words if len(w) > 2]
        if not content_words:
            return 1.0, 0.0

        supported = sum(1 for w in content_words if w in context_words)
        groundedness = supported / len(content_words)
        hallucination = 1.0 - groundedness
        return groundedness, hallucination

    def get_summary(self) -> dict:
        """Aggregate RAG metrics."""
        if not self.results:
            return {}
        n = len(self.results)
        return {
            "total_queries": n,
            "avg_retrieval_recall": sum(r.retrieval_recall for r in self.results) / n,
            "avg_context_relevance": sum(r.context_relevance for r in self.results) / n,
            "avg_answer_groundedness": sum(r.answer_groundedness for r in self.results) / n,
            "avg_hallucination": sum(r.hallucination_score for r in self.results) / n,
            "avg_num_retrieved": sum(r.num_retrieved for r in self.results) / n,
            "avg_latency_ms": sum(r.latency_ms for r in self.results) / n,
            "has_answer_rate": sum(1 for r in self.results if r.has_answer) / n * 100,
        }

    def clear(self) -> None:
        self.results = []
