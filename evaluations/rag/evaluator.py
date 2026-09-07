"""RAG evaluator — runs retrieval against query/ground-truth pairs."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from src.logger import setup_logger
from evaluations.rag.metrics import RAGEvaluator

logger = setup_logger(__name__)


@dataclass
class RAGTestCase:
    """A single test case for RAG evaluation."""

    query: str
    ground_truth_sources: list[str]  # Expected source IDs
    expected_keywords: list[str] | None = None  # Keywords expected in retrieved chunks
    language: str = "en"
    filters: dict[str, Any] | None = None


class RAGEvaluatorRunner:
    """Runs RAG evaluation against test queries with known answers."""

    def __init__(
        self,
        retriever: Any,  # Retriever instance
        evaluator: RAGEvaluator | None = None,
    ):
        self.retriever = retriever
        self.evaluator = evaluator or RAGEvaluator()

    async def evaluate_case(
        self,
        case: RAGTestCase,
        generated_answer: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single RAG query."""
        start = time.time()
        try:
            chunks = await self.retriever.retrieve(
                case.query,
                filters=case.filters,
            )
            latency_ms = (time.time() - start) * 1000
        except Exception as exc:
            logger.error(f"[RAG Eval] Retrieval failed: {exc}")
            return {
                "case": case,
                "success": False,
                "error": str(exc),
            }

        metrics = self.evaluator.evaluate(
            query=case.query,
            retrieved_chunks=chunks,
            ground_truth_chunks=case.ground_truth_sources,
            generated_answer=generated_answer,
            latency_ms=latency_ms,
        )

        # Keyword coverage: fraction of expected keywords in retrieved chunks
        keyword_coverage = 0.0
        if case.expected_keywords and chunks:
            all_text = " ".join(c.chunk.text.lower() for c in chunks)
            matched = sum(1 for kw in case.expected_keywords if kw.lower() in all_text)
            keyword_coverage = matched / len(case.expected_keywords)

        return {
            "case": case,
            "success": True,
            "chunks": chunks,
            "keyword_coverage": keyword_coverage,
            "metrics": metrics,
        }

    async def run(
        self,
        cases: list[RAGTestCase],
        generate_answers: bool = False,
        llm_provider: Any = None,
    ) -> dict[str, Any]:
        """Run evaluation against all cases.

        Args:
            cases: List of RAG test cases.
            generate_answers: If True, generate answers via LLM for groundedness.
            llm_provider: Required if generate_answers=True.
        """
        results = []
        for case in cases:
            answer = None
            if generate_answers and llm_provider is not None:
                # Build a simple prompt
                chunks = await self.retriever.retrieve(case.query, filters=case.filters)
                context = self.retriever.build_context(chunks)
                prompt = f"Context:\n{context}\n\nQuestion: {case.query}"
                try:
                    from src.llm import Message, MessageRole
                    response = await llm_provider.chat([
                        Message(role=MessageRole.USER, content=prompt)
                    ])
                    answer = response.content
                except Exception as exc:
                    logger.warning(f"[RAG Eval] Answer generation failed: {exc}")

            result = await self.evaluate_case(case, generated_answer=answer)
            results.append(result)

        return {
            "summary": self.evaluator.get_summary(),
            "cases": results,
        }
