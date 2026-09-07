#!/usr/bin/env python3
"""Run Experiment 003: RAG Chunk Size.

Tests: 256 vs 512 vs 1024 char chunks — recall, relevance, keyword coverage.

Usage:
    python experiments/003_rag_chunk_size/run.py
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.chunker import TextChunker
from src.rag.embedder import LocalEmbedder
from src.rag.vector_store import VectorStore
from src.rag.retriever import RetrievedChunk


_TEST_QUERIES = [
    ("What is the JEE coaching fee?", ["jee_course"], ["150000", "installment", "scholarship"]),
    ("Do you offer NEET preparation?", ["neet_course"], ["NEET", "Biology", "Physics", "Chemistry"]),
    ("How do I take admission?", ["faq_admissions"], ["marksheets", "counseling"]),
    ("Is hostel facility available?", ["faq_admissions"], ["hostel", "boys", "girls", "mess"]),
    ("What scholarships are available?", ["jee_course", "faq_admissions"], ["scholarship", "50%", "meritorious"]),
    ("What is the CBSE board exam fee?", ["cbse_boards"], ["CBSE", "50000", "board"]),
    ("Who are the faculty?", ["jee_course", "neet_course"], ["faculty", "IIT", "experienced"]),
    ("What batches are available?", ["jee_course", "neet_course"], ["batch", "June", "September"]),
    ("Is there a free demo class?", ["faq_admissions"], ["demo", "free", "schedule"]),
    ("How can I contact you?", ["faq_admissions"], ["9876543210", "phone", "contact"]),
]


async def run_chunk_size_experiment(
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    """Run retrieval experiment for one chunk size."""
    from src.rag.knowledge_base import COURSE_DOCUMENTS

    embedder = LocalEmbedder(dimension=384)
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, min_chunk_size=50)
    store = VectorStore()

    # Index all docs
    all_chunks = []
    for doc in COURSE_DOCUMENTS:
        chunks = chunker.chunk_text(doc["content"], doc.get("metadata", {}), doc["source"])
        all_chunks.extend(chunks)

    if not all_chunks:
        return {"num_chunks": 0, "recall": 0.0, "avg_keyword_coverage": 0.0}

    texts = [c.text for c in all_chunks]
    vectors = await embedder.embed(texts)
    store.add(all_chunks, vectors)

    recalls = []
    keyword_coverages = []

    for query, expected_sources, expected_keywords in _TEST_QUERIES:
        qvec = await embedder.embed_query(query)
        results = store.search(qvec, top_k=10)
        retrieved_chunks = [
            RetrievedChunk(chunk=ch, score=sc, rank=i + 1)
            for i, (ch, sc) in enumerate(results)
        ]

        # Recall
        retrieved_sources = {rc.chunk.metadata.get("source", "") for rc in retrieved_chunks}
        matched = sum(1 for src in expected_sources if src in retrieved_sources)
        recall = matched / len(expected_sources) if expected_sources else 0.0
        recalls.append(recall)

        # Keyword coverage
        all_retrieved_text = " ".join(rc.chunk.text.lower() for rc in retrieved_chunks)
        kw_matched = sum(1 for kw in expected_keywords if kw.lower() in all_retrieved_text)
        kw_cov = kw_matched / len(expected_keywords) if expected_keywords else 0.0
        keyword_coverages.append(kw_cov)

    return {
        "num_chunks": len(all_chunks),
        "avg_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "avg_keyword_coverage": sum(keyword_coverages) / len(keyword_coverages) if keyword_coverages else 0.0,
        "per_query": list(zip([q for q, *_ in _TEST_QUERIES], recalls, keyword_coverages)),
    }


async def main() -> None:
    experiment_dir = Path(__file__).parent
    results_path = experiment_dir / "results.json"
    config = json.loads((experiment_dir / "config.json").read_text())

    print("=" * 60)
    print("  Experiment 003: RAG Chunk Size")
    print("=" * 60)

    chunk_sizes = config["chunk_sizes"]
    overlap = config["chunk_overlap"]
    all_results = {}

    for size in chunk_sizes:
        print(f"\n  Chunk size: {size} ...", end="", flush=True)
        result = await run_chunk_size_experiment(chunk_size=size, chunk_overlap=overlap)
        all_results[str(size)] = result
        print(
            f" {result['num_chunks']} chunks, "
            f"recall={result['avg_recall']:.2%}, "
            f"kw_cov={result['avg_keyword_coverage']:.2%}"
        )

    results = {
        "experiment_id": "003_rag_chunk_size",
        "config": config,
        "results": all_results,
    }
    results_path.write_text(json.dumps(results, indent=2))

    print(f"\n  Results saved to {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
