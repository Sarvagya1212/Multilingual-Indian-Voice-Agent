"""Demo 5: RAG (Retrieval-Augmented Generation).

The agent retrieves relevant chunks from the knowledge base, then
generates a response grounded in the retrieved context.

This demo uses the real `src/rag/knowledge_base.py` (no API keys needed
because the LocalEmbedder is hash-based and the re-ranker is lexical).
"""
from __future__ import annotations

from demos._common import header, info, ok, run, step, warn


async def main() -> None:
    header("Demo 5: Retrieval-Augmented Generation")
    info("Query the knowledge base, retrieve top-k chunks, ground the answer.")
    print()

    # Try the real RAG pipeline; fall back gracefully if not available
    try:
        from src.rag.knowledge_base import KnowledgeBase, COURSE_DOCUMENTS
        from src.rag.config import RAGConfig
    except Exception as e:
        warn(f"RAG modules not importable: {e}")
        warn("Run `pip install -r requirements.txt` from the project root.")
        return

    step(1, 4, "Build knowledge base")
    kb = KnowledgeBase()
    await kb.build()
    n_chunks = len(kb.retriever.vector_store.chunks)
    info(f"Knowledge base built: {n_chunks} chunks across {len(COURSE_DOCUMENTS)} sources:")
    for doc in COURSE_DOCUMENTS:
        info(f"  - {doc.get('source', '?')}: {len(doc.get('text', ''))} chars")
    print()

    step(2, 4, "Query knowledge base")
    config = RAGConfig()
    info(f"Chunk size: {config.chunk_size} | Overlap: {config.chunk_overlap}")
    print()

    step(3, 4, "Retrieve for queries")
    queries = [
        "What is the fee for JEE?",
        "Am I eligible for NEET?",
        "Where is the hostel facility?",
    ]
    for q in queries:
        context_str = await kb.query_rag(q)
        preview = context_str[:120] + ("..." if len(context_str) > 120 else "")
        print(f"  Q: {q}")
        print(f"  Context: {preview}")
        print()

    step(4, 4, "Build context string")
    context = await kb.query_rag("What is the fee for JEE?")
    print(f"  Context (first 200 chars):")
    print(f"  {context[:200]!r}")
    print()

    ok("RAG demo completed successfully.")


if __name__ == "__main__":
    run(main)
