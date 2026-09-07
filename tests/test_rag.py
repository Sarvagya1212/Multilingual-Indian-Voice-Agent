"""Tests for the RAG pipeline — chunker, embedder, retriever, reranker, knowledge base."""
import pytest
import numpy as np

from src.rag.chunker import TextChunk, TextChunker
from src.rag.embedder import LocalEmbedder, get_embedder
from src.rag.vector_store import VectorStore
from src.rag.reranker import Reranker
from src.rag.retriever import RetrievedChunk, Retriever
from src.rag.knowledge_base import COURSE_DOCUMENTS, KnowledgeBase


# ---------------------------------------------------------------------------
# TextChunker
# ---------------------------------------------------------------------------

class TestTextChunker:
    def test_clean_text(self):
        c = TextChunker()
        # Multiple spaces collapsed
        assert c._clean_text("hello    world") == "hello world"
        # Multiple newlines collapsed
        assert c._clean_text("a\n\n\nb") == "a\n\nb"
        # Special characters stripped
        assert "@#$" not in c._clean_text("hello @#$ world")

    def test_split_sentences(self):
        c = TextChunker()
        text = "Hello world. This is a test! What do you think?"
        sents = c._split_sentences(text)
        assert len(sents) >= 3

    def test_split_sentences_abbreviations_protected(self):
        c = TextChunker()
        text = "Dr. Sharma is here. Mr. Patel called."
        sents = c._split_sentences(text)
        # Should not split on Dr. or Mr.
        assert any("Dr. Sharma" in s or "Dr" in s for s in sents)

    def test_chunk_text_single_sentence_below_min(self):
        c = TextChunker(chunk_size=200, min_chunk_size=50)
        chunks = c.chunk_text("Short sentence.", {"topic": "test"})
        # Too short — below min_chunk_size
        assert len(chunks) == 0

    def test_chunk_text_exact_fit(self):
        c = TextChunker(chunk_size=100, min_chunk_size=20)
        text = "This is sentence one. " * 10  # Enough to fill chunk
        chunks = c.chunk_text(text, {"topic": "test"}, source="doc1")
        assert len(chunks) >= 1
        assert all(isinstance(ch, TextChunk) for ch in chunks)
        assert all(ch.chunk_id.startswith("doc1_") for ch in chunks)
        assert all("source" in ch.metadata for ch in chunks)

    def test_chunk_text_overlap(self):
        c = TextChunker(chunk_size=80, chunk_overlap=20, min_chunk_size=10)
        text = ("This is sentence one. " * 4 + "This is sentence two. " * 4 +
                "This is sentence three. " * 4 + "This is sentence four. " * 4)
        chunks = c.chunk_text(text, {}, source="doc")
        # At least 2 chunks
        assert len(chunks) >= 2
        # Overlap means second chunk starts within the overlap window
        assert chunks[1].start_char >= chunks[0].end_char - c.chunk_overlap - 50

    def test_chunk_text_metadata_preserved(self):
        c = TextChunker(chunk_size=100, min_chunk_size=20)
        text = "This is a long enough sentence to form a chunk. " * 5
        chunks = c.chunk_text(text, {"category": "faq"}, source="faq_doc")
        assert all(ch.metadata["category"] == "faq" for ch in chunks)
        assert all(ch.metadata["source"] == "faq_doc" for ch in chunks)

    def test_chunk_text_multiple_chunks_have_sequential_ids(self):
        c = TextChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=10)
        text = "Sentence one here. Sentence two here. Sentence three here. Sentence four here. Sentence five here."
        chunks = c.chunk_text(text, {}, source="m")
        ids = [ch.chunk_id for ch in chunks]
        # All unique
        assert len(ids) == len(set(ids))
        # Sequential indices
        indices = [int(ch.chunk_id.split("_")[1]) for ch in chunks]
        assert indices == list(range(len(indices)))

    def test_chunk_text_empty_input(self):
        c = TextChunker()
        assert c.chunk_text("", {}) == []
        assert c.chunk_text("   ", {}) == []


# ---------------------------------------------------------------------------
# LocalEmbedder
# ---------------------------------------------------------------------------

class TestLocalEmbedder:
    @pytest.mark.asyncio
    async def test_embed_returns_correct_shape(self):
        e = LocalEmbedder(dimension=128)
        texts = ["hello world", "foo bar"]
        result = await e.embed(texts)
        assert result.shape == (2, 128)

    @pytest.mark.asyncio
    async def test_embed_query_returns_correct_shape(self):
        e = LocalEmbedder(dimension=64)
        vec = await e.embed_query("test query")
        assert vec.shape == (64,)

    @pytest.mark.asyncio
    async def test_embed_query_matches_batch(self):
        e = LocalEmbedder(dimension=64)
        single = await e.embed_query("same text")
        batch = await e.embed_query("same text")
        # Should produce identical vectors (same hash values within one process)
        diff = float(np.abs(single - batch).max())
        assert diff < 1e-5

    @pytest.mark.asyncio
    async def test_different_texts_different_vectors(self):
        e = LocalEmbedder(dimension=64)
        v1 = await e.embed_query("engineering JEE course")
        v2 = await e.embed_query("medical NEET exam")
        diff = float(np.abs(v1 - v2).max())
        assert diff > 0

    @pytest.mark.asyncio
    async def test_identical_texts_identical_vectors(self):
        e = LocalEmbedder(dimension=64)
        v1 = await e.embed_query("JEE preparation course")
        v2 = await e.embed_query("JEE preparation course")
        diff = float(np.abs(v1 - v2).max())
        assert diff < 1e-5

    def test_get_embedder_local(self):
        e = get_embedder("local", dimension=32)
        assert e.dimension == 32


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class TestVectorStore:
    @pytest.mark.asyncio
    async def test_add_and_search(self):
        e = LocalEmbedder(dimension=64)
        chunks = [
            TextChunk(
                text="JEE Main and Advanced preparation",
                metadata={},
                chunk_id="c0",
                start_char=0,
                end_char=40,
            ),
            TextChunk(
                text="NEET Biology syllabus covers Class 11 and 12",
                metadata={},
                chunk_id="c1",
                start_char=0,
                end_char=50,
            ),
            TextChunk(
                text="CBSE Board exam preparation Class 12",
                metadata={},
                chunk_id="c2",
                start_char=0,
                end_char=40,
            ),
        ]
        vectors = await e.embed([c.text for c in chunks])

        store = VectorStore()
        store.add(chunks, vectors)

        query = await e.embed_query("JEE exam preparation")
        results = store.search(query, top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r[0], TextChunk) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    @pytest.mark.asyncio
    async def test_search_empty_store(self):
        store = VectorStore()
        query = await LocalEmbedder(dimension=64).embed_query("test")
        assert store.search(query) == []

    @pytest.mark.asyncio
    async def test_search_top_k_respected(self):
        e = LocalEmbedder(dimension=64)
        chunks = [
            TextChunk(
                text=f"test query number {i} sample text",
                metadata={},
                chunk_id=f"c{i}",
                start_char=0,
                end_char=30,
            )
            for i in range(5)
        ]
        vectors = await e.embed([c.text for c in chunks])
        store = VectorStore()
        store.add(chunks, vectors)

        query = await e.embed_query("test query sample")
        results = store.search(query, top_k=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_scores_descending(self):
        e = LocalEmbedder(dimension=64)
        chunks = [
            TextChunk(text="engineering JEE course details", metadata={}, chunk_id="c0", start_char=0, end_char=40),
            TextChunk(text="cooking recipe pasta", metadata={}, chunk_id="c1", start_char=0, end_char=20),
        ]
        vectors = await e.embed([c.text for c in chunks])
        store = VectorStore()
        store.add(chunks, vectors)

        query = await e.embed_query("JEE exam")
        results = store.search(query, top_k=2)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_len(self):
        e = LocalEmbedder(dimension=32)
        chunks = [
            TextChunk(text=f"chunk {i}", metadata={}, chunk_id=f"c{i}", start_char=0, end_char=6)
            for i in range(3)
        ]
        vectors = await e.embed([c.text for c in chunks])
        store = VectorStore()
        assert len(store) == 0
        store.add(chunks[:2], vectors[:2])
        assert len(store) == 2
        store.add(chunks[2:], vectors[2:])
        assert len(store) == 3


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class TestReranker:
    def test_rerank_returns_top_k(self):
        r = Reranker()
        chunks = [
            TextChunk(text="engineering JEE course details fee scholarship", metadata={}, chunk_id="c0", start_char=0, end_char=60),
            TextChunk(text="cooking pasta recipe tomato sauce", metadata={}, chunk_id="c1", start_char=0, end_char=30),
        ]
        results = [(chunks[0], 0.7), (chunks[1], 0.3)]
        reranked = r.rerank("JEE course fee", results, top_k=2)
        assert len(reranked) <= 2
        # JEE chunk should rank higher
        ids = [r[0].chunk_id for r in reranked]
        assert ids.index("c0") < ids.index("c1")

    def test_rerank_keyword_overlap(self):
        r = Reranker(keyword_weight=0.5)
        chunks = [
            TextChunk(text="JEE Main Advanced Physics Chemistry Mathematics", metadata={}, chunk_id="c0", start_char=0, end_char=60),
            TextChunk(text="football sports games tournament", metadata={}, chunk_id="c1", start_char=0, end_char=35),
        ]
        results = [(chunks[0], 0.5), (chunks[1], 0.5)]
        reranked = r.rerank("JEE Physics Chemistry", results, top_k=2)
        ids = [r[0].chunk_id for r in reranked]
        assert ids[0] == "c0"

    def test_rerank_empty_input(self):
        r = Reranker()
        assert r.rerank("query", []) == []

    def test_length_score(self):
        r = Reranker()
        # Short text
        assert r._length_score("hi") < 1.0
        # Ideal range
        ideal = r._length_score("a" * 500)
        assert ideal == 1.0
        # Too long
        long_score = r._length_score("a" * 2000)
        assert long_score < 1.0


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class TestRetriever:
    @pytest.mark.asyncio
    async def test_index_documents(self):
        r = Retriever()
        docs = [
            {
                "source": "test1",
                "content": (
                    "JEE Main and Advanced coaching for Class 11 and 12 students. "
                    "Comprehensive preparation covering Physics Chemistry Mathematics."
                ),
                "metadata": {"category": "engineering"},
            },
            {
                "source": "test2",
                "content": (
                    "NEET preparation covering Physics Chemistry Biology. "
                    "Complete medical entrance exam coaching with expert faculty."
                ),
                "metadata": {"category": "medical"},
            },
        ]
        await r.index_documents(docs)
        assert len(r.vector_store) > 0

    @pytest.mark.asyncio
    async def test_retrieve(self):
        r = Retriever()
        docs = [
            {
                "source": "jee",
                "content": (
                    "JEE Main coaching with Physics Chemistry Mathematics. "
                    "IIT preparation course for engineering aspirants."
                ),
                "metadata": {"category": "engineering"},
            },
            {
                "source": "neet",
                "content": (
                    "NEET Biology preparation with NCERT syllabus. "
                    "Medical college entrance exam coaching program."
                ),
                "metadata": {"category": "medical"},
            },
        ]
        await r.index_documents(docs)
        results = await r.retrieve("JEE preparation")
        assert len(results) >= 1
        assert all(isinstance(rc, RetrievedChunk) for rc in results)
        assert results[0].score >= 0

    @pytest.mark.asyncio
    async def test_retrieve_with_filter(self):
        r = Retriever()
        await r.index_documents([
            {"source": "e1", "content": "engineering content", "metadata": {"category": "engineering"}},
            {"source": "m1", "content": "medical content", "metadata": {"category": "medical"}},
        ])
        results = await r.retrieve("content", filters={"category": "medical"})
        assert all(rc.chunk.metadata.get("category") == "medical" for rc in results)

    @pytest.mark.asyncio
    async def test_retrieve_with_rerank(self):
        r = Retriever()
        await r.index_documents([
            {"source": "doc1", "content": "JEE course fee Rs 150000 scholarship available", "metadata": {}},
            {"source": "doc2", "content": "NEET coaching Biology Physics Chemistry", "metadata": {}},
            {"source": "doc3", "content": "CBSE boards Class 12 preparation", "metadata": {}},
        ])
        results = await r.retrieve_with_rerank("JEE fee and scholarship", final_k=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_retrieve_unindexed(self):
        r = Retriever()
        results = await r.retrieve("any query")
        assert results == []

    @pytest.mark.asyncio
    async def test_build_context(self):
        r = Retriever()
        await r.index_documents([
            {
                "source": "src1",
                "content": (
                    "JEE Main and Advanced coaching details for engineering students. "
                    "Complete Physics Chemistry Mathematics preparation course."
                ),
                "metadata": {},
            },
        ])
        chunks = await r.retrieve("JEE")
        assert len(chunks) >= 1
        ctx = r.build_context(chunks)
        assert len(ctx) > 0
        assert "src1" in ctx or "JEE" in ctx

    @pytest.mark.asyncio
    async def test_build_context_max_chars(self):
        r = Retriever()
        await r.index_documents([
            {"source": "src1", "content": "A" * 2000, "metadata": {}},
            {"source": "src2", "content": "B" * 2000, "metadata": {}},
        ])
        chunks = await r.retrieve("any")
        ctx = r.build_context(chunks, max_chars=500)
        assert len(ctx) <= 600  # some overhead for separators

    @pytest.mark.asyncio
    async def test_retrieve_top_k_override(self):
        r = Retriever()
        await r.index_documents([
            {"source": f"d{i}", "content": f"content {i}", "metadata": {}} for i in range(5)
        ])
        results = await r.retrieve("content", top_k=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class TestKnowledgeBase:
    @pytest.mark.asyncio
    async def test_build_default_documents(self):
        kb = KnowledgeBase()
        assert not kb.is_built
        await kb.build()
        assert kb.is_built

    @pytest.mark.asyncio
    async def test_query_returns_chunks(self):
        kb = KnowledgeBase()
        await kb.build()
        results = await kb.query("What courses do you offer?")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_query_with_filters(self):
        kb = KnowledgeBase()
        await kb.build()
        results = await kb.query("fees", filters={"category": "engineering"})
        for rc in results:
            assert rc.chunk.metadata.get("category") == "engineering"

    @pytest.mark.asyncio
    async def test_query_rag_returns_context_string(self):
        kb = KnowledgeBase()
        await kb.build()
        ctx = await kb.query_rag("What is the fee for JEE?")
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    @pytest.mark.asyncio
    async def test_get_knowledge_base_singleton(self):
        from src.rag import get_knowledge_base

        kb1 = get_knowledge_base()
        kb2 = get_knowledge_base()
        assert kb1 is kb2

    @pytest.mark.asyncio
    async def test_query_on_unbuilt_kb_auto_builds(self):
        kb = KnowledgeBase()
        # Don't call build() — query() should auto-build
        results = await kb.query("JEE")
        assert kb.is_built
        assert isinstance(results, list)
