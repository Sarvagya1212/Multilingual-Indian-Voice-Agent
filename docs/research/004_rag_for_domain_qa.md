# Research: RAG for Domain QA

## Paper/Source

**Retrieval-Augmented Generation for Domain-Specific Question Answering**

- Type: Survey + industry pattern (Lewis et al. 2020 RAG paper +
  Anthropic / OpenAI / Cohere RAG blogs)
- URL: https://arxiv.org/abs/2005.11401
- Related: Anthropic Contextual Retrieval (2024)

## Problem

A domain-specific voice agent (e.g. education counsellor) must answer
questions from a private knowledge base (courses, fees, eligibility) with
high factual accuracy. Pure LLM responses hallucinate; pure retrieval
without generation lacks natural language. The right approach is RAG:
retrieve relevant context, then generate a response conditioned on that
context.

## Key Approach

- **Chunk documents into retrieval-friendly units.** A common pattern is
  256-1024 character chunks with 10-20% overlap. Chunk size is a
  trade-off: small = precise but loses context; large = preserves
  context but lowers precision.
- **Embed chunks with a semantic encoder.** OpenAI `text-embedding-3-small`
  is the de-facto baseline; alternatives include Cohere `embed-english-v3.0`
  and BGE.
- **Retrieve top-k by cosine similarity.** k=5-10 is typical. Some systems
  add a cross-encoder re-ranker on top to refine the order.
- **Generate with retrieved context as a system message.** The LLM is
  instructed to ground its response in the provided context and to
  say "I don't know" if the context doesn't contain the answer.

## Relevant Ideas for Our Project

1. **Re-ranking is a force multiplier.** Cosine similarity from a generic
   embedder often misses the most relevant chunk; a re-ranker (lexical
   overlap + position) can rescue recall without retraining the embedder.
2. **Chunk size is a hyperparameter, not a constant.** Our experiment
   (`experiments/003_rag_chunk_size/`) found 512 chars is the sweet spot
   for our 5-document knowledge base.
3. **Faithfulness scoring.** A ground-truth retrieval recall metric + a
   word-overlap groundedness score are cheap proxies for hallucination;
   a stronger approach uses an LLM-as-judge.
4. **Citation in responses.** The LLM should reference which document the
   answer came from; reduces hallucination and makes answers auditable.

## What We Implemented

- [x] **Sentence-aware chunker** (`src/rag/chunker.py`): respects sentence
  boundaries, abbreviation detection, configurable size and overlap.
- [x] **Pluggable embedder** (`src/rag/embedder.py`): `LocalEmbedder`
  (hash-based, for tests) + `OpenAIEmbedder` (real). Factory function
  selects based on `OPENAI_API_KEY`.
- [x] **In-memory vector store with cosine similarity**
  (`src/rag/vector_store.py`): normalised vectors, `>= 0` filter.
- [x] **Lexical re-ranker** (`src/rag/reranker.py`): 60% cosine +
  30% keyword overlap + 10% length; weights tuned empirically.
- [x] **Retriever with metadata filtering** (`src/rag/retriever.py`):
  filter by source type, recency, or course ID.
- [x] **Knowledge base with 5 source documents**
  (`src/rag/knowledge_base.py`): JEE, NEET, CBSE, FAQ admissions, FAQ
  general — covers the demo's domain completely.
- [x] **RAG evaluation harness** (`evaluations/rag/`): recall,
  context relevance, groundedness.
- [x] **Chunk-size experiment** (`experiments/003_rag_chunk_size/`):
  empirically determined 512 chars is optimal.

## What We Did NOT Implement

- **Cross-encoder re-ranking (BGE-reranker)** — adds 200-500ms latency
  and requires GPU for the model. Our lexical re-ranker is a good
  approximation for our scale.
- **Hybrid BM25 + vector** — pure vector + re-ranker achieves
  100% recall on our small KB; hybrid would help at scale.
- **GraphRAG** — Microsoft Research's GraphRAG (knowledge graph
  construction from documents) is overkill for our 5-doc KB.
- **LLM-as-judge faithfulness** — a strong approach but slow and
  expensive. We use word-overlap groundedness as a cheap proxy.
- **Citation in responses** — the LLM is instructed to ground in
  context but doesn't currently return the source document ID in
  its response.

## Results

- Chunk-size experiment: 512 chars wins on both recall and keyword
  coverage (90% vs 86.67% for 256-char chunks). Decision: ship 512
  as default in `RAGConfig`.
- RAG evaluation: 100% retrieval recall on 10 ground-truth queries
  with 512-char chunks.
- Re-ranker improves precision by ~10-15% vs raw cosine on the
  same query set (measured in `tests/test_rag.py`).

## Inspiration Statement

The chunk-size + re-ranker + groundedness-evaluation pattern is the
production baseline; we ship a slightly simpler version (lexical
re-ranker instead of cross-encoder) that meets our recall and latency
targets on a 5-doc knowledge base.
