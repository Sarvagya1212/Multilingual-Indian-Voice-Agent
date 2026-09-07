# Experiment 003: RAG Chunk Size

## Hypothesis

Chunk size affects retrieval quality vs answer completeness. Larger chunks
(1024) provide more context per retrieval but reduce precision. Smaller
chunks (256) are more precise but may lose information.

## Setup

- **Knowledge base**: `src/rag/knowledge_base.py` default documents
- **Chunk sizes tested**: 256, 512, 1024 characters
- **Chunk overlap**: fixed at 50 characters
- **Embedding model**: `LocalEmbedder` (hashed word vectors) for reproducibility
- **Test queries**: 10 ground-truth-annotated queries (see run.py)
- **Metrics**: retrieval recall, keyword coverage, context relevance

## Results

| Chunk Size | Num Chunks | Recall | Keyword Coverage |
|------------|-----------|--------|------------------|
| 256        | 20        | 100%   | 86.67%           |
| 512        | 11        | 100%   | 90.00%           |
| 1024       | 6         | 100%   | 90.00%           |

## Analysis

- **Recall is identical (100%)** for all chunk sizes on this small test set — every relevant source is in the top-10 regardless of how it's chunked.
- **Keyword coverage** peaks at 512 (90%) and stays flat at 1024 (90%) — diminishing returns.
- **256-char chunks produce more chunks** (20 vs 11 vs 6), increasing embedding cost and noise in retrieval.
- **1024-char chunks preserve more context per retrieval** but reduce granularity — if a query needs just one fact from a 1024-char chunk, you retrieve 3 other facts too.
- **Sweet spot: 512 chars** — best keyword coverage with reasonable granularity.

## Decision

**Use 512-char chunks with 50-char overlap** as the default for `RAGConfig`.

- 100% recall on all 10 test queries
- 90% keyword coverage (highest tied with 1024)
- 11 chunks is a good middle ground between precision and context
- Embedding cost is 45% lower than 256-char (11 vs 20 chunks)

## Follow-up

- [ ] Test with OpenAI embeddings to see if trends hold
- [ ] Vary chunk overlap (10%, 20%, 30% of chunk size)
- [ ] Try semantic chunking (split at embedding-distance boundaries)

## Follow-up

- Test with OpenAI embeddings to see if trends hold
- Vary chunk overlap (10%, 20%, 30% of chunk size)
- Try semantic chunking (split at embedding-distance boundaries)
