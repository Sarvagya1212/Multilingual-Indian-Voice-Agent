# Research Documentation

This directory contains summaries of research papers, technical blog
posts, and reference implementations that influenced our engineering
decisions for the Multilingual Indian Voice Agent.

The goal is **traceable design choices**: every meaningful architectural
or component decision in `flow.md` should be backed by either a real
measurement (from `experiments/`) or a reference in this folder.

## Format

Each entry uses a consistent template so they can be parsed and indexed:

1. **Paper/Source** — citation with link
2. **Problem** — what the paper solved
3. **Key Approach** — the technique in 2-3 sentences
4. **Relevant Ideas for Our Project** — bullets of what we could borrow
5. **What We Implemented** — checked off
6. **What We Did NOT Implement** — with reason
7. **Results** — quantified impact if we measured it
8. **Inspiration Statement** — one-line summary

## Papers Reviewed

| # | Topic | Influence on Project |
|---|-------|---------------------|
| 001 | [Speech Recognition for Indian Languages](./001_speech_recognition_indian_languages.md) | Whisper language hints; Hinglish preprocessing |
| 002 | [Streaming STT](./002_streaming_stt.md) | Partial-result streaming; VAD-first design |
| 003 | [Voice Agent Architecture](./003_voice_agent_architecture.md) | State machine over LangChain; explicit tool registry |
| 004 | [RAG for Domain QA](./004_rag_for_domain_qa.md) | Re-ranking over cosine-only retrieval; chunk size sensitivity |
| 005 | [Code-Switching in Multilingual Models](./005_code_switching.md) | Language detection heuristics; prompt language locks |

## Research Process

For each paper:

1. Read the abstract + method (skip the maths; we're engineers, not researchers)
2. Ask: does this solve a problem we have?
3. If yes, write 2-3 bullet ideas worth borrowing
4. Note what we actually shipped and what we deferred
5. Cite in the relevant section of `flow.md`

## Key Takeaways

1. **Language-specific optimisation improves accuracy.** Whisper's
   `language` parameter cuts Hinglish WER by 25-30% on the IndicWhisper
   evaluation set; we ship the equivalent with a `language` hint
   passed to `STTProvider.transcribe(audio, language=...)`.
2. **Cascaded pipeline (VAD → STT → LLM) is the production pattern.**
   End-to-end speech-LLMs are impressive in demos but still lose on
   latency and control; the cascaded approach lets us swap each stage
   independently.
3. **Explicit state machines beat implicit ones.** Tool-calling agents
   without a state graph hallucinate transitions; the LLM orchestrator
   inside our state machine (Prompt 3.1) is bounded by valid edges.
4. **Re-ranking rescues weak embeddings.** When the embedder is
   cheap (LocalEmbedder) or off-domain, a re-ranker with keyword
   overlap can still surface the right chunk.
5. **Code-switching needs a multi-signal detector.** Script ratio alone
   fails on transliterated Hinglish; combining script + character n-gram
   + LLM fallback is the pragmatic baseline.

## Inspiration vs Citation

These documents are **inspirations**, not citations. The papers are
referenced because they shaped our thinking; we did not reproduce their
experiments or results. Where we cite a number (e.g. "Hinglish WER
0.32 → 0.24"), the source is the paper's own evaluation — we have not
yet measured it on our pipeline.
