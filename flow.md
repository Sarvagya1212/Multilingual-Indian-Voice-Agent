# Project Decision Log — Multilingual Indian Voice Agent

## 1. Problem Definition

Build a real-time conversational AI voice agent for an Indian education counsellor that:
- Accepts streaming audio input
- Supports English, Hindi, Hinglish, and extensible to other Indian languages
- Detects language and code-switching
- Retrieves information from a knowledge base
- Calls tools when required
- Streams audio responses
- Handles interruptions naturally
- Records telemetry and evaluates interactions

## 2. Goals

- Real-time voice conversation (< 3s end-to-end latency target)
- Accurate STT for Indian English, Hindi, Hinglish
- Natural TTS with Indian accent
- Tool calling for course information
- RAG for dynamic knowledge retrieval
- Barge-in / interruption handling
- Full evaluation framework
- Dashboard for monitoring

## 3. Non-Goals

- Mobile app (web-based only)
- Real payment processing
- Student data storage beyond session
- Training custom foundation models
- Production deployment infrastructure

## 4. Initial Architecture

Frontend → Voice Gateway → Orchestrator → Response Pipeline → User

Voice Gateway components: VAD → STT
Orchestrator components: Language Detection → Intent Detection → Memory → RAG → Tool Calling → LLM
Response Pipeline components: Text normalization → TTS → Audio streaming

## 5. Technology Selection

[TO BE DECIDED IN PROMPTS 1.X]

## 6. STT Decision

[To be determined in Prompt 1.2]

## 7. TTS Decision

[To be determined in Prompt 1.3]

## 8. LLM Decision

[To be determined in Prompt 1.4]

## 9. Vector Database Decision

[To be determined in Prompt 4.1]

## 10. RAG Decision

[To be determined in Prompt 4.1]

## 11. Agent Architecture Decision

[To be determined in Prompt 3.1]

## 12. Tool Calling Decision

[To be determined in Prompt 3.1]

## 13. Memory Decision

[To be determined in Prompt 3.1]

## 14. Streaming Decision

[To be determined in Prompt 2.1]

## 15. VAD / Barge-In Decision

[To be determined in Prompt 2.1]

## 16. Dataset Decisions

[To be determined in Prompt 0.1]

## 17. Evaluation Decisions

[To be determined in Prompt 5.1]

## 18. Prompt Engineering Decisions

[To be determined in Prompt 1.4]

## 19. Model Training Decisions

[To be determined in Prompt 10.1]

## 20. Infrastructure Decisions

[To be determined in Prompt 0.1]

## 21. Security Decisions

[To be determined in Prompt 0.1]

## 22. Latency Optimization Decisions

[To be determined throughout]

## 23. Failed Experiments

[To be determined throughout]

## 24. Decision Reversals

[To be determined throughout]

## 25. Current Architecture

[Current implementation based on prompts completed]

## 26. Known Limitations

[To be determined throughout]

## 27. Future Experiments

[To be determined throughout]

Last updated: 2024-09-07