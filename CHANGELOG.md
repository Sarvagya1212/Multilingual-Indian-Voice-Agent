# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.3.0] - 2026-09-08

### Added

- **Live launch script** (`run_live.py`): single entry point that wires
  `ConversationOrchestrator` to `VoiceGateway` WebSocket server and serves the
  dashboard frontend over HTTP. Includes prerequisite checks (Ollama, internet,
  Whisper, API keys), auto-enables telemetry, and opens the browser.
  - HTTP server on `http://localhost:8080` — serves `dashboard/index.html`
  - WebSocket server on `ws://localhost:8765` — real-time audio streaming
  - Ports configurable via `HTTP_PORT` and `WS_PORT` environment variables

### Fixed

- **Frontend audio never reached the gateway** (`dashboard/app.js`):
  - `ScriptProcessor` was not connected to the `MediaStreamSource` — no audio flowed
  - No `{"type": "start"}` / `{"type": "stop"}` control messages were sent, so
    `_process_turn()` on the gateway was never triggered
  - Audio was buffered and sent as a single blob on mouseup instead of streamed
    in real-time as PCM binary frames
  - WebSocket URL used `window.location.port` (8080 when served by HTTP server)
    instead of the gateway port (8765)
- Added `AudioContext.resume()` call to handle browser autoplay policy

### Changed

- `README.md`: "Talk to the Agent (Live)" section added to Quick Start
- `ARCHITECTURE.md`: added "Live Usage" section explaining how `run_live.py` wires components
- `flow.md`: added section "29. Live Voice Agent Entry Point"
- `demos/README.md`: added note pointing to `run_live.py` for live experience

---

## [0.2.0] - 2026-09-07

### Added

- **RAG pipeline** (`src/rag/`): sentence-aware chunker (512-char, 50-char overlap),
  pluggable embedder (LocalEmbedder hash-based + OpenAIEmbedder), in-memory vector store,
  lexical re-ranker (60/30/10 cosine/keyword/length), retriever with metadata filtering,
  knowledge base with 5 education domain documents (JEE, NEET, CBSE, FAQ-admissions, FAQ-general).
- **Evaluation framework** (`evaluations/`): STT WER/CER with self-contained Levenshtein,
  TTS acronym counting and RTF, RAG recall/context-relevance/groundedness,
  agent task-completion metrics; master CLI `python -m evaluations.run [--component stt|tts|rag|agent]`.
- **Experiment tracking** (`experiments/`): structured experiment directory convention,
  `experiments/run_all.py` orchestrator, `experiments_index.md` status table.
  - Experiment 001 (Whisper model comparison): complete — `base` model selected
  - Experiment 002 (TTS provider comparison): skeleton
  - Experiment 003 (RAG chunk size): complete — **512-char chunks selected**
- **Metrics dashboard** (`dashboard/`): Streamlit app with KPI strip, daily-conversation
  bar chart, latency-by-component chart, language pie, recent-conversations viewer.
  Auto-seeds 45 synthetic events in `logs/demo_events.jsonl` when no real log exists.
- **Failure analysis** (`failures/`): structured failure log with 13 real failures
  (F001–F013), parser-validated summary table, `get_stats()` for live counts.
  Covers: STT, TTS, language detection, RAG, test infrastructure, build, prompts.
- **Research documentation** (`docs/research/`): 5 papers documented with inspiration
  vs citation distinction: IndicWhisper, streaming STT, voice agent architecture,
  RAG for domain QA, code-switching.
- **Demo scripts** (`demos/`): 8 standalone demos (English, Hindi, Hinglish,
  tool calling, RAG, interruption, failure recovery, dashboard). All work without
  API keys; real RAG pipeline in demo 5; cooperative cancellation in demo 6.
- **Tests** (`tests/`): 465 tests total across 15 test files.
  - `test_rag.py`: 38 tests
  - `test_evaluations.py`: 56 tests
  - `test_experiments.py`: 20 tests
  - `test_dashboard_app.py`: 23 tests
  - `test_failure_analysis.py`: 28 tests
  - `test_research_docs.py`: 67 tests
  - `test_demos.py`: 32 tests
  - Plus: stt (16), tts (28), llm (30), orchestrator (16), agent (49), gateway (33), websocket_server (10), dashboard (19)

### Fixed

- RAG `min_similarity_score=0.3` too strict for LocalEmbedder → lowered to 0.0
- RAG `min_chunk_size=100` dropped short FAQ docs → lowered to 50
- RAG vector store `> 0` filter dropped zero-similarity candidates → changed to `>= 0`
- RAG retriever `async def` called without `await` in test → added `@pytest.mark.asyncio`
- Evaluator percentile math wrong → corrected to linear interpolation (p50=450, p90=810)
- TTS acronym regex too strict → simplified to `\b[A-Z]\b` for letter counting
- Windows cp1252 encoding crash on emoji in `print()` → replaced with ASCII labels
- PowerShell 5.1 `&&` chain operator not supported → use `;`
- Experiment test asserted `status="skeleton"` after running → updated to `"complete"`
- Stdout cp1252 crash on Hindi Devanagari in demos → `sys.stdout.reconfigure(encoding="utf-8")`

### Changed

- `src/rag/config.py`: `chunk_size` default 512, `chunk_overlap` default 50
- `src/rag/vector_store.py`: similarity filter changed to `>= 0`
- `src/stt/`: VAD module moved from `src/gateway/`
- `tests/test_demos.py`: subprocess tests use `encoding="utf-8", errors="replace"`
- `demos/_common.py`: `sys.stdout.reconfigure(encoding="utf-8")` on module load

---

## [0.1.0] - 2026-09-07

### Added
- Project structure with all directories (`src/`, `tests/`, `prompts/`, etc.)
- `src/config.py`: Pydantic BaseSettings with environment variable loading
- `src/logger.py`: structured logging (timestamp | name | level | message)
- `src/main.py`: entry point
- `src/stt/`: STT abstraction (Whisper provider, language hint support)
- `src/tts/`: TTS abstraction (OpenAI streaming, TextNormalizer for Indian content)
- `src/llm/`: LLM abstraction (Anthropic Claude Sonnet 4, versioned prompts)
- `src/pipeline/`: orchestrator, types (AgentState, Session, ConversationMessage, PipelineMetrics), baseline test
- `src/gateway/`: WebSocket server, energy-based VAD with hysteresis, audio utils
- `dashboard/index.html` + `styles.css` + `app.js`: voice interaction frontend
- `tests/`: 16 stt, 28 tts, 30 llm, 16 orchestrator, 33 gateway, 10 websocket_server, 19 dashboard = 152 tests
- `flow.md`: decision log (sections 1–22)
- `ARCHITECTURE.md`, `EVALUATION.md`: documentation
- `.env.example`, `requirements.txt`: configuration
- `failures/failure_analysis.md`: framework template (12 failure categories, severity levels)
- `experiments/`: template for experiment tracking

### Security
- `.env` not committed (`.gitignore` updated)
- No API keys in source code
- Structured logging without sensitive user content
