# Demos

Executable demonstrations of the Multilingual Indian Voice Agent's key
features. Each demo is a standalone script that runs without API keys
(using deterministic stubs for STT/LLM/TTS) so it works out-of-the-box
in CI and for portfolio reviewers.

> **Want the real, live voice experience?** Run `python run_live.py` from the
> project root. It starts the full pipeline (STT → LLM → TTS) with a browser
> UI where you press-and-hold the mic to talk. See the main
> [README](../README.md) for details.

## Running Demos

```bash
# Run all 8 demos
python demos/run_all.py

# Run individual demos
python demos/demo_01_english.py
python demos/demo_02_hindi.py
python demos/demo_03_hinglish.py
python demos/demo_04_tool_calling.py
python demos/demo_05_rag.py
python demos/demo_06_interruption.py
python demos/demo_07_failure_recovery.py
python demos/demo_08_dashboard.py

# Run a specific subset
python demos/run_all.py 1 3 5

# Stop on first failure
python demos/run_all.py --stop
```

## Demo Descriptions

### Demo 1: English Conversation
Basic single-turn English interaction. Demonstrates the STT -> LLM -> TTS
pipeline end-to-end. No API keys required (uses a deterministic simulation).

### Demo 2: Hindi Conversation
Full Hindi conversation (Devanagari script). The system detects Hindi
via the Devanagari script ratio, locks the response to Hindi, and
generates a Hindi TTS output.

### Demo 3: Hinglish / Code-Switching
Natural Hindi-English code-switching in a single utterance. Demonstrates
the multi-signal language detector (script ratio + function-word presence)
and the Hinglish response style in the system prompt.

### Demo 4: Tool Calling
The LLM decides which tool to call (search_courses,
get_course_details, check_eligibility, get_fee_structure,
schedule_demo), executes it, and incorporates the result into the response.
Uses the mock tool registry — the real one is in `src/agent/tools/`.

### Demo 5: RAG (Retrieval-Augmented Generation)
Queries the knowledge base with 3 test queries, retrieves top-k chunks,
and builds the context string that would be passed to the LLM.
Uses the **real** RAG pipeline — LocalEmbedder is hash-based and needs
no API key.

### Demo 6: Interruption / Barge-in
The user interrupts the agent mid-response. The state machine halts the
SPEAKING state and transitions to INTERRUPTED, allowing the next turn
to begin immediately.

### Demo 7: Failure Recovery
Three simulated error scenarios (STT failure, LLM timeout, tool error).
The state machine transitions to ERROR and recovers to IDLE in each case.
Demonstrates graceful degradation rather than crashes.

### Demo 8: Dashboard
Exercises the dashboard data layer with the synthetic event log that
populates `logs/demo_events.jsonl` on first run. Prints the same KPIs
the Streamlit dashboard renders. To launch the real UI:

```bash
streamlit run dashboard/app.py
```

## With Real API Keys

Set the following environment variables to exercise the live pipeline:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_WHISPER_API_KEY=...   # optional; uses local Whisper if absent
```

Each demo checks for these keys and prints `[SIM]` for steps that are
simulated vs `[OK]` for real ones. The script itself does not gate on
presence of keys — it always produces a meaningful output.

## Architecture (demo -> real)

| Demo component | Real component |
|---------------|---------------|
| `SimulatedAgent` | `ConversationOrchestrator` / `AgentOrchestrator` |
| `fake_tts_audio()` | `OpenAITTSProvider.synthesize_stream()` |
| Mock tool registry | `src/agent/tools/tool_registry.py` |
| `load_events()` | `logs/events.jsonl` written by pipeline at runtime |
