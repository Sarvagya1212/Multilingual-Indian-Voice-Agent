# Research: Voice Agent Architecture

## Paper/Source

**Voice Agent Patterns: From Turn-Taking to Multi-Modal Dialogue**

- Type: Survey + industry patterns (inspired by Siri/Google Assistant architecture blogs)
- URL: https://developer.apple.com/documentation/sirikit
- Related: LangChain Agents (https://docs.langchain.com/docs/get_started/introduction)

## Problem

A real-time voice agent must handle: multi-step reasoning, tool calls, memory
across turns, language switching, and interruption — all with sub-5-second
per-turn latency. A naive LLM-in-the-loop approach (one giant prompt)
fails on latency, coherence, and tool-calling reliability. We need an
explicit architecture that constrains the system's behaviour.

## Key Approach

- **State machine for dialogue management**: the agent is always in exactly
  one state (IDLE, LISTENING, TRANSCRIBING, THINKING, CALLING_TOOL, GENERATING,
  SPEAKING, INTERRUPTED, ERROR). Transitions are explicit; invalid transitions
  are rejected.
- **Structured tool registry**: each tool is a named, versioned function with
  a schema. The LLM calls tools by name; a registry validates arguments,
  enforces timeouts, and retries on transient failures.
- **Two-pass tool calling**: (1) LLM decides which tool to call and with what
  args; (2) tool executes; (3) results are appended to the conversation;
  (4) LLM produces the final response. This prevents the LLM from
  hallucinating tool results.
- **Sliding-window memory**: last N conversation turns are kept in the prompt;
  structured facts (class, exam interest, preferred language) are extracted
  and passed as a separate context dict.

## Relevant Ideas for Our Project

1. **Explicit state machine over implicit reasoning.** The LLM decides
   what to do; the state machine decides what it is allowed to do next.
   This bounds the agent — even a confused LLM cannot jump from IDLE
   to SPEAKING without going through TRANSCRIBING and THINKING.
2. **Two-pass tool calling.** Our `AgentOrchestrator` (Prompt 3.1) implements
   this pattern: LLM decides → tool executes → results appended → LLM
   finalises. This is the production pattern used by Claude, GPT-4, and
   most commercial assistants.
3. **Structured tool registry.** A registry (not ad-hoc function passing)
   makes tools discoverable, testable, and versioned. We validate schemas
   with Pydantic and enforce 30s timeouts.
4. **Bounded memory.** Passing the entire conversation to the LLM is
   expensive and slow; sliding-window + structured fact extraction is
   the right trade-off for a counselling use case.

## What We Implemented

- [x] **Formal state machine** (`src/agent/state_machine.py`): explicit
  state enum + valid transition graph; all transitions are logged.
  Valid states: IDLE, LISTENING, TRANSCRIBING, THINKING,
  CALLING_TOOL, GENERATING, SPEAKING, INTERRUPTED, ERROR.
- [x] **Tool registry** (`src/agent/tools/tool_registry.py`): 5 tools
  (search_courses, get_course_details, check_eligibility,
  get_fee_structure, schedule_demo) with Pydantic validation and
  structured logging.
- [x] **Two-pass tool calling** (`src/agent/orchestrator.py`):
  LLM decides → executes → appends results → LLM finalises.
- [x] **Short-term memory** (`src/agent/memory/short_term.py`):
  sliding window (max 10 turns), TTL, keyword-based fact extraction.
- [x] **Barge-in interrupt** (`ConversationOrchestrator.interrupt()`):
  halts SPEAKING/GENERATING; next utterance starts immediately.
- [x] **Long-term memory interface** (`src/agent/memory/long_term.py`):
  stub with Redis/SQLite drop-in; not yet connected.

## What We Did NOT Implement

- **LLM-guided state transitions** — we use a fixed transition graph; a more
  sophisticated agent would let the LLM suggest the next state and have the
  machine validate it. This adds latency and complexity; the fixed graph is
  sufficient for a turn-based counselling use case.
- **Multi-turn tool chains** — the current two-pass handles single tool calls.
  Multi-step chains (search → check eligibility → schedule) require the LLM
  to decide after each result. Planned for a future enhancement.
- **Long-term memory backend** — the interface exists; the production swap
  (Redis or PostgreSQL) is deferred until after the MVP demo.
- **Structured output parsing for tool arguments** — we rely on the LLM's
  JSON output and Pydantic validation; a structured-output mode (Claude's
  `input_schema`) would be more reliable but requires provider support.

## Results

- State machine passes all 49 agent tests including invalid transition
  rejection and interrupt handling.
- Tool registry: all 5 tools load and execute in the agent tests.
- Two-pass tool calling: integrated with `AgentOrchestrator.process_turn()`;
  tool results are correctly appended to the conversation history.
- Barge-in: interrupt() returns True for SPEAKING and GENERATING states;
  False for LISTENING and IDLE (user mid-utterance cannot be interrupted).

## Inspiration Statement

The state machine + tool registry + two-pass pattern is the industry
standard for production voice agents (Siri, Google Assistant, Alexa all use
variants). We ship the minimal version that meets our latency and
reliability targets; the architecture scales to more complex use cases
without re-engineering.
