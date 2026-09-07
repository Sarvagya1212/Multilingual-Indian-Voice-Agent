"""Full agent orchestrator — state machine + tools + memory + pipeline.

Extends the baseline pipeline (STT → LLM → TTS) with:
- Formal state machine (AgentStateMachine)
- Tool calling (via ToolRegistry)
- Short-term memory (ConversationTurn history)
- Tool latency telemetry
- Interrupt handling via state machine
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.agent.config import AgentConfig
from src.agent.memory.short_term import ShortTermMemory
from src.agent.state_machine import AgentStateMachine
from src.agent.tools.tool_registry import tool_registry as default_registry, ToolRegistry
from src.llm import get_llm_provider, Message, MessageRole, LLMResponse
from src.llm.config import load_prompt, build_system_prompt
from src.pipeline.types import AgentState
from src.stt import get_stt_provider, STTResult
from src.tts import get_tts_provider, TTSResult
from src.logger import setup_logger

logger = setup_logger(__name__)


class AgentOrchestrator:
    """Full agent orchestrator.

    Pipeline: STT → LLM (with tool calls) → TTS

    Uses AgentStateMachine for formal state transitions,
    ShortTermMemory for conversation context, and
    ToolRegistry for tool calling.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        tools: ToolRegistry | None = None,
        stt_provider: str | None = None,
        llm_provider: str | None = None,
        tts_provider: str | None = None,
    ):
        self.config = config or AgentConfig()
        self.stt = get_stt_provider(stt_provider)
        self.llm = get_llm_provider(llm_provider)
        self.tts = get_tts_provider(tts_provider)
        self.state_machine = AgentStateMachine()
        self.memory = ShortTermMemory(
            max_turns=self.config.max_turns,
            ttl_minutes=self.config.memory_ttl_minutes,
        )
        self.tools = tools if tools is not None else default_registry
        self.metrics: Dict[str, Any] = {
            "stt_latency_ms": 0.0,
            "llm_latency_ms": 0.0,
            "tts_latency_ms": 0.0,
            "tool_latency_ms": 0.0,
            "total_latency_ms": 0.0,
            "tool_calls": 0,
            "interruptions": 0,
        }

        logger.info(
            f"AgentOrchestrator initialised: "
            f"tools=[{', '.join(self.tools.list_names())}]"
        )

    @property
    def state(self) -> AgentState:
        """Current agent state."""
        return self.state_machine.state

    async def process_turn(self, audio: bytes) -> TTSResult:
        """Process one complete conversation turn.

        Steps:
        1. STT: audio → text + language
        2. THINKING: LLM → response (with tool calls)
        3. GENERATING: TTS → audio response

        Args:
            audio: Raw audio bytes (WAV or PCM).

        Returns:
            TTSResult with synthesised speech.
        """
        start = time.time()

        # 1. Transcribe
        self.state_machine.transition(AgentState.TRANSCRIBING, "start_processing")
        stt_start = time.time()
        stt_result: STTResult = await self.stt.transcribe(audio)
        self.metrics["stt_latency_ms"] = (time.time() - stt_start) * 1000
        logger.info(
            f"[STT] {self.metrics['stt_latency_ms']:.0f}ms: "
            f"'{stt_result.text[:50]}' [{stt_result.language}]"
        )

        if not stt_result.text.strip():
            logger.warning("[STT] Empty transcription — skipping turn")
            self.state_machine.transition(AgentState.IDLE, "empty_transcription")
            return TTSResult(audio=b"", duration=0.0)

        # 2. Generate response (LLM + tools)
        self.state_machine.transition(AgentState.THINKING, "start_llm")
        llm_start = time.time()
        response: LLMResponse = await self._generate_response(
            stt_result.text,
            stt_result.language,
        )
        self.metrics["llm_latency_ms"] = (time.time() - llm_start) * 1000
        logger.info(
            f"[LLM] {self.metrics['llm_latency_ms']:.0f}ms: "
            f"'{response.content[:80]}' [{self.metrics['tool_calls']} tool calls]"
        )

        # 3. Synthesize
        self.state_machine.transition(AgentState.GENERATING, "start_tts")
        tts_start = time.time()
        tts_result: TTSResult = await self.tts.synthesize(
            response.content,
            language=stt_result.language,
        )
        self.metrics["tts_latency_ms"] = (time.time() - tts_start) * 1000
        logger.info(
            f"[TTS] {self.metrics['tts_latency_ms']:.0f}ms: "
            f"{tts_result.duration:.1f}s audio"
        )

        # 4. Record to memory
        self.memory.add_turn(
            user_message=stt_result.text,
            agent_message=response.content,
            language=stt_result.language,
            tools_used=self._get_tool_names(response),
        )

        # 5. Transition to IDLE
        self.state_machine.transition(AgentState.IDLE, "turn_complete")
        self.metrics["total_latency_ms"] = (time.time() - start) * 1000
        logger.info(
            f"[Turn] Total: {self.metrics['total_latency_ms']:.0f}ms "
            f"(STT={self.metrics['stt_latency_ms']:.0f}, "
            f"LLM={self.metrics['llm_latency_ms']:.0f}, "
            f"TTS={self.metrics['tts_latency_ms']:.0f})"
        )

        return tts_result

    async def _generate_response(
        self,
        user_text: str,
        language: str,
    ) -> LLMResponse:
        """Generate a response using the LLM with tool calling.

        Implements two-pass tool calling:
        1. Call LLM with tool schema → LLM decides to call tools
        2. Execute tools → Feed results back to LLM → Final response

        Args:
            user_text: User's transcribed text.
            language: Detected language.

        Returns:
            LLMResponse with content and optional tool_calls.
        """
        system_prompt = self._build_system_prompt(language)
        context = self.memory.get_recent_context()

        messages: List[Message] = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
        ]

        if context:
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=f"Recent conversation:\n{context}",
                )
            )

        messages.append(Message(role=MessageRole.USER, content=user_text))

        # First pass: LLM with tools
        tools_schema = self.tools.get_tools_schema()
        response: LLMResponse = await self.llm.chat(
            messages=messages,
            tools=tools_schema,
            temperature=0.7,
            max_tokens=512,
        )

        self.metrics["tool_calls"] = len(response.tool_calls)

        # Handle tool calls if any
        if response.tool_calls:
            self.state_machine.transition(
                AgentState.CALLING_TOOL, "executing_tools"
            )

            tool_messages: List[Message] = [
                Message(role=MessageRole.ASSISTANT, content=response.content),
            ]

            for tc in response.tool_calls:
                tool_start = time.time()
                result = await self.tools.execute_tool(tc.name, tc.arguments)
                self.metrics["tool_latency_ms"] += (time.time() - tool_start) * 1000

                content = (
                    f"Tool '{tc.name}' result: {result.data}"
                    if result.success
                    else f"Tool '{tc.name}' error: {result.error}"
                )
                tool_messages.append(
                    Message(
                        role=MessageRole.TOOL_RESULT,
                        content=content,
                        tool_call_id=tc.id,
                    )
                )

            # Second pass: LLM with tool results
            messages.extend(tool_messages)
            response = await self.llm.chat(
                messages=messages,
                tools=[],  # No tools on second pass
                temperature=0.7,
                max_tokens=512,
            )

        return response

    def _build_system_prompt(self, language: str) -> str:
        """Build the system prompt with agent persona and user context."""
        parts = [
            load_prompt("system_v1"),
            load_prompt("tool_use_v1"),
        ]

        if language == "hi":
            parts.append("Respond primarily in Hindi.")
        elif language == "hinglish":
            parts.append("Respond in natural Hinglish (Hindi written in Roman script).")
        else:
            parts.append("Respond in English.")

        # Inject known user context
        summary = self.memory.get_context_summary()
        if summary["user_context"]:
            ctx_parts = [
                f"- {k}: {v}" for k, v in summary["user_context"].items()
            ]
            parts.append("\nKnown user context:\n" + "\n".join(ctx_parts))

        return "\n\n".join(parts)

    def _get_tool_names(self, response: LLMResponse) -> List[str]:
        return [tc.name for tc in response.tool_calls]

    def interrupt(self) -> bool:
        """Handle a user interruption (barge-in).

        Returns:
            True if interruption was handled.
        """
        if self.state_machine.can_interrupt():
            self.state_machine.force_interrupt()
            self.metrics["interruptions"] += 1
            logger.info("[Interrupt] User interruption handled")
            return True
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Return current telemetry."""
        return {
            **self.metrics,
            "state": self.state.value,
            "context": self.memory.get_context_summary(),
        }
