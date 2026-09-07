"""Conversation orchestrator - wires STT → LLM → TTS together."""
import uuid
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

from src.stt import get_stt_provider, STTResult
from src.tts import get_tts_provider, TTSResult
from src.llm import get_llm_provider, Message, MessageRole, LLMResponse
from src.llm.config import load_prompt
from src.pipeline.types import (
    AgentState,
    Session,
    ConversationMessage,
    PipelineMetrics,
)
from src.logger import setup_logger

# Telemetry is optional — never crash the pipeline if it's missing.
try:
    from src.telemetry import write_turn_event
except Exception:
    write_turn_event = None  # type: ignore[assignment, misc]

logger = setup_logger(__name__)


class ConversationOrchestrator:
    """Main orchestrator for the voice agent.

    Coordinates:
    - STT for transcription
    - LLM for reasoning and tool use
    - TTS for speech synthesis
    - Memory for conversation context
    """

    def __init__(
        self,
        stt_provider: Optional[str] = None,
        tts_provider: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ):
        """Initialize the orchestrator.

        Args:
            stt_provider: STT provider name (default: from config)
            tts_provider: TTS provider name (default: from config)
            llm_provider: LLM provider name (default: from config)
        """
        # Initialize providers
        self.stt = get_stt_provider(stt_provider)
        self.tts = get_tts_provider(tts_provider)
        self.llm = get_llm_provider(llm_provider)

        # State
        self.current_session: Optional[Session] = None
        self._turn_count = 0

        # Load system prompts
        self._system_prompt = load_prompt("system_v1")
        self._tool_prompt = load_prompt("tool_use_v1")
        self._multilingual_prompt = load_prompt("multilingual_v1")

        logger.info("ConversationOrchestrator initialized")

    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self.current_session.state if self.current_session else AgentState.IDLE

    def create_session(self) -> Session:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        self.current_session = Session(
            id=session_id,
            created_at=datetime.now(),
            state=AgentState.IDLE,
            conversation=[],
            user_context={},
        )
        self._turn_count = 0
        logger.info(f"Created session: {session_id}")
        return self.current_session

    async def process_turn(self, audio: bytes) -> TTSResult:
        """Process one complete conversation turn.

        Pipeline:
        1. STT: Audio → Text
        2. LLM: Text → Response (+ tool calls if needed)
        3. TTS: Text → Audio

        Args:
            audio: Audio bytes (WAV format)

        Returns:
            TTSResult with synthesized speech
        """
        start_time = time.time()
        turn_start = datetime.now()
        self._turn_count += 1

        # Ensure we have a session
        if not self.current_session:
            self.create_session()

        metrics = PipelineMetrics(
            session_id=self.current_session.id,
            turn_id=self._turn_count,
            timestamp=turn_start,
            language="en",
        )

        try:
            # Update state
            self.current_session.state = AgentState.TRANSCRIBING

            # Step 1: STT - Transcribe audio to text
            stt_start = time.time()
            stt_result: STTResult = await self.stt.transcribe(audio)
            metrics.stt_latency_ms = (time.time() - stt_start) * 1000
            metrics.language = stt_result.language

            logger.info(
                f"STT ({metrics.stt_latency_ms:.0f}ms): "
                f"'{stt_result.text[:50]}...' [{stt_result.language}]"
            )

            if not stt_result.text.strip():
                logger.warning("Empty transcription, skipping")
                return TTSResult(audio=b"", duration=0.0)

            # Update state
            self.current_session.state = AgentState.THINKING

            # Step 2: LLM - Generate response
            llm_start = time.time()
            response: LLMResponse
            tool_names: List[str]
            response, tool_names = await self._generate_response(
                stt_result.text,
                stt_result.language,
            )
            metrics.llm_latency_ms = (time.time() - llm_start) * 1000
            metrics.tool_calls = len(tool_names)

            logger.info(
                f"LLM ({metrics.llm_latency_ms:.0f}ms): "
                f"'{response.content[:50]}...' [{len(response.tool_calls)} tools]"
            )

            # Update state
            self.current_session.state = AgentState.GENERATING

            # Step 3: TTS - Synthesize speech
            tts_start = time.time()
            tts_result: TTSResult = await self.tts.synthesize(
                response.content,
                language=stt_result.language,
            )
            metrics.tts_latency_ms = (time.time() - tts_start) * 1000

            logger.info(
                f"TTS ({metrics.tts_latency_ms:.0f}ms): {tts_result.duration:.1f}s audio"
            )

            # Calculate total
            metrics.total_latency_ms = (time.time() - start_time) * 1000

            # Record conversation
            self.current_session.conversation.extend([
                ConversationMessage(
                    role="user",
                    content=stt_result.text,
                    timestamp=turn_start,
                    language=stt_result.language,
                ),
                ConversationMessage(
                    role="agent",
                    content=response.content,
                    timestamp=datetime.now(),
                    language=stt_result.language,
                ),
            ])

            # Update state
            self.current_session.state = AgentState.IDLE

            # Log metrics
            logger.info(f"Turn complete: {metrics.total_latency_ms:.0f}ms total")

            # Live telemetry — append a turn event to logs/events.jsonl when enabled.
            # No-op when telemetry is disabled or the writer is unavailable.
            if write_turn_event is not None:
                try:
                    write_turn_event(
                        session_id=self.current_session.id,
                        language=stt_result.language,
                        transcript=stt_result.text,
                        response=response.content,
                        stt_latency_ms=metrics.stt_latency_ms,
                        llm_latency_ms=metrics.llm_latency_ms,
                        tts_latency_ms=metrics.tts_latency_ms,
                        total_latency_ms=metrics.total_latency_ms,
                        tools_used=tool_names,
                    )
                except Exception as telemetry_err:
                    logger.warning(f"Telemetry write failed: {telemetry_err}")

            return tts_result

        except Exception as e:
            logger.error(f"Error processing turn: {e}")
            self.current_session.state = AgentState.ERROR
            metrics.errors.append(str(e))
            metrics.total_latency_ms = (time.time() - start_time) * 1000
            raise

    async def _generate_response(
        self,
        user_text: str,
        language: str,
    ) -> tuple[LLMResponse, List[str]]:
        """Generate response using LLM with tool support.

        Args:
            user_text: User's transcribed text
            language: Detected language

        Returns:
            LLM response with content and optional tool calls
        """
        # Build system prompt
        system_prompt = self._build_system_prompt(language)

        # Build messages
        messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
        ]

        # Add recent conversation context (last 5 turns)
        recent = self.current_session.conversation[-10:] if self.current_session.conversation else []
        for msg in recent:
            role = MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT
            messages.append(Message(role=role, content=msg.content))

        # Current user message
        messages.append(Message(role=MessageRole.USER, content=user_text))

        # Get tool schemas (empty for baseline - tools in Prompt 3.1)
        tools = []

        # Call LLM
        response: LLMResponse = await self.llm.chat(
            messages=messages,
            tools=tools,
            temperature=0.7,
            max_tokens=512,
        )

        tool_names = [tc.name for tc in response.tool_calls]
        return response, tool_names

    def _build_system_prompt(self, language: str) -> str:
        """Build combined system prompt.

        Args:
            language: Detected language

        Returns:
            Combined system prompt
        """
        parts = [self._system_prompt]

        # Add tool instructions
        parts.append(self._tool_prompt)

        # Add multilingual instructions
        if language == "hi":
            parts.append("Respond primarily in Hindi.")
        elif language == "hinglish":
            parts.append("Respond in natural Hinglish (Hindi written in Roman script).")
        else:
            parts.append("Respond in English.")

        # Add known user context
        if self.current_session and self.current_session.user_context:
            context = self.current_session.user_context
            context_parts = []
            for key, value in context.items():
                context_parts.append(f"- {key}: {value}")
            if context_parts:
                parts.append(f"\nKnown user context:\n" + "\n".join(context_parts))

        return "\n\n".join(parts)

    def get_metrics(self) -> Optional[PipelineMetrics]:
        """Get metrics for the current turn."""
        # This would be populated from process_turn
        return None

    def interrupt(self) -> bool:
        """Handle user interruption (barge-in).

        Returns:
            True if successfully interrupted
        """
        if self.current_session and self.current_session.state in (
            AgentState.SPEAKING,
            AgentState.GENERATING,
        ):
            self.current_session.state = AgentState.INTERRUPTED
            logger.info("User interruption handled")
            return True
        return False

    def get_conversation_history(self) -> list:
        """Get conversation history for current session."""
        if not self.current_session:
            return []
        return [
            {"role": msg.role, "content": msg.content, "language": msg.language}
            for msg in self.current_session.conversation
        ]
