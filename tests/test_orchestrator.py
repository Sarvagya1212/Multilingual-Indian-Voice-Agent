"""Tests for pipeline orchestrator."""
import pytest
import io
import wave
from datetime import datetime

from src.pipeline.orchestrator import ConversationOrchestrator
from src.pipeline.types import AgentState, Session, ConversationMessage
from src.stt import get_stt_provider


def create_test_audio(duration: float = 1.0) -> bytes:
    """Create simple test audio bytes."""
    import numpy as np
    import math

    sample_rate = 16000
    frequency = 440
    samples = int(duration * sample_rate)
    audio_data = []

    for i in range(samples):
        value = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * i / sample_rate))
        audio_data.append(value)

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(np.array(audio_data, dtype=np.int16)))
    return buffer.getvalue()


class TestConversationOrchestrator:
    """Tests for ConversationOrchestrator."""

    def test_init(self):
        """Test orchestrator initialization."""
        orchestrator = ConversationOrchestrator()
        assert orchestrator.stt is not None
        assert orchestrator.tts is not None
        assert orchestrator.llm is not None
        assert orchestrator.current_session is None

    def test_create_session(self):
        """Test session creation."""
        orchestrator = ConversationOrchestrator()
        session = orchestrator.create_session()

        assert isinstance(session, Session)
        assert session.id is not None
        assert session.state == AgentState.IDLE
        assert orchestrator.current_session == session

    def test_state_property(self):
        """Test state property."""
        orchestrator = ConversationOrchestrator()
        assert orchestrator.state == AgentState.IDLE

        orchestrator.create_session()
        assert orchestrator.state == AgentState.IDLE

    def test_get_conversation_history_empty(self):
        """Test empty conversation history."""
        orchestrator = ConversationOrchestrator()
        assert orchestrator.get_conversation_history() == []

    def test_get_conversation_history(self):
        """Test conversation history."""
        orchestrator = ConversationOrchestrator()
        session = orchestrator.create_session()

        # Add some messages
        session.conversation.extend([
            ConversationMessage(
                role="user",
                content="Hello",
                timestamp=datetime.now(),
                language="en",
            ),
            ConversationMessage(
                role="agent",
                content="Hi there!",
                timestamp=datetime.now(),
                language="en",
            ),
        ])

        history = orchestrator.get_conversation_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"

    def test_interrupt_no_session(self):
        """Test interrupt with no session."""
        orchestrator = ConversationOrchestrator()
        result = orchestrator.interrupt()
        assert result is False

    def test_interrupt_idle(self):
        """Test interrupt when idle."""
        orchestrator = ConversationOrchestrator()
        orchestrator.create_session()
        result = orchestrator.interrupt()
        assert result is False

    def test_interrupt_speaking(self):
        """Test interrupt when speaking."""
        orchestrator = ConversationOrchestrator()
        orchestrator.create_session()
        orchestrator.current_session.state = AgentState.SPEAKING
        result = orchestrator.interrupt()
        assert result is True
        assert orchestrator.state == AgentState.INTERRUPTED

    def test_interrupt_generating(self):
        """Test interrupt when generating."""
        orchestrator = ConversationOrchestrator()
        orchestrator.create_session()
        orchestrator.current_session.state = AgentState.GENERATING
        result = orchestrator.interrupt()
        assert result is True
        assert orchestrator.state == AgentState.INTERRUPTED

    def test_interrupt_listening(self):
        """Test interrupt when listening (should fail)."""
        orchestrator = ConversationOrchestrator()
        orchestrator.create_session()
        orchestrator.current_session.state = AgentState.LISTENING
        result = orchestrator.interrupt()
        assert result is False


class TestAgentState:
    """Tests for AgentState enum."""

    def test_all_states_exist(self):
        """Test all expected states exist."""
        expected_states = [
            "IDLE",
            "LISTENING",
            "TRANSCRIBING",
            "THINKING",
            "CALLING_TOOL",
            "GENERATING",
            "SPEAKING",
            "INTERRUPTED",
            "ERROR",
        ]
        for state_name in expected_states:
            assert hasattr(AgentState, state_name)
            state = getattr(AgentState, state_name)
            assert state.value == state_name.lower()

    def test_state_values(self):
        """Test state values are lowercase strings."""
        for state in AgentState:
            assert isinstance(state.value, str)
            assert state.value == state.value.lower()


class TestSession:
    """Tests for Session dataclass."""

    def test_session_creation(self):
        """Test session creation."""
        session = Session(
            id="test-123",
            created_at=datetime.now(),
            state=AgentState.IDLE,
        )
        assert session.id == "test-123"
        assert session.state == AgentState.IDLE
        assert session.conversation == []
        assert session.user_context == {}

    def test_session_with_context(self):
        """Test session with user context."""
        session = Session(
            id="test-456",
            created_at=datetime.now(),
            state=AgentState.IDLE,
            user_context={"language": "en", "interest": "JEE"},
        )
        assert session.user_context["language"] == "en"
        assert session.user_context["interest"] == "JEE"


class TestConversationMessage:
    """Tests for ConversationMessage dataclass."""

    def test_message_creation(self):
        """Test message creation."""
        msg = ConversationMessage(
            role="user",
            content="Hello",
            timestamp=datetime.now(),
            language="en",
        )
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.language == "en"
        assert msg.audio_duration == 0.0

    def test_message_with_metadata(self):
        """Test message with metadata."""
        msg = ConversationMessage(
            role="agent",
            content="Hi!",
            timestamp=datetime.now(),
            language="en",
            audio_duration=1.5,
            metadata={"stt_confidence": 0.95},
        )
        assert msg.audio_duration == 1.5
        assert msg.metadata["stt_confidence"] == 0.95


# Run tests with: pytest tests/test_orchestrator.py -v
