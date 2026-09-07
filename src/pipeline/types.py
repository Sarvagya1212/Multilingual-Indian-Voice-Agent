"""Pipeline types - shared data classes for the voice agent."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class AgentState(Enum):
    """Agent conversation state machine."""
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    CALLING_TOOL = "calling_tool"
    GENERATING = "generating"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"


@dataclass
class ConversationMessage:
    """A single message in the conversation."""
    role: str  # "user" or "agent"
    content: str
    timestamp: datetime
    language: str = "en"
    audio_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """A conversation session with one user."""
    id: str
    created_at: datetime
    state: AgentState
    conversation: List[ConversationMessage] = field(default_factory=list)
    user_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineMetrics:
    """Metrics for one pipeline turn."""
    session_id: str
    turn_id: int
    timestamp: datetime
    language: str
    # Latencies in milliseconds
    stt_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tts_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    rag_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    # Counts
    tool_calls: int = 0
    interruptions: int = 0
    # Errors
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp.isoformat(),
            "language": self.language,
            "latency": {
                "stt_ms": self.stt_latency_ms,
                "llm_ms": self.llm_latency_ms,
                "tts_ms": self.tts_latency_ms,
                "tool_ms": self.tool_latency_ms,
                "rag_ms": self.rag_latency_ms,
                "total_ms": self.total_latency_ms,
            },
            "tool_calls": self.tool_calls,
            "interruptions": self.interruptions,
            "errors": self.errors,
        }
