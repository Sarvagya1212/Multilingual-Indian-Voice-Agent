"""LLM (Large Language Model) abstraction layer."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncGenerator
from enum import Enum


class MessageRole(Enum):
    """Message role in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"


@dataclass
class Message:
    """A single message in a conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


@dataclass
class ToolCall:
    """A tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """Response from the LLM."""
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM implementations must inherit from this class.
    """

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a chat completion.

        Args:
            messages: List of conversation messages
            tools: Optional tool schemas (OpenAI format)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with content and optional tool calls
        """
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion.

        Args:
            messages: List of conversation messages
            tools: Optional tool schemas
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Text chunks as they are generated
        """
        pass

    @abstractmethod
    async def detect_language(self, text: str) -> str:
        """Detect the primary language of text.

        Args:
            text: Input text

        Returns:
            Language code (en, hi, hinglish)
        """
        pass

    @property
    @abstractmethod
    def latency_ms(self) -> float:
        """Average latency in milliseconds."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
