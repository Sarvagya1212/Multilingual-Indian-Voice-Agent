"""LLM provider implementations."""
import time
from typing import List, Dict, Any, Optional, AsyncGenerator

from src.llm.base import (
    LLMProvider,
    Message,
    MessageRole,
    LLMResponse,
    ToolCall,
)
from src.llm.config import LLMConfig
from src.logger import setup_logger

logger = setup_logger(__name__)


class AnthropicLLMProvider(LLMProvider):
    """Anthropic Claude LLM provider.

    Supports tool use, streaming, and multilingual conversation.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        config: LLMConfig = None,
    ):
        """Initialize Anthropic LLM provider.

        Args:
            model: Model name (e.g., claude-sonnet-4-20250514)
            api_key: Anthropic API key (uses env if not provided)
            config: Optional LLMConfig
        """
        self.config = config or LLMConfig()
        self.model = model or self.config.model
        self._latency_ms = 0.0
        self._client = None

        logger.info(f"AnthropicLLMProvider initialized: model={self.model}")

    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(api_key=self._get_api_key())
            except ImportError:
                logger.error("anthropic package not installed. Run: pip install anthropic")
                raise
        return self._client

    def _get_api_key(self) -> Optional[str]:
        """Get API key from env or parameter."""
        from src.config import settings
        return settings.anthropic_api_key

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a chat completion with Claude.

        Args:
            messages: Conversation messages
            tools: Tool schemas (OpenAI format, will be converted)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            LLMResponse with content and tool calls
        """
        start = time.time()

        # Convert messages to Anthropic format
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Convert tools to Anthropic format
        anthropic_tools = self._convert_tools(tools) if tools else None

        # Build request
        request_params = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_prompt:
            request_params["system"] = system_prompt

        if anthropic_tools:
            request_params["tools"] = anthropic_tools

        # Call API
        try:
            response = await self.client.messages.create(**request_params)
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise

        self._latency_ms = (time.time() - start) * 1000

        # Parse response
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        usage = {
            "input_tokens": response.usage.input_tokens if response.usage else 0,
            "output_tokens": response.usage.output_tokens if response.usage else 0,
        }

        logger.info(
            f"LLM: content={len(content)} chars, "
            f"tools={len(tool_calls)}, "
            f"tokens={usage.get('input_tokens', 0)}+{usage.get('output_tokens', 0)}, "
            f"({self._latency_ms:.0f}ms)"
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "stop",
            usage=usage,
        )

    async def chat_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion.

        Args:
            messages: Conversation messages
            tools: Tool schemas
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Yields:
            Text chunks as they are generated
        """
        # Convert messages
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None

        request_params = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_prompt:
            request_params["system"] = system_prompt
        if anthropic_tools:
            request_params["tools"] = anthropic_tools

        try:
            async with self.client.messages.stream(**request_params) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise

    async def detect_language(self, text: str) -> str:
        """Detect the primary language of text.

        Uses character-based detection for Hindi script vs Latin.

        Args:
            text: Input text

        Returns:
            Language code: 'hi' (Hindi), 'en' (English), 'hinglish' (mixed)
        """
        if not text or not text.strip():
            return "en"

        # Count Devanagari characters
        hindi_chars = set(
            "अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
            "ािीुूृेैोौंःँ"
        )

        hindi_count = sum(1 for c in text if c in hindi_chars)
        total_chars = len([c for c in text if c.isalpha()])

        if total_chars == 0:
            return "en"

        hindi_ratio = hindi_count / total_chars

        if hindi_ratio > 0.5:
            return "hi"
        elif hindi_ratio > 0.15:
            return "hinglish"
        else:
            return "en"

    def _convert_messages(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[Dict]]:
        """Convert our messages to Anthropic format.

        Returns:
            (system_prompt, anthropic_messages)
        """
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Anthropic has a single system prompt
                if system_prompt is None:
                    system_prompt = msg.content
                else:
                    system_prompt += "\n\n" + msg.content
            elif msg.role == MessageRole.USER:
                anthropic_messages.append({
                    "role": "user",
                    "content": msg.content,
                })
            elif msg.role == MessageRole.ASSISTANT:
                anthropic_messages.append({
                    "role": "assistant",
                    "content": msg.content,
                })
            elif msg.role == MessageRole.TOOL_RESULT:
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }],
                })

        return system_prompt, anthropic_messages

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI-format tools to Anthropic format.

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "search_courses",
                "description": "...",
                "parameters": {...}
            }
        }

        Anthropic format:
        {
            "name": "search_courses",
            "description": "...",
            "input_schema": {...}
        }
        """
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func["description"],
                    "input_schema": func.get("parameters", {
                        "type": "object",
                        "properties": {},
                    }),
                })
            else:
                # Already in Anthropic format
                anthropic_tools.append(tool)

        return anthropic_tools

    @property
    def latency_ms(self) -> float:
        return self._latency_ms

    @property
    def name(self) -> str:
        return f"anthropic-{self.model}"


# Provider registry
LLM_PROVIDERS = {
    "anthropic": AnthropicLLMProvider,
    "claude": AnthropicLLMProvider,
}


def get_llm_provider(provider_name: Optional[str] = None, **kwargs) -> LLMProvider:
    """Get an LLM provider by name.

    Args:
        provider_name: Provider name (default: from config)
        **kwargs: Provider-specific arguments

    Returns:
        LLMProvider instance
    """
    from src.config import settings

    name = provider_name or settings.llm_provider
    provider_class = LLM_PROVIDERS.get(name)

    if not provider_class:
        raise ValueError(
            f"Unknown LLM provider: {name}. "
            f"Available: {list(LLM_PROVIDERS.keys())}"
        )

    return provider_class(**kwargs)
