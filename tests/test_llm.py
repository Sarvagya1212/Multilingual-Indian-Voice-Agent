"""Tests for LLM module."""
import pytest
import os

from src.llm import (
    Message,
    MessageRole,
    LLMResponse,
    ToolCall,
    load_prompt,
    build_system_prompt,
    PROMPT_FILES,
)
from src.llm.config import LLMConfig
from src.llm.providers import AnthropicLLMProvider, get_llm_provider


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.name is None
        assert msg.tool_call_id is None

    def test_message_with_metadata(self):
        msg = Message(
            role=MessageRole.TOOL_RESULT,
            content="Result",
            tool_call_id="tool_123",
        )
        assert msg.tool_call_id == "tool_123"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_response_creation(self):
        response = LLMResponse(content="Hello")
        assert response.content == "Hello"
        assert response.tool_calls == []
        assert response.finish_reason == "stop"
        assert response.usage == {}

    def test_response_with_tool_calls(self):
        tool_call = ToolCall(
            id="call_1",
            name="search_courses",
            arguments={"query": "JEE"},
        )
        response = LLMResponse(
            content="",
            tool_calls=[tool_call],
            finish_reason="tool_use",
        )
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search_courses"


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_config(self):
        config = LLMConfig()
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.7

    def test_custom_config(self):
        config = LLMConfig(
            model="claude-sonnet-4-20250514",
            temperature=0.3,
            max_tokens=512,
        )
        assert config.temperature == 0.3
        assert config.max_tokens == 512


class TestPromptLoading:
    """Tests for prompt file loading."""

    def test_load_system_prompt(self):
        prompt = load_prompt("system_v1")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "education" in prompt.lower() or "course" in prompt.lower()

    def test_load_tool_prompt(self):
        prompt = load_prompt("tool_use_v1")
        assert isinstance(prompt, str)
        assert "tool" in prompt.lower() or "course" in prompt.lower()

    def test_load_multilingual_prompt(self):
        prompt = load_prompt("multilingual_v1")
        assert isinstance(prompt, str)
        assert any(lang in prompt.lower() for lang in ["english", "hindi", "hinglish"])

    def test_load_unknown_prompt(self):
        with pytest.raises(ValueError, match="Unknown prompt"):
            load_prompt("unknown_prompt")

    def test_prompt_files_mapping(self):
        assert "system_v1" in PROMPT_FILES
        assert "tool_use_v1" in PROMPT_FILES
        assert "multilingual_v1" in PROMPT_FILES

    def test_build_system_prompt(self):
        system = "You are a counsellor."
        tool = "Use search_courses tool."
        multilingual = "Respond in Hinglish."

        combined = build_system_prompt(system, tool, multilingual)
        assert "counsellor" in combined
        assert "search_courses" in combined
        assert "Hinglish" in combined

    def test_build_system_prompt_no_tools(self):
        system = "You are a counsellor."
        combined = build_system_prompt(system)
        assert combined == "You are a counsellor."


class TestAnthropicProvider:
    """Tests for AnthropicLLMProvider."""

    def test_provider_init(self):
        provider = AnthropicLLMProvider()
        assert provider.model == "claude-sonnet-4-20250514"
        assert provider.name == "anthropic-claude-sonnet-4-20250514"

    def test_provider_custom_model(self):
        provider = AnthropicLLMProvider(model="claude-sonnet-4-20250514")
        assert provider.model == "claude-sonnet-4-20250514"

    def test_get_llm_provider_default(self):
        provider = get_llm_provider()
        assert isinstance(provider, AnthropicLLMProvider)

    def test_get_llm_provider_by_name(self):
        provider = get_llm_provider("anthropic")
        assert isinstance(provider, AnthropicLLMProvider)

    def test_get_llm_provider_claude_alias(self):
        provider = get_llm_provider("claude")
        assert isinstance(provider, AnthropicLLMProvider)

    def test_get_llm_provider_unknown(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider("unknown_provider")


@pytest.mark.asyncio
class TestLanguageDetection:
    """Tests for language detection (no API call needed)."""

    async def test_detect_english(self):
        provider = AnthropicLLMProvider()
        lang = await provider.detect_language("Hello, how are you?")
        assert lang == "en"

    async def test_detect_hindi(self):
        provider = AnthropicLLMProvider()
        lang = await provider.detect_language("मुझे जेईई की तैयारी करनी है")
        assert lang == "hi"

    async def test_detect_hinglish(self):
        provider = AnthropicLLMProvider()
        # Pure Roman transliteration - the heuristic returns "en" by default
        # since there are no Devanagari chars. That's expected behavior
        # for the simple character-based detection.
        lang = await provider.detect_language("JEE ke liye preparation karna hai")
        # This is Roman transliteration, so it will be "en" - that's correct
        # for our character-based detection. Real detection needs LLM call.
        assert lang == "en"

    async def test_detect_hinglish_with_devanagari(self):
        provider = AnthropicLLMProvider()
        # Mixed: Devanagari + English
        lang = await provider.detect_language("मैं JEE की preparation कर रहा हूं")
        # Should be hinglish (mix of Hindi and English)
        assert lang in ("hi", "hinglish")

    async def test_detect_empty(self):
        provider = AnthropicLLMProvider()
        lang = await provider.detect_language("")
        assert lang == "en"

    async def test_detect_hindi_mixed(self):
        provider = AnthropicLLMProvider()
        # Mostly Hindi
        text = "मैं JEE exam की तैयारी कर रहा हूं"
        lang = await provider.detect_language(text)
        # Could be 'hi' or 'hinglish' depending on ratio
        assert lang in ("hi", "hinglish")


class TestMessageConversion:
    """Tests for message format conversion."""

    def test_convert_simple_messages(self):
        provider = AnthropicLLMProvider()
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.USER, content="Hello!"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!"),
        ]
        system, anthropic_msgs = provider._convert_messages(messages)
        assert system == "You are helpful."
        assert len(anthropic_msgs) == 2
        assert anthropic_msgs[0]["role"] == "user"
        assert anthropic_msgs[1]["role"] == "assistant"

    def test_convert_multiple_system(self):
        provider = AnthropicLLMProvider()
        messages = [
            Message(role=MessageRole.SYSTEM, content="First system."),
            Message(role=MessageRole.SYSTEM, content="Second system."),
            Message(role=MessageRole.USER, content="Hi"),
        ]
        system, _ = provider._convert_messages(messages)
        assert "First system" in system
        assert "Second system" in system

    def test_convert_tool_result(self):
        provider = AnthropicLLMProvider()
        messages = [
            Message(role=MessageRole.USER, content="Search JEE"),
            Message(role=MessageRole.ASSISTANT, content=""),
            Message(
                role=MessageRole.TOOL_RESULT,
                content="Found 3 courses",
                tool_call_id="call_1",
            ),
        ]
        _, anthropic_msgs = provider._convert_messages(messages)
        # Last message should be user with tool_result content block
        assert anthropic_msgs[-1]["role"] == "user"
        assert isinstance(anthropic_msgs[-1]["content"], list)
        assert anthropic_msgs[-1]["content"][0]["type"] == "tool_result"


class TestToolConversion:
    """Tests for tool format conversion."""

    def test_convert_openai_format(self):
        provider = AnthropicLLMProvider()
        openai_tools = [{
            "type": "function",
            "function": {
                "name": "search_courses",
                "description": "Search courses",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                },
            },
        }]
        anthropic_tools = provider._convert_tools(openai_tools)
        assert len(anthropic_tools) == 1
        assert anthropic_tools[0]["name"] == "search_courses"
        assert anthropic_tools[0]["description"] == "Search courses"
        assert "input_schema" in anthropic_tools[0]

    def test_convert_passthrough(self):
        provider = AnthropicLLMProvider()
        anthropic_tools_in = [{
            "name": "test",
            "description": "Test tool",
            "input_schema": {"type": "object"},
        }]
        anthropic_tools_out = provider._convert_tools(anthropic_tools_in)
        assert anthropic_tools_out == anthropic_tools_in


# Run tests with: pytest tests/test_llm.py -v
