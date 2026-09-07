"""LLM (Large Language Model) module.

Provides a provider-agnostic abstraction for language model inference.

Example:
    from src.llm import get_llm_provider, Message, MessageRole

    # Get default provider
    llm = get_llm_provider()

    # Simple chat
    messages = [
        Message(role=MessageRole.USER, content="Hello!")
    ]
    response = await llm.chat(messages)
    print(response.content)

    # With tool use
    tools = [{
        "type": "function",
        "function": {
            "name": "search_courses",
            "description": "Search for courses",
            "parameters": {"type": "object", "properties": {}}
        }
    }]
    response = await llm.chat(messages, tools=tools)
    if response.tool_calls:
        print(f"Tool called: {response.tool_calls[0].name}")

    # Detect language
    lang = await llm.detect_language("JEE ke liye preparation")
    print(f"Language: {lang}")
"""

from src.llm.base import (
    LLMProvider,
    Message,
    MessageRole,
    LLMResponse,
    ToolCall,
)
from src.llm.providers import (
    AnthropicLLMProvider,
    OllamaLLMProvider,
    get_llm_provider,
    LLM_PROVIDERS,
)
from src.llm.config import (
    LLMConfig,
    load_prompt,
    build_system_prompt,
    PROMPT_FILES,
)

__all__ = [
    # Core classes
    "LLMProvider",
    "Message",
    "MessageRole",
    "LLMResponse",
    "ToolCall",
    # Implementations
    "AnthropicLLMProvider",
    "OllamaLLMProvider",
    "get_llm_provider",
    "LLM_PROVIDERS",
    # Config
    "LLMConfig",
    "load_prompt",
    "build_system_prompt",
    "PROMPT_FILES",
]
