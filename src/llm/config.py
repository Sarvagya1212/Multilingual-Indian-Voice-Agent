"""LLM configuration."""
from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""

    # Provider selection
    provider: str = "ollama"  # ollama (free), anthropic (paid)

    # Model selection
    model: str = "llama3.1:latest"  # free via Ollama; change to "claude-sonnet-4-20250514" for Anthropic
    max_tokens: int = 1024

    # Generation parameters
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: List[str] = []

    # System prompt
    system_prompt_version: str = "v1"

    # Tool use
    enable_tools: bool = True
    tool_choice: str = "auto"  # auto, required, none

    class Config:
        extra = "ignore"


# Prompt file paths
PROMPT_FILES = {
    "system_v1": "prompts/system_v1.txt",
    "tool_use_v1": "prompts/tool_use_v1.txt",
    "multilingual_v1": "prompts/multilingual_v1.txt",
}


def load_prompt(prompt_name: str) -> str:
    """Load a prompt from file.

    Args:
        prompt_name: Name of the prompt (e.g., 'system_v1')

    Returns:
        Prompt text content
    """
    import os

    filepath = PROMPT_FILES.get(prompt_name)
    if not filepath:
        raise ValueError(f"Unknown prompt: {prompt_name}")

    # Try relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    full_path = os.path.join(base_dir, filepath)

    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    return ""


def build_system_prompt(
    system_prompt: str,
    tool_prompt: str = "",
    multilingual_prompt: str = "",
) -> str:
    """Build combined system prompt from components.

    Args:
        system_prompt: Base system prompt
        tool_prompt: Tool use instructions
        multilingual_prompt: Multilingual instructions

    Returns:
        Combined system prompt
    """
    parts = [system_prompt.strip()]

    if tool_prompt:
        parts.append("\n\n" + tool_prompt.strip())

    if multilingual_prompt:
        parts.append("\n\n" + multilingual_prompt.strip())

    return "\n\n".join(parts)
