"""Agent configuration."""
from __future__ import annotations

from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Configuration for the full agent orchestrator."""

    max_turns: int = 10  # Short-term memory window
    memory_ttl_minutes: int = 30  # Session TTL
    tool_timeout_seconds: float = 30.0
    max_retries: int = 1

    class Config:
        extra = "ignore"
