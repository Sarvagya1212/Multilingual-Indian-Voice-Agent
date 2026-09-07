"""Short-term memory — recent conversation context within a session."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List

from src.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ConversationTurn:
    """A single user ↔ agent exchange."""

    user_message: str
    agent_message: str
    language: str
    timestamp: datetime
    tools_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ShortTermMemory:
    """Sliding-window conversation context for the current session.

    - Stores the last N turns
    - Tracks user facts (class, interests, language preference)
    - Auto-expires after TTL to avoid stale context
    """

    def __init__(self, max_turns: int = 10, ttl_minutes: int = 30):
        self.max_turns = max_turns
        self.ttl_minutes = ttl_minutes
        self.turns: List[ConversationTurn] = []
        self.session_start = datetime.now()
        self.user_context: Dict[str, Any] = {}

    def add_turn(
        self,
        user_message: str,
        agent_message: str,
        language: str,
        tools_used: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Add a conversation turn and update derived user context.

        Args:
            user_message: The user's message text.
            agent_message: The agent's response text.
            language: Detected language (en/hi/hinglish).
            tools_used: Names of tools the agent invoked.
            metadata: Additional metadata.
        """
        turn = ConversationTurn(
            user_message=user_message,
            agent_message=agent_message,
            language=language,
            timestamp=datetime.now(),
            tools_used=tools_used or [],
            metadata=metadata or {},
        )
        self.turns.append(turn)

        # Trim to max_turns (keep most recent)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

        self._update_context(user_message, language)

    def _update_context(self, message: str, language: str) -> None:
        """Extract simple facts from the user message.

        Args:
            message: User message.
            language: Detected language.
        """
        self.user_context["preferred_language"] = language
        message_lower = message.lower()

        if "class" in message_lower:
            for cls in ("10", "11", "12"):
                if cls in message_lower:
                    self.user_context["current_class"] = f"Class {cls}"
                    break

        if any(kw in message_lower for kw in ("jee", "engineering", "iit", "btech", "b.tech")):
            self.user_context["interest"] = "engineering"
        elif any(kw in message_lower for kw in ("neet", "medical", "doctor", "mbbs")):
            self.user_context["interest"] = "medical"
        elif any(kw in message_lower for kw in ("cbse", "board", "icse", "state board")):
            self.user_context["interest"] = "boards"

    def get_recent_context(self, num_turns: int = 3) -> str:
        """Get recent conversation as a string.

        Args:
            num_turns: Number of most recent turns to include.

        Returns:
            Formatted conversation history string.
        """
        recent = self.turns[-num_turns:] if self.turns else []
        if not recent:
            return "No previous context."

        parts: List[str] = []
        for turn in recent:
            parts.append(f"User: {turn.user_message}")
            parts.append(f"Agent: {turn.agent_message}")
        return "\n".join(parts)

    def is_expired(self) -> bool:
        """Return True if the session has exceeded its TTL."""
        return datetime.now() - self.session_start > timedelta(minutes=self.ttl_minutes)

    def clear(self) -> None:
        """Reset memory and user context."""
        self.turns = []
        self.user_context = {}
        self.session_start = datetime.now()

    def get_context_summary(self) -> Dict[str, Any]:
        """Return a summary of stored context."""
        return {
            "turns_count": len(self.turns),
            "session_duration_minutes": (
                datetime.now() - self.session_start
            ).total_seconds() / 60,
            "user_context": dict(self.user_context),
            "languages_used": list({t.language for t in self.turns}),
        }

    def __len__(self) -> int:
        return len(self.turns)
