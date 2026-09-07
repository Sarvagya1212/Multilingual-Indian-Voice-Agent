"""Long-term memory — persistent user preferences across sessions.

Currently a stub; in a production deployment this would back onto a database
or persistent key-value store. The interface is designed to be drop-in
compatible with ShortTermMemory so the orchestrator can be swapped.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from src.logger import setup_logger

logger = setup_logger(__name__)


class LongTermMemory:
    """Persistent user profile and preferences.

    The actual storage layer is intentionally absent — this is a
    protocol-style stub that downstream deployments can implement against
    Redis, SQLite, Postgres, etc.
    """

    def __init__(self, storage_backend: str = "memory"):
        self.storage_backend = storage_backend
        self.profiles: Dict[str, Dict[str, Any]] = {}
        logger.info(
            f"LongTermMemory initialised (backend={storage_backend})"
        )

    def load_profile(self, user_id: str) -> Dict[str, Any]:
        """Load a user profile by ID.

        Args:
            user_id: Stable user identifier.

        Returns:
            Profile dict (empty if not found).
        """
        return self.profiles.get(user_id, {})

    def save_profile(self, user_id: str, profile: Dict[str, Any]) -> None:
        """Persist a user profile.

        Args:
            user_id: Stable user identifier.
            profile: Profile fields.
        """
        profile["_updated_at"] = datetime.now().isoformat()
        self.profiles[user_id] = profile

    def list_users(self) -> List[str]:
        """List all known user IDs."""
        return list(self.profiles.keys())
