"""Agent state machine — manages valid state transitions."""
from __future__ import annotations

from datetime import datetime
from typing import Callable

from src.pipeline.types import AgentState
from src.logger import setup_logger

logger = setup_logger(__name__)


class StateTransition:
    """Describes a single valid state transition."""

    def __init__(
        self,
        from_state: AgentState,
        to_state: AgentState,
        event: str,
        condition: Callable[[], bool] | None = None,
        action: Callable[[], None] | None = None,
    ):
        self.from_state = from_state
        self.to_state = to_state
        self.event = event
        self.condition = condition
        self.action = action


class AgentStateMachine:
    """Manages the agent's state machine and valid transitions.

    States: IDLE → LISTENING → TRANSCRIBING → THINKING → CALLING_TOOL →
            GENERATING → SPEAKING → LISTENING
    Special: INTERRUPTED (user barge-in), ERROR (recovery state)
    """

    def __init__(self):
        self.current_state = AgentState.IDLE
        self.state_history: list[tuple[datetime, AgentState, str]] = []
        self.transitions: list[StateTransition] = []
        self._register_default_transitions()

    def _register_default_transitions(self) -> None:
        """Register all valid transitions from AgentState."""
        valid = {
            AgentState.IDLE: [AgentState.LISTENING],
            AgentState.LISTENING: [AgentState.TRANSCRIBING, AgentState.IDLE],
            AgentState.TRANSCRIBING: [AgentState.THINKING, AgentState.ERROR],
            AgentState.THINKING: [
                AgentState.CALLING_TOOL,
                AgentState.GENERATING,
                AgentState.ERROR,
            ],
            AgentState.CALLING_TOOL: [AgentState.GENERATING, AgentState.ERROR],
            AgentState.GENERATING: [
                AgentState.SPEAKING,
                AgentState.INTERRUPTED,
                AgentState.LISTENING,
                AgentState.ERROR,
            ],
            AgentState.SPEAKING: [
                AgentState.LISTENING,
                AgentState.INTERRUPTED,
                AgentState.IDLE,
            ],
            AgentState.INTERRUPTED: [AgentState.LISTENING, AgentState.TRANSCRIBING],
            AgentState.ERROR: [AgentState.IDLE, AgentState.LISTENING],
        }

        for from_state, to_states in valid.items():
            for to_state in to_states:
                self.transitions.append(
                    StateTransition(
                        from_state=from_state,
                        to_state=to_state,
                        event=f"{from_state.value}_to_{to_state.value}",
                    )
                )

    def transition(self, new_state: AgentState, reason: str = "") -> bool:
        """Attempt to transition to a new state.

        Args:
            new_state: The target state.
            reason: Human-readable reason for the transition.

        Returns:
            True if the transition was valid and applied, False otherwise.
        """
        if self._is_valid_transition(self.current_state, new_state):
            old_state = self.current_state
            self.current_state = new_state
            self.state_history.append((datetime.now(), old_state, reason))
            logger.info(
                f"[State] {old_state.value} → {new_state.value}" +
                (f" ({reason})" if reason else "")
            )
            return True

        logger.warning(
            f"[State] Invalid transition: "
            f"{self.current_state.value} → {new_state.value}"
        )
        return False

    def _is_valid_transition(
        self,
        from_state: AgentState,
        to_state: AgentState,
    ) -> bool:
        """Check if a transition is registered and passes its condition."""
        for t in self.transitions:
            if t.from_state == from_state and t.to_state == to_state:
                if t.condition is None or t.condition():
                    return True
        return False

    def can_interrupt(self) -> bool:
        """Return True if the current state is interruptible.

        States that can be interrupted: SPEAKING, GENERATING, CALLING_TOOL.
        """
        return self.current_state in (
            AgentState.SPEAKING,
            AgentState.GENERATING,
            AgentState.CALLING_TOOL,
        )

    def force_interrupt(self) -> bool:
        """Force the state machine to INTERRUPTED then LISTENING.

        Returns:
            True if interrupted successfully.
        """
        if not self.can_interrupt():
            return False

        self.transition(AgentState.INTERRUPTED, "user_barge_in")
        self.transition(AgentState.LISTENING, "continue_after_interrupt")
        return True

    def reset(self) -> None:
        """Reset to IDLE state."""
        self.current_state = AgentState.IDLE
        logger.info("[State] Reset to IDLE")

    @property
    def state(self) -> AgentState:
        """Current state."""
        return self.current_state

    def get_history(self) -> list[tuple[str, str, str]]:
        """Return state history as [(timestamp, from, reason), ...]."""
        return [
            (ts.isoformat(), from_state.value, reason)
            for ts, from_state, reason in self.state_history
        ]
