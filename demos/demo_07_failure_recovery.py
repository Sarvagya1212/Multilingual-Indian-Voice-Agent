"""Demo 7: Failure Recovery.

The agent encounters errors (STT failure, LLM timeout, tool error) and
handles them gracefully. The state machine transitions to ERROR and
recovers back to IDLE rather than crashing.
"""
from __future__ import annotations

from demos._common import (
    header,
    info,
    ok,
    run,
    step,
    warn,
)


# ----------------------------------------------------------------------
# Error simulation
# ----------------------------------------------------------------------


class STTError(RuntimeError):
    pass


class LLMTimeout(RuntimeError):
    pass


class ToolError(RuntimeError):
    pass


# ----------------------------------------------------------------------
# Simulated agent that can inject errors
# ----------------------------------------------------------------------


class RecoveryAgent:
    """Agent that simulates errors and recovers cleanly."""

    def __init__(self):
        self.state = "IDLE"
        self.session_id = "recovery-demo"
        self.errors_caught: list[tuple[str, Exception]] = []

    def _transition(self, target: str) -> None:
        print(f"    State: IDLE -> {target}")
        self.state = target

    def _recover(self, error: Exception) -> None:
        print(f"    [ERROR CAUGHT] {type(error).__name__}: {error}")
        self.errors_caught.append((self.state, error))
        self._transition("ERROR")
        print("    Recovery: ERROR -> IDLE (ready for next turn)")
        self.state = "IDLE"

    async def turn(self, name: str, error_factory) -> None:
        self._transition("TRANSCRIBING")
        self._transition("THINKING")
        self._transition("GENERATING")
        try:
            raise error_factory()
        except Exception as exc:  # noqa: PERF203
            self._recover(exc)


async def main() -> None:
    header("Demo 7: Failure Recovery")
    info("Three error scenarios — agent transitions to ERROR and recovers to IDLE.")
    print()

    agent = RecoveryAgent()
    scenarios = [
        ("STT failure", lambda: STTError("Audio too short: 0 bytes")),
        ("LLM timeout", lambda: LLMTimeout("Claude response timed out after 30s")),
        ("Tool error", lambda: ToolError("check_eligibility: invalid percentage 'NaN'")),
    ]
    for i, (label, error_factory) in enumerate(scenarios, start=1):
        step(i, len(scenarios), f"Scenario: {label}")
        print(f"  Simulating: {label}")
        await agent.turn(label, error_factory)
        print()

    step(len(scenarios) + 1, len(scenarios) + 1, "Error log")
    print(f"  Total errors caught: {len(agent.errors_caught)}")
    for i, (state, exc) in enumerate(agent.errors_caught, 1):
        print(f"  {i}. {type(exc).__name__} while in {state}")
    print()

    ok("All three error scenarios recovered cleanly to IDLE.")
    ok("Failure recovery demo completed successfully.")


if __name__ == "__main__":
    run(main)
