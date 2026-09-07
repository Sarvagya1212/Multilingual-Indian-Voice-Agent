"""Demo 6: Interruption / Barge-in.

The user interrupts the agent mid-sentence. The state machine should
halt the agent's current GENERATING / SPEAKING state and move to
INTERRUPTED, allowing the next turn to begin immediately.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List

from demos._common import (
    SimTurnResult,
    header,
    info,
    ok,
    run,
    step,
    warn,
    scripted_response,
)


@dataclass
class InterruptibleAgent:
    """Agent whose SPEAKING state can be interrupted.

    Uses cooperative cancellation: the SPEAKING phase awaits a sleep
    that we cancel externally when the user interrupts.
    """

    session_id: str = ""
    state: str = "IDLE"
    turns: int = 0
    history: List[SimTurnResult] = field(default_factory=list)
    _speaking_task: asyncio.Task | None = None

    def create_session(self) -> "InterruptibleAgent":
        self.session_id = f"demo-{int(time.time() * 1000)}"
        self.state = "IDLE"
        return self

    def interrupt(self) -> bool:
        if self.state in ("SPEAKING", "GENERATING") and self._speaking_task:
            self.state = "INTERRUPTED"
            self._speaking_task.cancel()
            return True
        return False

    async def process_turn(self, transcript: str, language: str = "en") -> SimTurnResult:
        self.turns += 1
        start = time.perf_counter()
        self.state = "TRANSCRIBING"
        await asyncio.sleep(0.01)
        self.state = "THINKING"
        await asyncio.sleep(0.02)
        self.state = "GENERATING"
        await asyncio.sleep(0.05)

        # Long SPEAKING phase — the cancelable point
        self.state = "SPEAKING"
        was_interrupted = False
        try:
            self._speaking_task = asyncio.current_task()
            await asyncio.sleep(0.5)  # cancellable
        except asyncio.CancelledError:
            was_interrupted = True
            self.state = "INTERRUPTED"
            raise
        finally:
            self._speaking_task = None
            if not was_interrupted and self.state == "SPEAKING":
                self.state = "IDLE"

        response = scripted_response(transcript, language)
        elapsed_ms = (time.perf_counter() - start) * 1000
        result = SimTurnResult(
            transcript=transcript,
            response_text=response,
            detected_language=language,
            audio=b"\x00" * 1024,
            duration=0.5,
            latency_ms=elapsed_ms,
            state=self.state,
            notes=["interrupted" if was_interrupted else "completed"],
        )
        self.history.append(result)
        return result


async def main() -> None:
    header("Demo 6: Interruption / Barge-in")
    info("The user interrupts the agent mid-response. State flips SPEAKING -> INTERRUPTED.")
    print()

    agent = InterruptibleAgent().create_session()
    info(f"Session: {agent.session_id}")
    print()

    step(1, 3, "Agent starts speaking")
    print("  Agent state: SPEAKING (will run for 500ms)")
    print()

    step(2, 3, "User interrupts at +50ms")
    print(f"  User: \"Wait, I have a different question.\"")
    turn_task = asyncio.create_task(
        agent.process_turn("Wait, I have a different question.", language="en")
    )
    # Wait until the agent enters SPEAKING, then interrupt
    while agent.state != "SPEAKING":
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.05)  # simulate user delay before interrupting
    interrupted = agent.interrupt()
    print(f"  [interrupt() returned: {interrupted}]")
    try:
        result = await turn_task
    except asyncio.CancelledError:
        result = SimTurnResult(
            transcript="Wait, I have a different question.",
            response_text="<cancelled before final response>",
            detected_language="en",
            audio=b"",
            duration=0,
            latency_ms=0,
            state=agent.state,
            notes=["interrupted"],
        )
    print(f"  Final state: {agent.state}")
    print(f"  Notes: {result.notes}")
    print()

    step(3, 3, "Verify state machine behaviour")
    if agent.state == "INTERRUPTED":
        ok("State machine correctly halted the agent mid-sentence.")
    else:
        warn(f"Expected INTERRUPTED, got {agent.state}.")
    print()

    ok("Interruption demo completed.")


if __name__ == "__main__":
    run(main)
