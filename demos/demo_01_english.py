"""Demo 1: English Conversation.

A basic English interaction with the education counsellor.
Demonstrates the STT -> LLM -> TTS pipeline on a single turn.
"""
from __future__ import annotations

from demos._common import (
    SimulatedAgent,
    create_synthetic_wav,
    has_api_key,
    header,
    info,
    miss,
    ok,
    run,
    step,
)


async def main() -> None:
    header("Demo 1: English Conversation")
    info("A natural English turn with the education counsellor.")
    print()

    audio = create_synthetic_wav(duration=1.0, frequency=440.0)
    info(f"Generated {len(audio)} bytes of synthetic audio (1.0s sine wave).")
    print()

    has_openai = has_api_key("OPENAI_API_KEY")
    has_anthropic = has_api_key("ANTHROPIC_API_KEY")
    if has_openai and has_anthropic:
        info("API keys detected: will use the live pipeline.")
    else:
        miss("API keys missing: running with deterministic simulation.")
        if not has_openai:
            miss("Set OPENAI_API_KEY for live OpenAI TTS.")
        if not has_anthropic:
            miss("Set ANTHROPIC_API_KEY for live Claude LLM.")
    print()

    agent = SimulatedAgent().create_session()
    info(f"Session: {agent.session_id}")
    print()

    # Single turn
    step(1, 2, "User speaks")
    transcript = "Hello, what courses do you offer?"
    print(f"  User: \"{transcript}\"")
    result = await agent.process_turn(transcript, language="en")
    print()

    step(2, 2, "Agent responds")
    print(f"  Agent: \"{result.response_text}\"")
    print(f"  Detected language: {result.detected_language}")
    print(f"  State: {result.state}")
    print(f"  Latency: {result.latency_ms:.0f} ms")
    print(f"  Audio generated: {len(result.audio)} bytes ({result.duration:.2f}s)")
    print()

    ok("English conversation completed successfully.")


if __name__ == "__main__":
    run(main)
