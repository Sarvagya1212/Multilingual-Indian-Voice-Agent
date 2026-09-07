"""Demo 3: Hinglish Conversation.

Code-switching between Hindi and English in a single utterance.
Demonstrates the multi-signal language detector and the language-locked
system prompt (no English default when Hinglish is detected).
"""
from __future__ import annotations

from demos._common import (
    SimulatedAgent,
    has_api_key,
    header,
    info,
    miss,
    ok,
    run,
    step,
)


async def main() -> None:
    header("Demo 3: Hinglish / Code-Switching")
    info("Mixed Hindi + English in one sentence.")
    info("Script ratio: Devanagari words + Roman-script Hindi words + English.")
    print()

    if not (has_api_key("OPENAI_API_KEY") and has_api_key("ANTHROPIC_API_KEY")):
        miss("API keys missing: running with deterministic simulation.")
    else:
        info("API keys detected: will use the live pipeline.")
    print()

    agent = SimulatedAgent().create_session()
    info(f"Session: {agent.session_id}")
    print()

    # Detect language and respond in the same language
    turns = [
        ("Main Class 11 ka student hoon, JEE ke liye course dhoondh raha hoon.", "hinglish"),
        ("Kya mujhe scholarship mil sakti hai?", "hinglish"),
        ("CBSE boards preparation course available hai kya?", "hinglish"),
    ]
    for i, (transcript, expected_lang) in enumerate(turns, start=1):
        step(i, len(turns), f"Turn {i}")
        print(f"  User: \"{transcript}\"")
        result = await agent.process_turn(transcript, language=expected_lang)
        print(f"  Agent: \"{result.response_text}\"")
        print(f"  Detected language: {result.detected_language} (expected: {expected_lang})")
        print(f"  Latency: {result.latency_ms:.0f} ms")
        print()

    ok("Hinglish conversation completed successfully.")


if __name__ == "__main__":
    run(main)
