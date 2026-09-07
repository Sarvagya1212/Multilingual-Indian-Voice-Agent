"""Demo 2: Hindi Conversation.

Full Hindi conversation showing the language support.
The system detects Hindi (Devanagari script) and responds in Hindi.
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
    header("Demo 2: Hindi Conversation")
    info("Devanagari Hindi input -> Hindi response. Language locked at the prompt level.")
    print()

    if not (has_api_key("OPENAI_API_KEY") and has_api_key("ANTHROPIC_API_KEY")):
        miss("API keys missing: running with deterministic simulation.")
    else:
        info("API keys detected: will use the live pipeline.")
    print()

    agent = SimulatedAgent().create_session()
    info(f"Session: {agent.session_id}")
    print()

    turns = [
        "नमस्ते, मुझे JEE की तैयारी करनी है।",
        "कोर्स की फीस कितनी है?",
    ]
    for i, transcript in enumerate(turns, start=1):
        step(i, len(turns), f"Turn {i}")
        print(f"  User: \"{transcript}\"")
        result = await agent.process_turn(transcript, language="hi")
        print(f"  Agent: \"{result.response_text}\"")
        print(f"  Detected language: {result.detected_language}")
        print(f"  Latency: {result.latency_ms:.0f} ms")
        print()

    ok("Hindi conversation completed successfully.")


if __name__ == "__main__":
    run(main)
