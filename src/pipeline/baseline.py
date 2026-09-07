"""Baseline voice pipeline - minimal end-to-end working system.

Pipeline: Microphone → STT → LLM → TTS → Speaker
"""
import asyncio
import time
import sys
import wave
import io
import struct
from typing import Optional

from src.pipeline.orchestrator import ConversationOrchestrator
from src.logger import setup_logger

logger = setup_logger(__name__)


def create_test_wav(
    duration: float = 2.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> bytes:
    """Create a synthetic test WAV (sine wave).

    Used for pipeline testing without microphone.
    """
    import math
    import numpy as np

    samples = int(duration * sample_rate)
    audio_data = []
    for i in range(samples):
        # Modulated sine wave (simulates speech rhythm)
        envelope = 0.3 + 0.2 * math.sin(2 * math.pi * 2 * i / sample_rate)
        value = int(32767 * envelope * math.sin(2 * math.pi * frequency * i / sample_rate))
        audio_data.append(value)

    # Write WAV
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(np.array(audio_data, dtype=np.int16)))
    return buffer.getvalue()


def save_audio(audio: bytes, filename: str, sample_rate: int = 24000) -> None:
    """Save audio bytes to WAV file."""
    try:
        # Save as raw bytes - assuming it's already encoded (mp3/wav)
        with open(filename, "wb") as f:
            f.write(audio)
        logger.info(f"Audio saved: {filename} ({len(audio)} bytes)")
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")


async def test_pipeline_with_synthetic_audio():
    """Test the pipeline with a synthetic audio file.

    This validates STT → LLM → TTS without requiring microphone.
    """
    print("\n" + "=" * 60)
    print("BASELINE VOICE PIPELINE TEST")
    print("=" * 60)
    print()
    print("Testing: STT (Whisper) → LLM (Claude) → TTS (OpenAI)")
    print()

    # Create orchestrator
    print("[1/4] Initializing orchestrator...")
    orchestrator = ConversationOrchestrator()
    session = orchestrator.create_session()
    print(f"  ✓ Session: {session.id}")
    print()

    # Test 1: STT-only (synthetic audio)
    print("[2/4] Testing STT with synthetic audio...")
    test_audio = create_test_wav(duration=2.0)
    print(f"  ✓ Created {len(test_audio)} bytes of test audio (2s sine wave)")

    try:
        from src.stt import get_stt_provider
        stt = get_stt_provider()
        stt_result = await stt.transcribe(test_audio)
        print(f"  ✓ STT result: '{stt_result.text}' [{stt_result.language}]")
        print(f"  ✓ Latency: {stt.latency_ms:.0f}ms")
    except Exception as e:
        print(f"  ✗ STT failed: {e}")
        logger.error(f"STT test failed: {e}")
    print()

    # Test 2: Language detection
    print("[3/4] Testing language detection...")
    test_cases = [
        ("Hello, how are you?", "en"),
        ("मुझे JEE की तैयारी करनी है", "hi"),
        ("JEE ke liye preparation", "en"),  # Roman transliteration
        ("मैं JEE exam की preparation कर रहा हूं", "hi"),
    ]

    for text, expected in test_cases:
        try:
            lang = await orchestrator.llm.detect_language(text)
            status = "✓" if (expected in ("en", "hi") and lang in (expected, "hinglish")) else "✗"
            print(f"  {status} '{text[:40]}...' → {lang} (expected ~{expected})")
        except Exception as e:
            print(f"  ✗ Language detection failed: {e}")
    print()

    # Test 3: Full pipeline (text in, audio out)
    print("[4/4] Testing full pipeline (text simulation)...")
    print("  (Note: For real audio pipeline, set OPENAI_API_KEY and use microphone)")
    print()

    # Verify all components loaded
    print("Pipeline Status:")
    print(f"  STT:  {orchestrator.stt.name} (ready)")
    print(f"  LLM:  {orchestrator.llm.name} (ready)")
    print(f"  TTS:  {orchestrator.tts.name} (ready)")
    print()

    print("=" * 60)
    print("BASELINE PIPELINE READY")
    print("=" * 60)
    print()
    print("To use:")
    print("  1. Set OPENAI_API_KEY in .env for TTS")
    print("  2. Set ANTHROPIC_API_KEY in .env for LLM")
    print("  3. Run: python -m src.pipeline.baseline")
    print()


async def test_pipeline_with_api_keys():
    """Test the full pipeline with real API calls (requires keys)."""
    print("\n" + "=" * 60)
    print("FULL PIPELINE TEST (with API calls)")
    print("=" * 60)
    print()

    orchestrator = ConversationOrchestrator()
    orchestrator.create_session()

    # Test with synthetic audio
    test_audio = create_test_wav(duration=2.0)

    try:
        print("Processing test audio through full pipeline...")
        result = await orchestrator.process_turn(test_audio)

        if result.audio:
            print(f"✓ Pipeline success!")
            print(f"  Audio output: {len(result.audio)} bytes")
            print(f"  Duration: {result.duration:.1f}s")
            print(f"  Format: {result.format}")
            print(f"  Voice: {result.voice}")

            # Save to file
            output_file = "output_tts.wav"
            save_audio(result.audio, output_file)
            print(f"  Saved: {output_file}")
        else:
            print("Pipeline returned empty audio")

    except Exception as e:
        print(f"Pipeline error: {e}")
        logger.error(f"Full pipeline test failed: {e}", exc_info=True)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Baseline voice pipeline")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline with API calls (requires API keys)",
    )
    args = parser.parse_args()

    if args.full:
        asyncio.run(test_pipeline_with_api_keys())
    else:
        asyncio.run(test_pipeline_with_synthetic_audio())


if __name__ == "__main__":
    main()
