#!/usr/bin/env python3
"""Run Experiment 002: TTS Provider Comparison.

Tests: OpenAI tts-1 vs tts-1-hd on Indian-language test sentences.

Usage:
    python experiments/002_tts_provider_comparison/run.py
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluations.tts.metrics import TTSEvaluator


_TEST_SENTENCES = [
    "Hello, I want to join JEE classes. What are the fees?",
    "The total fee for NEET coaching is Rs 1,80,000.",
    "IIT Delhi is one of the top engineering colleges in India.",
    "Call 9876543210 to schedule your demo class today.",
    "We offer scholarships up to 50% for meritorious students.",
    "New batches start in June, September, and January.",
    "नमस्ते, मुझे NEET कोचिंग चाहिए।",
    "JEE Main exam ke liye preparation course available hai.",
    "CBSE board exam ke liye special batch hai.",
    "Hostel facility boys and girls ke liye available hai.",
]


async def evaluate_provider(
    provider_name: str,
    model: str,
    voice: str,
    sentences: list[str],
) -> dict[str, Any]:
    """Evaluate a TTS provider on the test sentences."""
    import openai

    client = openai.OpenAI()
    ev = TTSEvaluator()

    for text in sentences:
        start = time.time()
        try:
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
            )
            audio_bytes = response.content
            latency_ms = (time.time() - start) * 1000
            # Approximate duration from character count
            duration = len(text) / 150 * 60
        except Exception as exc:
            print(f"\n  [Error] {provider_name}: {exc}")
            continue

        ev.evaluate(
            text=text,
            audio_bytes=audio_bytes,
            audio_duration=duration,
            latency_ms=latency_ms,
        )

    return ev.get_summary()


async def main() -> None:
    experiment_dir = Path(__file__).parent
    results_path = experiment_dir / "results.json"
    config = json.loads((experiment_dir / "config.json").read_text())

    print("=" * 60)
    print("  Experiment 002: TTS Provider Comparison")
    print("=" * 60)

    providers = config["providers"]
    all_results = {}
    for prov in providers:
        name = prov["name"]
        print(f"\n  Evaluating provider: {name} ...", end="", flush=True)
        summary = await evaluate_provider(
            provider_name=name,
            model=prov["model"],
            voice=prov["voice"],
            sentences=_TEST_SENTENCES,
        )
        all_results[name] = summary
        avg_lat = summary.get("avg_latency_ms", 0)
        print(f" avg_latency={avg_lat:.0f}ms")

    results = {
        "experiment_id": "002_tts_provider_comparison",
        "config": config,
        "results": all_results,
    }
    results_path.write_text(json.dumps(results, indent=2))

    print(f"\n  Results saved to {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
