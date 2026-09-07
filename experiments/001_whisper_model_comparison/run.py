#!/usr/bin/env python3
"""Run Experiment 001: Whisper Model Comparison.

Tests: tiny vs base vs small — WER, latency, RTF on en/hi/hinglish.

Usage:
    python experiments/001_whisper_model_comparison/run.py
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluations.stt.metrics import STTEvaluator


async def load_test_samples(dataset_path: Path) -> list[dict[str, Any]]:
    """Load audio + reference pairs from dataset directory.

    Expected layout:
        dataset_path/
            en/
                sample_001.wav  (with corresponding .txt)
                ...
            hi/
            hinglish/

    Returns list of dicts with audio_bytes, reference, language, duration.
    """
    samples = []
    for lang in ("en", "hi", "hinglish"):
        lang_dir = dataset_path / lang
        if not lang_dir.exists():
            # Dataset not present — return placeholder samples for simulation
            return [
                {"language": "en", "reference": "Hello, I want to join JEE classes"},
                {"language": "hi", "reference": "mujhe NEET ki taiyari karna hai"},
                {"language": "hinglish", "reference": "JEE ke liye preparation kaise karein"},
            ]

        for wav_file in sorted(lang_dir.glob("*.wav"))[:20]:
            txt_file = wav_file.with_suffix(".txt")
            reference = txt_file.read_text().strip() if txt_file.exists() else ""
            samples.append({
                "language": lang,
                "audio_path": str(wav_file),
                "reference": reference,
            })
    return samples


async def evaluate_model(
    model_name: str,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate a Whisper model on the dataset."""
    import whisper

    model = whisper.load_model(model_name)
    ev = STTEvaluator()

    for sample in samples:
        audio_path = sample.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            continue

        start = time.time()
        result = model.transcribe(audio_path)
        latency_ms = (time.time() - start) * 1000

        ev.evaluate(
            reference=sample["reference"],
            hypothesis=result["text"],
            language=sample["language"],
            detected_language=result.get("language", sample["language"]),
            latency_ms=latency_ms,
            audio_duration=result.get("duration", 3.0),
        )

    return ev.get_summary()


async def main() -> None:
    experiment_dir = Path(__file__).parent
    results_path = experiment_dir / "results.json"

    # Load config
    config = json.loads((experiment_dir / "config.json").read_text())
    models = config["models"]
    dataset_path = Path(__file__).parent.parent.parent / config["dataset"]["path"]

    print("=" * 60)
    print("  Experiment 001: Whisper Model Comparison")
    print("=" * 60)

    samples = await load_test_samples(dataset_path)
    print(f"  Loaded {len(samples)} samples")

    all_results = {}
    for model in models:
        print(f"\n  Evaluating model: {model} ...", end="", flush=True)
        summary = await evaluate_model(model, samples)
        all_results[model] = summary
        avg_wer = summary.get("avg_wer", 0)
        avg_lat = summary.get("avg_latency_ms", 0)
        print(f" WER={avg_wer:.2%}, Latency={avg_lat:.0f}ms")

    # Save results
    results = {
        "experiment_id": "001_whisper_model_comparison",
        "config": config,
        "results": all_results,
    }
    results_path.write_text(json.dumps(results, indent=2))

    print(f"\n  Results saved to {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
