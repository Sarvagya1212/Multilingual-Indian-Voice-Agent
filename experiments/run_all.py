#!/usr/bin/env python3
"""Run all experiments and print a comparison table.

Usage:
    python -m experiments.run_all
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
EXPERIMENTS = [
    "001_whisper_model_comparison",
    "002_tts_provider_comparison",
    "003_rag_chunk_size",
]


def _print_divider():
    print("=" * 70)


def _run_experiment(exp_id: str) -> dict:
    exp_dir = ROOT / exp_id
    run_script = exp_dir / "run.py"
    config_path = exp_dir / "config.json"
    results_path = exp_dir / "results.json"

    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())

    status = config.get("status", "unknown")
    result_summary = {}

    if results_path.exists():
        raw = json.loads(results_path.read_text())
        result_summary = raw.get("results", {})

    return {
        "id": exp_id,
        "status": status,
        "results": result_summary,
    }


async def main() -> None:
    print("\n")
    _print_divider()
    print("  EXPERIMENT TRACKING — Multilingual Indian Voice Agent")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    _print_divider()

    for exp_id in EXPERIMENTS:
        print(f"\n  [{exp_id}]")
        exp_dir = ROOT / exp_id

        if not exp_dir.exists():
            print("    [NOT FOUND]")
            continue

        # Load status from config
        config_path = exp_dir / "config.json"
        results_path = exp_dir / "results.json"

        if config_path.exists():
            config = json.loads(config_path.read_text())
            status = config.get("status", "unknown")
            hypothesis = config.get("hypothesis", "no hypothesis")
            print(f"    Status:   {status}")
            print(f"    Hypothesis: {hypothesis[:70]}...")

        if results_path.exists():
            raw = json.loads(results_path.read_text())
            results = raw.get("results", {})
            print(f"    Results:   {len(results)} configuration(s) tested")
        else:
            print("    Results:   (not yet run)")

        print("    notes.md:  [see file for analysis]")

    print("\n")
    _print_divider()
    print("  To run an experiment:")
    print("    python experiments/<id>/run.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
