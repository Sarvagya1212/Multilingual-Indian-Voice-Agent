"""Run every demo in sequence.

Usage:
    python demos/run_all.py           # all 8 demos
    python demos/run_all.py 1 3 5     # only demos 1, 3, 5
    python demos/run_all.py --stop    # stop on first failure
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

DEMO_MODULES = [
    "demos.demo_01_english",
    "demos.demo_02_hindi",
    "demos.demo_03_hinglish",
    "demos.demo_04_tool_calling",
    "demos.demo_05_rag",
    "demos.demo_06_interruption",
    "demos.demo_07_failure_recovery",
    "demos.demo_08_dashboard",
]


def banner(title: str) -> None:
    bar = "#" * 72
    print(bar)
    print(f"  {title}")
    print(bar)


def run_demo(module_name: str) -> bool:
    """Run a single demo module's main() coroutine. Return True on success."""
    mod = importlib.import_module(module_name)
    print()
    start = time.perf_counter()
    try:
        # Each demo module exposes a `run(coro)` helper that calls asyncio.run.
        # If only `main` is defined, fall back to asyncio.run directly.
        if hasattr(mod, "run"):
            mod.run(mod.main)
        else:
            import asyncio

            asyncio.run(mod.main())
        elapsed = (time.perf_counter() - start) * 1000
        print(f"\n  [PASS] {module_name} ({elapsed:.0f} ms)")
        return True
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - start) * 1000
        print(f"\n  [FAIL] {module_name} after {elapsed:.0f} ms: {type(exc).__name__}: {exc}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run voice-agent demos")
    parser.add_argument(
        "indices",
        nargs="*",
        type=int,
        help="Optional 1-based demo numbers to run (default: all)",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop on first failure",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.indices:
        selected = [DEMO_MODULES[i - 1] for i in args.indices if 1 <= i <= len(DEMO_MODULES)]
    else:
        selected = DEMO_MODULES

    banner(f"Running {len(selected)} of {len(DEMO_MODULES)} demos")
    start = time.perf_counter()
    passed = 0
    failed: List[str] = []
    for module_name in selected:
        if run_demo(module_name):
            passed += 1
        else:
            failed.append(module_name)
            if args.stop:
                break
    total = (time.perf_counter() - start) * 1000

    banner(f"Summary: {passed}/{len(selected)} passed in {total:.0f} ms")
    if failed:
        for f in failed:
            print(f"  [FAIL] {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
