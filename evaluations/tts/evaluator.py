"""TTS evaluator — runs synthesis against text cases and reports metrics."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from src.logger import setup_logger
from evaluations.tts.metrics import TTSEvaluator

logger = setup_logger(__name__)


@dataclass
class TTSTestCase:
    """A single test case for TTS evaluation."""

    text: str
    language: str = "en"
    expected_duration_min: float = 0.5  # Sanity check
    expected_duration_max: float = 60.0


class TTSEvaluatorRunner:
    """Runs TTS evaluation against text cases."""

    def __init__(
        self,
        tts_provider: Any,  # TTSProvider instance
        evaluator: TTSEvaluator | None = None,
    ):
        self.tts = tts_provider
        self.evaluator = evaluator or TTSEvaluator()

    async def evaluate_case(self, case: TTSTestCase) -> dict[str, Any]:
        """Evaluate a single TTS case."""
        start = time.time()
        try:
            result = await self.tts.synthesize(
                case.text,
                language=case.language,
            )
            latency_ms = (time.time() - start) * 1000
        except Exception as exc:
            logger.error(f"[TTS Eval] Synthesis failed: {exc}")
            return {
                "case": case,
                "success": False,
                "error": str(exc),
            }

        # Sanity check duration
        duration_ok = case.expected_duration_min <= result.duration <= case.expected_duration_max

        metrics = self.evaluator.evaluate(
            text=case.text,
            audio_bytes=result.audio,
            audio_duration=result.duration,
            latency_ms=latency_ms,
        )

        return {
            "case": case,
            "success": True,
            "audio_duration": result.duration,
            "latency_ms": latency_ms,
            "duration_sane": duration_ok,
            "metrics": metrics,
        }

    async def run(self, cases: list[TTSTestCase]) -> dict[str, Any]:
        """Run evaluation against all cases."""
        results = []
        for case in cases:
            result = await self.evaluate_case(case)
            results.append(result)
        return {
            "summary": self.evaluator.get_summary(),
            "cases": results,
        }
