"""STT evaluator — runs transcription against a dataset and reports metrics."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from src.logger import setup_logger
from evaluations.stt.metrics import STTEvaluator

logger = setup_logger(__name__)


@dataclass
class STTTestCase:
    """A single test case for STT evaluation."""

    audio: bytes  # Raw audio bytes
    reference: str  # Ground truth transcription
    language: str  # Expected language code (en, hi, hinglish, etc.)
    audio_duration: float  # Duration in seconds
    metadata: dict[str, Any] | None = None


class STTEvaluatorRunner:
    """Runs STT evaluation against a list of test cases.

    Usage:
        runner = STTEvaluatorRunner(stt_provider)
        cases = [STTTestCase(audio=b"...", reference="hello", language="en", duration=2.0)]
        summary = await runner.run(cases)
    """

    def __init__(
        self,
        stt_provider: Any,  # STTProvider instance
        evaluator: STTEvaluator | None = None,
    ):
        self.stt = stt_provider
        self.evaluator = evaluator or STTEvaluator()

    async def evaluate_case(self, case: STTTestCase) -> dict[str, Any]:
        """Evaluate a single test case."""
        start = time.time()
        try:
            result = await self.stt.transcribe(case.audio, language=case.language)
            latency_ms = (time.time() - start) * 1000
        except Exception as exc:
            logger.error(f"[STT Eval] Transcription failed: {exc}")
            return {
                "case": case,
                "success": False,
                "error": str(exc),
            }

        metrics = self.evaluator.evaluate(
            reference=case.reference,
            hypothesis=result.text,
            language=case.language,
            detected_language=result.language,
            latency_ms=latency_ms,
            audio_duration=case.audio_duration,
        )

        return {
            "case": case,
            "success": True,
            "hypothesis": result.text,
            "metrics": metrics,
        }

    async def run(
        self,
        cases: list[STTTestCase],
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Run evaluation against all cases.

        Args:
            cases: List of test cases.
            progress_callback: Optional async fn called after each case.

        Returns:
            Summary dict with aggregate + per-case results.
        """
        results = []
        for i, case in enumerate(cases):
            result = await self.evaluate_case(case)
            results.append(result)
            if progress_callback:
                await progress_callback(i + 1, len(cases), result)

        return {
            "summary": self.evaluator.get_summary(),
            "per_language": self.evaluator.get_per_language_summary(),
            "cases": results,
        }
