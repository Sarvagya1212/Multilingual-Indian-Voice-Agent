"""TTS evaluation metrics — pronunciation, latency, audio quality proxies."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TTSMetrics:
    """Metrics for a single TTS evaluation."""

    text: str
    audio_duration: float  # Seconds
    text_length: int  # Character count
    latency_ms: float
    real_time_factor: float  # latency_ms / 1000 / audio_duration
    bytes_per_second: float
    pronunciation_corrections: int  # Acronyms normalized
    has_silence_pads: bool  # Did normalizer add padding?

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "audio_duration": self.audio_duration,
            "text_length": self.text_length,
            "latency_ms": self.latency_ms,
            "real_time_factor": self.real_time_factor,
            "bytes_per_second": self.bytes_per_second,
            "pronunciation_corrections": self.pronunciation_corrections,
        }


class TTSEvaluator:
    """Aggregates TTS evaluation results."""

    # Common Indian/edu acronyms that TTS gets wrong
    ACRONYM_PATTERNS = [
        r"\bJEE\b",
        r"\bNEET\b",
        r"\bIIT\b",
        r"\bCBSE\b",
        r"\bNDA\b",
        r"\bAIIMS\b",
    ]

    def __init__(self):
        self.results: list[TTSMetrics] = []

    def evaluate(
        self,
        text: str,
        audio_bytes: bytes,
        audio_duration: float,
        latency_ms: float,
        normalized_text: str | None = None,
    ) -> TTSMetrics:
        """Evaluate a single TTS synthesis.

        Args:
            text: Original input text.
            audio_bytes: Synthesized audio (raw).
            audio_duration: Duration in seconds (from TTS provider).
            latency_ms: Time-to-first-audio in ms.
            normalized_text: Text after normalization (for acronym detection).

        Returns:
            TTSMetrics for this sample.
        """
        # Count acronym normalizations
        corrections = 0
        for pat in self.ACRONYM_PATTERNS:
            corrections += len(re.findall(pat, text))

        rtf = latency_ms / 1000.0 / max(audio_duration, 1e-3)
        bps = len(audio_bytes) / max(audio_duration, 1e-3)

        metrics = TTSMetrics(
            text=text,
            audio_duration=audio_duration,
            text_length=len(text),
            latency_ms=latency_ms,
            real_time_factor=rtf,
            bytes_per_second=bps,
            pronunciation_corrections=corrections,
            has_silence_pads=bool(normalized_text and re.search(r"\b[A-Z]\b", normalized_text)),
        )
        self.results.append(metrics)
        return metrics

    def get_summary(self) -> dict:
        """Aggregate TTS metrics."""
        if not self.results:
            return {}
        n = len(self.results)
        return {
            "total_samples": n,
            "avg_audio_duration": sum(r.audio_duration for r in self.results) / n,
            "avg_latency_ms": sum(r.latency_ms for r in self.results) / n,
            "avg_real_time_factor": sum(r.real_time_factor for r in self.results) / n,
            "avg_bytes_per_second": sum(r.bytes_per_second for r in self.results) / n,
            "total_pronunciation_corrections": sum(r.pronunciation_corrections for r in self.results),
        }

    def clear(self) -> None:
        self.results = []
