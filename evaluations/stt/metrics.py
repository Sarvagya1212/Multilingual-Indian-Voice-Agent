"""STT evaluation metrics — WER, CER, language accuracy, latency."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Compute Levenshtein distance between two sequences (no external dep)."""
    if not a:
        return len(b)
    if not b:
        return len(a)

    # 1D rolling distance
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, start=1):
            cost = 0 if x == y else 1
            cur[j] = min(
                cur[j - 1] + 1,        # insertion
                prev[j] + 1,            # deletion
                prev[j - 1] + cost,     # substitution
            )
        prev = cur
    return prev[-1]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate (0 = perfect, 1 = all wrong)."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = _levenshtein(ref_words, hyp_words)
    return min(dist / len(ref_words), 1.0)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate."""
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    dist = _levenshtein(ref_chars, hyp_chars)
    return min(dist / len(ref_chars), 1.0)


@dataclass
class STTMetrics:
    """Metrics for a single STT evaluation."""

    wer: float
    cer: float
    accuracy: float  # (1 - WER) * 100
    language_correct: bool
    latency_ms: float
    real_time_factor: float  # latency_ms / 1000 / audio_duration
    reference: str
    hypothesis: str
    language: str
    detected_language: str


class STTEvaluator:
    """Aggregates STT evaluation results."""

    def __init__(self):
        self.results: list[STTMetrics] = []

    def evaluate(
        self,
        reference: str,
        hypothesis: str,
        language: str,
        detected_language: str,
        latency_ms: float,
        audio_duration: float,
    ) -> STTMetrics:
        """Evaluate a single STT result.

        Args:
            reference: Ground truth text.
            hypothesis: STT-transcribed text.
            language: Expected language code.
            detected_language: STT-detected language code.
            latency_ms: STT latency in ms.
            audio_duration: Audio length in seconds.

        Returns:
            STTMetrics for this sample.
        """
        wer = compute_wer(reference, hypothesis)
        cer = compute_cer(reference, hypothesis)
        accuracy = (1.0 - wer) * 100
        language_correct = language == detected_language
        rtf = (latency_ms / 1000.0) / max(audio_duration, 1e-3)

        metrics = STTMetrics(
            wer=wer,
            cer=cer,
            accuracy=accuracy,
            language_correct=language_correct,
            latency_ms=latency_ms,
            real_time_factor=rtf,
            reference=reference,
            hypothesis=hypothesis,
            language=language,
            detected_language=detected_language,
        )
        self.results.append(metrics)
        return metrics

    def get_summary(self) -> dict:
        """Aggregate metrics across all evaluations."""
        if not self.results:
            return {}
        n = len(self.results)
        return {
            "total_samples": n,
            "avg_wer": sum(r.wer for r in self.results) / n,
            "avg_cer": sum(r.cer for r in self.results) / n,
            "avg_accuracy": sum(r.accuracy for r in self.results) / n,
            "language_accuracy_pct": sum(r.language_correct for r in self.results) / n * 100,
            "avg_latency_ms": sum(r.latency_ms for r in self.results) / n,
            "avg_real_time_factor": sum(r.real_time_factor for r in self.results) / n,
            "p50_latency_ms": _percentile([r.latency_ms for r in self.results], 50),
            "p90_latency_ms": _percentile([r.latency_ms for r in self.results], 90),
            "p95_latency_ms": _percentile([r.latency_ms for r in self.results], 95),
        }

    def get_per_language_summary(self) -> dict[str, dict]:
        """Per-language breakdown of metrics."""
        langs: dict[str, list[STTMetrics]] = {}
        for r in self.results:
            langs.setdefault(r.language, []).append(r)

        return {
            lang: {
                "count": len(group),
                "avg_wer": sum(r.wer for r in group) / len(group),
                "avg_cer": sum(r.cer for r in group) / len(group),
                "avg_accuracy": sum(r.accuracy for r in group) / len(group),
                "language_accuracy_pct": sum(r.language_correct for r in group) / len(group) * 100,
                "avg_latency_ms": sum(r.latency_ms for r in group) / len(group),
            }
            for lang, group in langs.items()
        }

    def clear(self) -> None:
        self.results = []


def _percentile(values: list[float], p: float) -> float:
    """Compute a percentile (linear interpolation)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)
