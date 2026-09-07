"""Data source for the metrics dashboard.

Loads metrics from local JSONL logs (one event per line) and produces
aggregated views for the dashboard. Falls back to demo data when no
logs are present so the dashboard renders out-of-the-box.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default log location (created at runtime by the pipeline; safe if absent)
DEFAULT_LOG_PATH = Path("logs/events.jsonl")
DEMO_LOG_PATH = Path("logs/demo_events.jsonl")


def _load_events(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL events from `path`; return [] on missing/corrupt lines."""
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # skip corrupt line rather than failing the whole dashboard
            continue
    return events


def _ensure_demo_data(path: Path) -> None:
    """Write a small synthetic dataset so the dashboard is non-empty in demos."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    languages = ["en", "hi", "hinglish"]
    tools = [
        "search_courses",
        "get_course_details",
        "check_eligibility",
        "get_fee_structure",
        "schedule_demo",
    ]
    intents = ["course_inquiry", "fee_question", "eligibility", "demo_booking", "general"]

    transcripts = [
        ("JEE ke liye course dhoondh raha hoon", "hi", "course_inquiry"),
        ("What is the fee for NEET coaching?", "en", "fee_question"),
        ("Class 11 ka student hoon, kya main JEE ke liye eligible hoon?", "hinglish", "eligibility"),
        ("I want to schedule a demo for IIT JEE course", "en", "demo_booking"),
        ("CBSE boards preparation course available hai kya?", "hinglish", "course_inquiry"),
        ("AIIMS ke liye hostel facility hai kya?", "hi", "general"),
        ("Scholarship ka kya criteria hai?", "hi", "fee_question"),
    ]
    responses = [
        "We have JEE courses starting in June. Would you like details?",
        "The NEET course fee is INR 1,50,000 with EMI options available.",
        "Yes, Class 11 students are eligible. You need 75% in Class 10.",
        "Demo booked for IIT JEE course. You will receive a confirmation SMS.",
        "Yes, we offer CBSE boards preparation for Classes 9-12.",
        "Hostel facility is available at our Hyderabad and Bangalore centers.",
        "Scholarships are available for students with 90%+ in previous class.",
    ]

    rng = random.Random(42)
    base = datetime.now() - timedelta(days=14)
    lines: List[str] = []
    for i in range(45):
        ts = base + timedelta(hours=i * 7, minutes=rng.randint(0, 59))
        idx = rng.randint(0, len(transcripts) - 1)
        transcript, lang, intent = transcripts[idx]
        lines.append(
            json.dumps(
                {
                    "event": "turn_complete",
                    "session_id": f"conv_{i + 1:03d}",
                    "timestamp": ts.isoformat(),
                    "language": lang,
                    "intent": intent,
                    "transcript": transcript,
                    "response": responses[idx],
                    "tools_used": rng.sample(tools, k=rng.randint(0, 2)),
                    "stt_latency_ms": rng.randint(180, 520),
                    "llm_latency_ms": rng.randint(450, 1500),
                    "tts_latency_ms": rng.randint(280, 700),
                    "rag_latency_ms": rng.randint(80, 320),
                    "evaluation": {
                        "relevance": round(rng.uniform(0.65, 0.98), 2),
                        "naturalness": round(rng.uniform(0.6, 0.95), 2),
                        "groundedness": round(rng.uniform(0.55, 0.95), 2),
                    },
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_events(log_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load events from disk; seed demo data if no real logs exist.

    The first source tried is the configured log path. If it's empty
    AND we are in demo mode (i.e. caller passed DEFAULT_LOG_PATH or
    no path at all), we populate DEMO_LOG_PATH so the dashboard
    is not blank.
    """
    path = log_path or DEFAULT_LOG_PATH
    events = _load_events(path)
    if not events and path == DEFAULT_LOG_PATH:
        _ensure_demo_data(DEMO_LOG_PATH)
        events = _load_events(DEMO_LOG_PATH)
    return events


def aggregate_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce raw events into the summary metrics the dashboard displays."""
    if not events:
        return {
            "total_conversations": 0,
            "avg_latency_ms": 0,
            "stt_wer": 0.0,
            "tool_success_rate": 0.0,
            "rag_groundedness": 0.0,
            "language_distribution": {"en": 0, "hi": 0, "hinglish": 0},
            "daily_conversations": [],
            "latency_by_component": {
                "STT": 0,
                "LLM": 0,
                "TTS": 0,
                "RAG": 0,
            },
        }

    daily: Counter = Counter()
    langs: Counter = Counter()
    latencies: Dict[str, List[float]] = defaultdict(list)
    groundedness_scores: List[float] = []
    tool_calls = 0
    tool_failures = 0

    for ev in events:
        # Daily bucket
        ts = ev.get("timestamp")
        if ts:
            day = ts[:10]
            daily[day] += 1

        # Language distribution
        lang = ev.get("language")
        if lang:
            langs[lang] += 1

        # Latency
        for key in ("stt_latency_ms", "llm_latency_ms", "tts_latency_ms", "rag_latency_ms"):
            value = ev.get(key)
            if isinstance(value, (int, float)):
                component = {"stt_latency_ms": "STT", "llm_latency_ms": "LLM", "tts_latency_ms": "TTS", "rag_latency_ms": "RAG"}[key]
                latencies[component].append(float(value))

        # Groundedness (proxy: from evaluation block)
        ev_eval = ev.get("evaluation") or {}
        if isinstance(ev_eval, dict):
            g = ev_eval.get("groundedness")
            if isinstance(g, (int, float)):
                groundedness_scores.append(float(g))

        # Tools
        tools = ev.get("tools_used") or []
        if tools:
            tool_calls += len(tools)
            # Assume any tool_used was a success; failures would be logged
            # via separate 'tool_error' events which we don't currently emit
            tool_failures += sum(1 for _ in tools if ev.get("tool_error"))

    total = len(events)
    avg_latency = (
        sum(sum(v) for v in latencies.values()) / max(1, sum(len(v) for v in latencies.values()))
        if latencies
        else 0
    )
    avg_groundedness = (
        sum(groundedness_scores) / len(groundedness_scores) if groundedness_scores else 0.0
    )

    daily_conversations = [
        {"date": d, "count": c} for d, c in sorted(daily.items())
    ]

    latency_by_component = {
        comp: (sum(vals) / len(vals)) if vals else 0
        for comp, vals in latencies.items()
    }

    return {
        "total_conversations": total,
        "avg_latency_ms": round(avg_latency, 1),
        "stt_wer": 0.0,  # requires labelled transcripts; not derivable from raw events
        "tool_success_rate": (1 - (tool_failures / tool_calls)) if tool_calls else 0.0,
        "rag_groundedness": round(avg_groundedness, 3),
        "language_distribution": dict(langs) or {"en": 0, "hi": 0, "hinglish": 0},
        "daily_conversations": daily_conversations,
        "latency_by_component": latency_by_component,
    }


def load_recent_conversations(
    events: List[Dict[str, Any]], limit: int = 20
) -> List[Dict[str, Any]]:
    """Return the most recent N turn_complete events sorted by timestamp desc."""
    turns = [e for e in events if e.get("event") == "turn_complete"]
    turns.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return turns[:limit]
