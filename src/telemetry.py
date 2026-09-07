"""Live telemetry writer for the voice agent.

Writes a JSONL line per completed turn to `logs/events.jsonl`. The dashboard
(`dashboard/data_source.py`) reads that file to render live metrics.

Design goals:
- **Optional**: disabled by default in tests; enabled when `TELEMETRY_ENABLED=true`
  in the environment. No-op when disabled so demos/tests stay hermetic.
- **Resilient**: write failures are logged but never crash the pipeline.
- **Schema-compatible**: the on-disk schema matches what the dashboard expects.
- **Thread-safe**: a single lock guards the append (avoids interleaved lines
  when many turns finish concurrently).

Schema per line (one event per turn):

    {
      "event": "turn_complete",
      "session_id": "...",
      "timestamp": "2026-09-07T14:23:11.123456Z",
      "language": "en" | "hi" | "hinglish",
      "transcript": "user said...",
      "response": "agent said...",
      "tools_used": ["search_courses"],
      "tool_error": false,
      "stt_latency_ms": 412.0,
      "llm_latency_ms": 830.0,
      "tts_latency_ms": 290.0,
      "rag_latency_ms": 0.0,
      "total_latency_ms": 1532.0,
      "evaluation": {"groundedness": 0.0}
    }
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Default location; the dashboard looks here first.
DEFAULT_LOG_PATH = Path("logs/events.jsonl")


def _is_enabled() -> bool:
    """Telemetry is opt-in. Enable with `TELEMETRY_ENABLED=true` in the env."""
    return os.getenv("TELEMETRY_ENABLED", "").lower() in ("1", "true", "yes")


class TelemetryWriter:
    """Append-only JSONL writer for turn events.

    A single instance per process is sufficient; the write is guarded by a lock
    so concurrent turns cannot interleave their lines.
    """

    def __init__(
        self,
        path: Path = DEFAULT_LOG_PATH,
        enabled: Optional[bool] = None,
    ) -> None:
        self.path = Path(path)
        self.enabled = enabled if enabled is not None else _is_enabled()
        self._lock = threading.Lock()
        self._events_written = 0
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ writes

    def write_turn(
        self,
        *,
        session_id: str,
        language: str,
        transcript: str,
        response: str,
        stt_latency_ms: float,
        llm_latency_ms: float,
        tts_latency_ms: float,
        total_latency_ms: float,
        tools_used: Optional[Iterable[str]] = None,
        rag_latency_ms: float = 0.0,
        tool_error: bool = False,
        evaluation: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Append a single `turn_complete` event. Returns True on success.

        No-op (returns False) when telemetry is disabled.
        """
        if not self.enabled:
            return False

        ts = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
        event: Dict[str, Any] = {
            "event": "turn_complete",
            "session_id": session_id,
            "timestamp": ts.isoformat().replace("+00:00", "Z"),
            "language": language,
            "transcript": transcript,
            "response": response,
            "tools_used": list(tools_used or []),
            "tool_error": tool_error,
            "stt_latency_ms": round(float(stt_latency_ms), 1),
            "llm_latency_ms": round(float(llm_latency_ms), 1),
            "tts_latency_ms": round(float(tts_latency_ms), 1),
            "rag_latency_ms": round(float(rag_latency_ms), 1),
            "total_latency_ms": round(float(total_latency_ms), 1),
            "evaluation": dict(evaluation or {}),
        }

        line = json.dumps(event, ensure_ascii=False) + "\n"
        try:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line)
            self._events_written += 1
            return True
        except OSError:
            # Never crash the pipeline because logging failed.
            return False

    # ----------------------------------------------------------------- readers

    def read_events(self) -> List[Dict[str, Any]]:
        """Read back all events written to this writer's path (for tests)."""
        if not self.path.exists():
            return []
        events: List[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    @property
    def events_written(self) -> int:
        return self._events_written

    def reset(self) -> None:
        """Delete the log file. Used by tests; never call in production."""
        if self.path.exists():
            self.path.unlink()
        self._events_written = 0


# --------------------------------------------------------------------- global

# A process-wide default writer. Set `TELEMETRY_ENABLED=true` to activate.
_default_writer: Optional[TelemetryWriter] = None
_default_lock = threading.Lock()


def get_writer(path: Optional[Path] = None) -> TelemetryWriter:
    """Return the process-wide TelemetryWriter (lazy-initialized)."""
    global _default_writer
    with _default_lock:
        if _default_writer is None:
            _default_writer = TelemetryWriter(path=path or DEFAULT_LOG_PATH)
        return _default_writer


def write_turn_event(**kwargs: Any) -> bool:
    """Convenience wrapper around `TelemetryWriter.write_turn` on the default writer."""
    return get_writer().write_turn(**kwargs)


def reset_default_writer() -> None:
    """Drop the cached default writer (used by tests)."""
    global _default_writer
    with _default_lock:
        _default_writer = None
