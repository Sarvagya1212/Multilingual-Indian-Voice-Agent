"""Tests for src.telemetry — the JSONL turn-event writer."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Reset the module under test + the cached default writer between tests.
import src.telemetry as telemetry
from src.telemetry import (
    DEFAULT_LOG_PATH,
    TelemetryWriter,
    get_writer,
    reset_default_writer,
    write_turn_event,
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    """Each test gets a fresh tmp log file and a fresh default writer."""
    log = tmp_path / "events.jsonl"
    monkeypatch.setattr(telemetry, "_default_writer", None)
    yield log
    if log.exists():
        log.unlink()


# ---------------------------------------------------------------- enable flag


class TestEnableFlag:
    def test_default_is_disabled(self, _clean):
        """Without TELEMETRY_ENABLED, writes are no-ops."""
        w = TelemetryWriter(path=_clean)
        assert w.enabled is False
        assert w.write_turn(
            session_id="s1", language="en", transcript="hi", response="hello",
            stt_latency_ms=100, llm_latency_ms=200, tts_latency_ms=50,
            total_latency_ms=350,
        ) is False
        assert not _clean.exists()

    def test_enabled_when_env_var_set(self, _clean, monkeypatch):
        monkeypatch.setenv("TELEMETRY_ENABLED", "true")
        w = TelemetryWriter(path=_clean)
        assert w.enabled is True
        assert w.write_turn(
            session_id="s1", language="en", transcript="hi", response="hello",
            stt_latency_ms=100, llm_latency_ms=200, tts_latency_ms=50,
            total_latency_ms=350,
        ) is True

    def test_explicit_enabled_overrides_env(self, _clean, monkeypatch):
        monkeypatch.setenv("TELEMETRY_ENABLED", "false")
        w = TelemetryWriter(path=_clean, enabled=True)
        assert w.enabled is True
        assert w.write_turn(
            session_id="s1", language="en", transcript="hi", response="hello",
            stt_latency_ms=100, llm_latency_ms=200, tts_latency_ms=50,
            total_latency_ms=350,
        ) is True


# ----------------------------------------------------------------- schema


class TestSchema:
    def test_writes_valid_jsonl(self, _clean):
        w = TelemetryWriter(path=_clean, enabled=True)
        w.write_turn(
            session_id="sess-1",
            language="en",
            transcript="what is JEE?",
            response="JEE is the engineering entrance exam.",
            stt_latency_ms=412.3,
            llm_latency_ms=830.7,
            tts_latency_ms=290.1,
            total_latency_ms=1533.1,
        )
        lines = _clean.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event"] == "turn_complete"
        assert event["session_id"] == "sess-1"
        assert event["language"] == "en"
        assert event["transcript"] == "what is JEE?"
        assert event["response"] == "JEE is the engineering entrance exam."
        assert event["stt_latency_ms"] == 412.3
        assert event["llm_latency_ms"] == 830.7
        assert event["tts_latency_ms"] == 290.1
        assert event["total_latency_ms"] == 1533.1
        assert event["tools_used"] == []
        assert event["tool_error"] is False
        assert event["rag_latency_ms"] == 0.0
        # Timestamp parses + ends in 'Z' (UTC)
        ts = event["timestamp"]
        assert ts.endswith("Z")
        datetime.fromisoformat(ts.replace("Z", "+00:00"))

    def test_includes_tools_used(self, _clean):
        w = TelemetryWriter(path=_clean, enabled=True)
        w.write_turn(
            session_id="s", language="hi", transcript="x", response="y",
            stt_latency_ms=1, llm_latency_ms=1, tts_latency_ms=1, total_latency_ms=3,
            tools_used=["search_courses", "check_eligibility"],
        )
        event = json.loads(_clean.read_text().strip())
        assert event["tools_used"] == ["search_courses", "check_eligibility"]

    def test_includes_evaluation_block(self, _clean):
        w = TelemetryWriter(path=_clean, enabled=True)
        w.write_turn(
            session_id="s", language="en", transcript="x", response="y",
            stt_latency_ms=1, llm_latency_ms=1, tts_latency_ms=1, total_latency_ms=3,
            evaluation={"groundedness": 0.85, "relevance": 0.92},
        )
        event = json.loads(_clean.read_text().strip())
        assert event["evaluation"] == {"groundedness": 0.85, "relevance": 0.92}

    def test_handles_non_ascii(self, _clean):
        """Devanagari / Hindi text must round-trip without escaping (cp1252 safety)."""
        w = TelemetryWriter(path=_clean, enabled=True)
        w.write_turn(
            session_id="s", language="hi", transcript="JEE ke liye",
            response="JEE ke liye course hai",
            stt_latency_ms=1, llm_latency_ms=1, tts_latency_ms=1, total_latency_ms=3,
        )
        event = json.loads(_clean.read_text(encoding="utf-8").strip())
        assert event["transcript"] == "JEE ke liye"
        assert event["response"] == "JEE ke liye course hai"


# --------------------------------------------------------------- multiple


class TestMultipleEvents:
    def test_appends_one_line_per_turn(self, _clean):
        w = TelemetryWriter(path=_clean, enabled=True)
        for i in range(5):
            w.write_turn(
                session_id=f"s{i}", language="en", transcript=f"t{i}",
                response=f"r{i}", stt_latency_ms=1, llm_latency_ms=1,
                tts_latency_ms=1, total_latency_ms=3,
            )
        lines = _clean.read_text().strip().splitlines()
        assert len(lines) == 5
        for i, line in enumerate(lines):
            ev = json.loads(line)
            assert ev["session_id"] == f"s{i}"

    def test_events_written_counter(self, _clean):
        w = TelemetryWriter(path=_clean, enabled=True)
        assert w.events_written == 0
        w.write_turn(
            session_id="s", language="en", transcript="x", response="y",
            stt_latency_ms=1, llm_latency_ms=1, tts_latency_ms=1, total_latency_ms=3,
        )
        w.write_turn(
            session_id="s", language="en", transcript="x", response="y",
            stt_latency_ms=1, llm_latency_ms=1, tts_latency_ms=1, total_latency_ms=3,
        )
        assert w.events_written == 2

    def test_read_events_roundtrip(self, _clean):
        w = TelemetryWriter(path=_clean, enabled=True)
        w.write_turn(
            session_id="s", language="hi", transcript="नमस्ते", response="hello",
            stt_latency_ms=1, llm_latency_ms=1, tts_latency_ms=1, total_latency_ms=3,
        )
        events = w.read_events()
        assert len(events) == 1
        assert events[0]["transcript"] == "नमस्ते"

    def test_read_events_missing_file(self, _clean):
        w = TelemetryWriter(path=_clean, enabled=True)
        assert w.read_events() == []


# -------------------------------------------------------------- concurrency


class TestConcurrency:
    def test_concurrent_writes_dont_interleave(self, _clean):
        """100 turns from many threads must each occupy a single line."""
        w = TelemetryWriter(path=_clean, enabled=True)

        def fire(i: int) -> None:
            w.write_turn(
                session_id=f"s{i:03d}", language="en", transcript="x",
                response="y", stt_latency_ms=1, llm_latency_ms=1,
                tts_latency_ms=1, total_latency_ms=3,
            )

        threads = [threading.Thread(target=fire, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = _clean.read_text().strip().splitlines()
        assert len(lines) == 100
        # Every line must parse as JSON (no half-written lines).
        session_ids = {json.loads(line)["session_id"] for line in lines}
        assert len(session_ids) == 100


# -------------------------------------------------------------- resilience


class TestResilience:
    def test_write_failure_returns_false(self, _clean, monkeypatch):
        """A write failure must not raise — the pipeline must keep running."""
        w = TelemetryWriter(path=_clean, enabled=True)
        # Simulate a write failure by patching Path.open to raise OSError.
        original_open = Path.open

        def fail_open(self, *args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "open", fail_open)
        result = w.write_turn(
            session_id="s", language="en", transcript="x", response="y",
            stt_latency_ms=1, llm_latency_ms=1, tts_latency_ms=1, total_latency_ms=3,
        )
        monkeypatch.setattr(Path, "open", original_open)
        assert result is False

    def test_reset_deletes_file(self, _clean):
        w = TelemetryWriter(path=_clean, enabled=True)
        w.write_turn(
            session_id="s", language="en", transcript="x", response="y",
            stt_latency_ms=1, llm_latency_ms=1, tts_latency_ms=1, total_latency_ms=3,
        )
        assert _clean.exists()
        w.reset()
        assert not _clean.exists()
        assert w.events_written == 0


# ----------------------------------------------------- default writer API


class TestDefaultWriter:
    def test_get_writer_returns_singleton(self, _clean):
        w1 = get_writer(path=_clean)
        w2 = get_writer(path=_clean)
        assert w1 is w2

    def test_reset_default_writer_drops_cache(self, _clean):
        w1 = get_writer(path=_clean)
        reset_default_writer()
        w2 = get_writer(path=_clean)
        assert w1 is not w2

    def test_write_turn_event_helper(self, _clean, monkeypatch):
        """The module-level helper should hit the default writer."""
        monkeypatch.setattr(telemetry, "_default_writer",
                            TelemetryWriter(path=_clean, enabled=True))
        assert write_turn_event(
            session_id="s", language="en", transcript="x", response="y",
            stt_latency_ms=1, llm_latency_ms=1, tts_latency_ms=1, total_latency_ms=3,
        ) is True
        events = get_writer(path=_clean).read_events()
        assert len(events) == 1


# ----------------------------------------- integration with dashboard loader


class TestDashboardCompatibility:
    """The dashboard's loader must be able to consume our events unchanged."""

    def test_dashboard_can_load_written_events(self, _clean):
        # Arrange: writer fills the log
        w = TelemetryWriter(path=_clean, enabled=True)
        w.write_turn(
            session_id="s1", language="en", transcript="JEE?", response="JEE is...",
            stt_latency_ms=300, llm_latency_ms=700, tts_latency_ms=200,
            total_latency_ms=1200, tools_used=["search_courses"],
        )
        # Act: dashboard's data_source reads it back and aggregates
        from dashboard.data_source import aggregate_metrics
        events = w.read_events()
        metrics = aggregate_metrics(events)

        # Assert: dashboard sees this as 1 conversation in English
        assert metrics["total_conversations"] == 1
        assert metrics["language_distribution"].get("en") == 1
        assert metrics["latency_by_component"]["STT"] == 300
        assert metrics["latency_by_component"]["LLM"] == 700
        assert metrics["latency_by_component"]["TTS"] == 200

    def test_default_log_path_is_logs_events(self):
        """The writer's default path matches what the dashboard reads first."""
        assert str(DEFAULT_LOG_PATH).endswith("events.jsonl")
        assert "logs" in str(DEFAULT_LOG_PATH)
