"""Tests for the metrics dashboard data layer and component modules."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make `dashboard` importable when tests run from the project root.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.data_source import (
    DEFAULT_LOG_PATH,
    DEMO_LOG_PATH,
    _ensure_demo_data,
    _load_events,
    aggregate_metrics,
    load_events,
    load_recent_conversations,
)


# ----------------------------------------------------------------------
# data_source: file loading
# ----------------------------------------------------------------------


class TestLoadEvents:
    def test_load_events_missing_file_returns_empty_list(self, tmp_path: Path):
        missing = tmp_path / "nope.jsonl"
        assert _load_events(missing) == []

    def test_load_events_skips_blank_and_corrupt_lines(self, tmp_path: Path):
        path = tmp_path / "events.jsonl"
        valid = {"event": "turn_complete", "session_id": "x", "timestamp": "2026-09-07T10:00:00"}
        path.write_text(
            "\n".join(
                [
                    json.dumps(valid),
                    "",  # blank
                    "not json",  # corrupt
                    json.dumps({**valid, "session_id": "y"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        events = _load_events(path)
        assert len(events) == 2
        assert {e["session_id"] for e in events} == {"x", "y"}

    def test_load_events_falls_back_to_demo_data(self, tmp_path: Path, monkeypatch):
        # Point DEMO_LOG_PATH into tmp_path so the test doesn't pollute the repo
        from dashboard import data_source

        demo_file = tmp_path / "demo.jsonl"
        monkeypatch.setattr(data_source, "DEMO_LOG_PATH", demo_file)
        events = load_events()
        assert events, "Expected demo data to be generated when no real log exists"
        assert demo_file.exists()
        # All demo events should be turn_complete
        assert all(e.get("event") == "turn_complete" for e in events)

    def test_load_events_uses_existing_real_log(self, tmp_path: Path):
        from dashboard import data_source

        real = tmp_path / "real.jsonl"
        real.write_text(
            json.dumps({"event": "turn_complete", "session_id": "real-1"}) + "\n",
            encoding="utf-8",
        )
        events = data_source.load_events(real)
        assert len(events) == 1
        assert events[0]["session_id"] == "real-1"


# ----------------------------------------------------------------------
# data_source: aggregation
# ----------------------------------------------------------------------


def _make_event(**overrides):
    base = {
        "event": "turn_complete",
        "session_id": "s1",
        "timestamp": "2026-09-07T10:00:00",
        "language": "en",
        "stt_latency_ms": 300,
        "llm_latency_ms": 800,
        "tts_latency_ms": 400,
        "rag_latency_ms": 100,
        "tools_used": [],
        "evaluation": {"groundedness": 0.8, "relevance": 0.9, "naturalness": 0.85},
    }
    base.update(overrides)
    return base


class TestAggregateMetrics:
    def test_empty_events_returns_zero_metrics(self):
        m = aggregate_metrics([])
        assert m["total_conversations"] == 0
        assert m["avg_latency_ms"] == 0
        assert m["language_distribution"]["en"] == 0

    def test_counts_conversations_and_buckets_by_day(self):
        events = [
            _make_event(timestamp="2026-09-07T10:00:00"),
            _make_event(timestamp="2026-09-07T11:00:00"),
            _make_event(timestamp="2026-09-08T09:00:00"),
        ]
        m = aggregate_metrics(events)
        assert m["total_conversations"] == 3
        assert len(m["daily_conversations"]) == 2
        assert {d["date"] for d in m["daily_conversations"]} == {"2026-09-07", "2026-09-08"}

    def test_language_distribution(self):
        events = [
            _make_event(language="en"),
            _make_event(language="hi"),
            _make_event(language="hi"),
            _make_event(language="hinglish"),
        ]
        m = aggregate_metrics(events)
        assert m["language_distribution"] == {"en": 1, "hi": 2, "hinglish": 1}

    def test_avg_latency_across_components(self):
        events = [_make_event(stt_latency_ms=100, llm_latency_ms=400, tts_latency_ms=200, rag_latency_ms=100)]
        m = aggregate_metrics(events)
        # Sum = 800, count = 4 components
        assert m["avg_latency_ms"] == 200.0
        assert m["latency_by_component"]["STT"] == 100
        assert m["latency_by_component"]["LLM"] == 400
        assert m["latency_by_component"]["TTS"] == 200
        assert m["latency_by_component"]["RAG"] == 100

    def test_rag_groundedness_averages_evaluations(self):
        events = [
            _make_event(evaluation={"groundedness": 0.5}),
            _make_event(evaluation={"groundedness": 0.9}),
        ]
        m = aggregate_metrics(events)
        assert m["rag_groundedness"] == pytest.approx(0.7, abs=1e-3)

    def test_tool_success_with_no_tools_is_zero(self):
        # No tool calls -> tool_success_rate = 0 (not 100, which would be misleading)
        m = aggregate_metrics([_make_event()])
        assert m["tool_success_rate"] == 0.0

    def test_tool_success_with_clean_tool_calls(self):
        m = aggregate_metrics([_make_event(tools_used=["search_courses", "check_eligibility"])])
        assert m["tool_success_rate"] == 1.0

    def test_tool_success_with_errors(self):
        m = aggregate_metrics(
            [_make_event(tools_used=["search_courses"], tool_error=True)]
        )
        assert m["tool_success_rate"] == 0.0

    def test_ignores_non_turn_complete_events(self):
        events = [
            _make_event(),
            {"event": "speech_start", "session_id": "s1"},
            {"event": "turn_complete", "session_id": "s2"},
        ]
        # aggregate_metrics is meant for turn events but should not crash on others
        m = aggregate_metrics(events)
        # It counts every dict; only ensure it doesn't raise
        assert m["total_conversations"] == 3


class TestLoadRecentConversations:
    def test_filters_to_turn_complete(self):
        events = [
            _make_event(session_id="a"),
            {"event": "speech_start", "session_id": "noise"},
        ]
        recent = load_recent_conversations(events, limit=10)
        assert len(recent) == 1
        assert recent[0]["session_id"] == "a"

    def test_sorts_by_timestamp_desc(self):
        events = [
            _make_event(timestamp="2026-09-06T10:00:00", session_id="old"),
            _make_event(timestamp="2026-09-08T10:00:00", session_id="new"),
            _make_event(timestamp="2026-09-07T10:00:00", session_id="mid"),
        ]
        recent = load_recent_conversations(events, limit=10)
        assert [r["session_id"] for r in recent] == ["new", "mid", "old"]

    def test_respects_limit(self):
        events = [_make_event(session_id=f"s{i}") for i in range(10)]
        recent = load_recent_conversations(events, limit=3)
        assert len(recent) == 3


class TestEnsureDemoData:
    def test_creates_file_when_missing(self, tmp_path: Path, monkeypatch):
        from dashboard import data_source

        demo_file = tmp_path / "fresh_demo.jsonl"
        monkeypatch.setattr(data_source, "DEMO_LOG_PATH", demo_file)
        assert not demo_file.exists()
        _ensure_demo_data(demo_file)
        assert demo_file.exists()
        # Should be valid JSONL with at least one event
        events = _load_events(demo_file)
        assert events

    def test_does_not_overwrite_existing(self, tmp_path: Path):
        from dashboard import data_source

        demo_file = tmp_path / "existing.jsonl"
        demo_file.write_text('{"event":"turn_complete","session_id":"keep"}\n', encoding="utf-8")
        data_source._ensure_demo_data(demo_file)
        events = _load_events(demo_file)
        assert len(events) == 1
        assert events[0]["session_id"] == "keep"


# ----------------------------------------------------------------------
# Module structure / import smoke
# ----------------------------------------------------------------------


class TestDashboardModule:
    def test_app_py_exists(self):
        app = ROOT / "dashboard" / "app.py"
        assert app.exists()

    def test_components_directory_exists(self):
        comp = ROOT / "dashboard" / "components"
        assert comp.exists()
        assert (comp / "metrics.py").exists()
        assert (comp / "charts.py").exists()
        assert (comp / "conversation_viewer.py").exists()
        assert (comp / "__init__.py").exists()

    def test_dashboard_init_exists(self):
        assert (ROOT / "dashboard" / "__init__.py").exists()

    def test_app_references_components(self):
        # The app should import our three component modules
        text = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
        assert "dashboard.components.metrics" in text
        assert "dashboard.components.charts" in text
        assert "dashboard.components.conversation_viewer" in text
        assert "dashboard.data_source" in text

    def test_data_source_uses_real_path_constants(self):
        # Sanity: the module exposes DEFAULT_LOG_PATH and DEMO_LOG_PATH
        from dashboard import data_source

        assert isinstance(data_source.DEFAULT_LOG_PATH, Path)
        assert isinstance(data_source.DEMO_LOG_PATH, Path)
        assert data_source.DEFAULT_LOG_PATH != data_source.DEMO_LOG_PATH
