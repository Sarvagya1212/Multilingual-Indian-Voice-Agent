"""Streamlit dashboard for the Multilingual Indian Voice Agent.

Run with:
    streamlit run dashboard/app.py

The dashboard reads turn events from `logs/events.jsonl` (or
`logs/demo_events.jsonl` if no real logs exist), aggregates them, and
displays KPIs, charts, and a recent-conversations viewer.

Modules:
    data_source          — load + aggregate events
    components/metrics   — top-row KPI strip
    components/charts    — daily, latency, language charts
    components/conversation_viewer — recent turns

All on-disk paths are configurable via the sidebar so a developer can
point the dashboard at any JSONL log file.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so the `dashboard` package is importable
# regardless of whether Streamlit changes the working directory.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from dashboard.components.charts import (
    render_daily_conversations,
    render_language_distribution,
    render_latency_by_component,
)
from dashboard.components.conversation_viewer import render_conversations
from dashboard.components.metrics import render_quality_metrics, render_top_metrics
from dashboard.data_source import (
    aggregate_metrics,
    load_events,
    load_recent_conversations,
)

DEFAULT_LOG_PATH = Path("logs/events.jsonl")

st.set_page_config(
    page_title="Voice Agent Dashboard",
    page_icon="[V]",
    layout="wide",
)

# Sidebar — settings
st.sidebar.header("Settings")
log_path_str = st.sidebar.text_input(
    "Log path (JSONL)",
    value=str(DEFAULT_LOG_PATH),
    help="Path to a JSONL file with turn_complete events. Falls back to demo data if missing.",
)
refresh_interval = st.sidebar.slider("Refresh (seconds)", 0, 60, 0)
limit = st.sidebar.slider("Recent conversations shown", 5, 50, 20)

# Load + aggregate
log_path = Path(log_path_str)
events = load_events(log_path)
metrics = aggregate_metrics(events)
conversations = load_recent_conversations(events, limit=limit)

# Main layout
st.title("Voice Agent Dashboard")
st.caption(
    f"Loaded {len(events)} events from `{log_path}` "
    f"({'demo data' if log_path == DEFAULT_LOG_PATH and not log_path.exists() else 'live'})."
)

render_top_metrics(metrics)
render_quality_metrics(metrics)

st.header("Metrics Over Time")
tab1, tab2, tab3 = st.tabs(["Conversations", "Latency", "Language"])
with tab1:
    render_daily_conversations(metrics)
with tab2:
    render_latency_by_component(metrics)
with tab3:
    render_language_distribution(metrics)

st.header("Recent Conversations")
render_conversations(conversations)

# Auto-refresh (only if a positive interval is set; 0 = manual refresh only)
if refresh_interval:
    import time

    time.sleep(refresh_interval)
    st.rerun()
