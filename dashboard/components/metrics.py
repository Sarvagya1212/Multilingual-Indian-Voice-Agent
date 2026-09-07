"""Top-row KPI metrics for the dashboard."""
from __future__ import annotations

import streamlit as st

from dashboard.data_source import aggregate_metrics


def render_top_metrics(metrics: dict) -> None:
    """Render the 4-column top KPI strip."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Conversations",
            metrics["total_conversations"],
            delta=None,
            help="Number of turn_complete events loaded from logs",
        )

    with col2:
        st.metric(
            "Avg Latency (ms)",
            f"{metrics['avg_latency_ms']:.0f}",
            delta=None,
            help="Mean of STT + LLM + TTS + RAG latencies across all turns",
        )

    with col3:
        st.metric(
            "RAG Groundedness",
            f"{metrics['rag_groundedness']:.1%}",
            delta=None,
            help="Mean groundedness score from per-turn evaluations",
        )

    with col4:
        st.metric(
            "Tool Success",
            f"{metrics['tool_success_rate']:.1%}",
            delta=None,
            help="Share of tool calls that did not produce an error",
        )


def render_quality_metrics(metrics: dict) -> None:
    """Render a secondary row with STT WER and language distribution."""
    col1, col2 = st.columns(2)

    with col1:
        # STT WER is not derivable from raw events (needs labelled transcripts);
        # we show 0.0 (no data) honestly rather than fabricating a number.
        st.metric(
            "STT WER",
            "n/a" if metrics["stt_wer"] == 0 else f"{metrics['stt_wer']:.1%}",
            delta=None,
            help=(
                "Requires labelled transcripts to compute. "
                "Run `python -m evaluations.run --component stt` "
                "to populate this metric."
            ),
        )

    with col2:
        langs = metrics["language_distribution"]
        total = sum(langs.values()) or 1
        top = max(langs, key=langs.get) if langs else "n/a"
        st.metric(
            "Top Language",
            top,
            delta=f"{(langs.get(top, 0) / total):.0%} of turns" if top != "n/a" else None,
            help="Most common detected language",
        )
