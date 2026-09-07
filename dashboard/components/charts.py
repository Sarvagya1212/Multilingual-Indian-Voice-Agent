"""Charts for the metrics dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def render_daily_conversations(metrics: dict) -> None:
    """Bar chart of daily conversation counts."""
    daily = metrics.get("daily_conversations") or []
    if not daily:
        st.info("No daily data available.")
        return
    df = pd.DataFrame(daily)
    fig = px.bar(df, x="date", y="count", title="Daily Conversations")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_latency_by_component(metrics: dict) -> None:
    """Bar chart of mean latency per component."""
    latencies = metrics.get("latency_by_component") or {}
    if not latencies:
        st.info("No latency data available.")
        return
    df = pd.DataFrame(
        [
            {"component": comp, "avg_ms": val}
            for comp, val in latencies.items()
        ]
    )
    fig = px.bar(
        df,
        x="component",
        y="avg_ms",
        title="Average Latency by Component (ms)",
        color="component",
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_language_distribution(metrics: dict) -> None:
    """Pie chart of language distribution."""
    langs = metrics.get("language_distribution") or {}
    if not langs or sum(langs.values()) == 0:
        st.info("No language data available.")
        return
    df = pd.DataFrame(
        [
            {"language": k, "count": v}
            for k, v in langs.items()
        ]
    )
    fig = px.pie(
        df,
        values="count",
        names="language",
        title="Language Distribution",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
