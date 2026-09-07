"""Demo 8: Dashboard.

Boots the Streamlit dashboard data layer with synthetic events and
prints a summary of what the dashboard would render. We don't actually
launch Streamlit from this script (that's `streamlit run dashboard/app.py`)
— we exercise the same data layer the dashboard uses and show its
output so the demo is non-interactive and CI-friendly.
"""
from __future__ import annotations

from demos._common import header, info, ok, run, step, warn


async def main() -> None:
    header("Demo 8: Metrics Dashboard")
    info("Exercise the dashboard data layer with synthetic events.")
    info("To launch the actual UI: streamlit run dashboard/app.py")
    print()

    try:
        from dashboard.data_source import (
            aggregate_metrics,
            load_events,
            load_recent_conversations,
        )
    except Exception as e:
        warn(f"dashboard module not importable: {e}")
        return

    step(1, 4, "Load events")
    events = load_events()
    info(f"Loaded {len(events)} turn_complete events.")
    print()

    step(2, 4, "Compute aggregate metrics")
    metrics = aggregate_metrics(events)
    print(f"  Total conversations: {metrics['total_conversations']}")
    print(f"  Avg latency (ms): {metrics['avg_latency_ms']:.0f}")
    print(f"  RAG groundedness: {metrics['rag_groundedness']:.1%}")
    print(f"  Tool success rate: {metrics['tool_success_rate']:.1%}")
    langs = metrics.get("language_distribution", {})
    if langs:
        print(f"  Language distribution: {langs}")
    print()

    step(3, 4, "Latency by component")
    for component, latency in metrics.get("latency_by_component", {}).items():
        print(f"  {component}: {latency:.0f} ms")
    print()

    step(4, 4, "Recent conversations")
    recent = load_recent_conversations(events, limit=3)
    for i, conv in enumerate(recent, 1):
        print(f"  {i}. [{conv.get('session_id', '?')}] {conv.get('language', '?')}")
        transcript = conv.get("transcript", "")
        if len(transcript) > 60:
            transcript = transcript[:57] + "..."
        print(f"     \"{transcript}\"")
    print()

    ok("Dashboard demo completed. Run `streamlit run dashboard/app.py` to view the UI.")


if __name__ == "__main__":
    run(main)
