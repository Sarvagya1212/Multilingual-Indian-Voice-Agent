"""Recent-conversation viewer."""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def _format_eval(evaluation: Dict[str, Any]) -> str:
    if not evaluation:
        return "n/a"
    parts = []
    for key in ("relevance", "naturalness", "groundedness"):
        v = evaluation.get(key)
        if isinstance(v, (int, float)):
            parts.append(f"{key}={v:.2f}")
    return ", ".join(parts) if parts else "n/a"


def render_conversations(conversations: List[Dict[str, Any]]) -> None:
    """Render the recent-conversations section as collapsible expanders."""
    if not conversations:
        st.info("No conversations yet. Run the pipeline or check the log path.")
        return

    for conv in conversations:
        session_id = conv.get("session_id", "unknown")
        ts = conv.get("timestamp", "")
        lang = conv.get("language", "n/a")
        header = f"Session {session_id} — {ts} — {lang}"
        with st.expander(header):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Transcript")
                st.markdown(f"**User:** {conv.get('transcript', '')}")
                st.markdown(f"**Agent:** {conv.get('response', '')}")

            with col2:
                st.subheader("Metrics")
                tools = conv.get("tools_used") or []
                st.write(f"Tools used: {', '.join(tools) if tools else '(none)'}")
                st.write(
                    f"Latency (ms): "
                    f"STT={conv.get('stt_latency_ms', '?')}, "
                    f"LLM={conv.get('llm_latency_ms', '?')}, "
                    f"TTS={conv.get('tts_latency_ms', '?')}, "
                    f"RAG={conv.get('rag_latency_ms', '?')}"
                )
                intent = conv.get("intent")
                if intent:
                    st.write(f"Intent: {intent}")
                st.write(f"Evaluation: {_format_eval(conv.get('evaluation') or {})}")
