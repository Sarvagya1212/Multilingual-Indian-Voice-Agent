"""Tests for the dashboard web interface.

Validates that the HTML, CSS, and JS files exist, are syntactically valid,
and reference the elements expected by the app.js code.
"""
import json
import re
from pathlib import Path

import pytest


DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"


def _read(name: str) -> str:
    """Read a dashboard file."""
    return (DASHBOARD_DIR / name).read_text(encoding="utf-8")


class TestDashboardFiles:
    def test_index_html_exists(self):
        """index.html must exist."""
        assert (DASHBOARD_DIR / "index.html").exists()

    def test_styles_css_exists(self):
        """styles.css must exist."""
        assert (DASHBOARD_DIR / "styles.css").exists()

    def test_app_js_exists(self):
        """app.js must exist."""
        assert (DASHBOARD_DIR / "app.js").exists()


class TestIndexHtml:
    def test_doctype(self):
        """HTML must start with <!DOCTYPE html>."""
        html = _read("index.html")
        assert html.lstrip().lower().startswith("<!doctype html>")

    def test_title(self):
        """HTML must have a title."""
        html = _read("index.html")
        assert re.search(r"<title>.*</title>", html, re.IGNORECASE)

    def test_meta_charset(self):
        """HTML must declare UTF-8."""
        html = _read("index.html")
        assert re.search(r'<meta\s+charset="utf-8"', html, re.IGNORECASE)

    def test_stylesheet_linked(self):
        """HTML must link the stylesheet."""
        html = _read("index.html")
        assert 'rel="stylesheet"' in html
        assert "styles.css" in html

    def test_script_included(self):
        """HTML must include the app.js script."""
        html = _read("index.html")
        assert "app.js" in html

    def test_required_element_ids(self):
        """HTML must contain all element IDs referenced by app.js."""
        html = _read("index.html")
        required = [
            "status",
            "state",
            "language",
            "session",
            "messages",
            "mic-btn",
            "interrupt-btn",
            "level-meter",
        ]
        for elem_id in required:
            assert f'id="{elem_id}"' in html, f"Missing id={elem_id}"


class TestStylesCss:
    def test_uses_css_variables(self):
        """CSS must define :root variables."""
        css = _read("styles.css")
        assert ":root" in css
        assert "--primary" in css

    def test_mic_button_style(self):
        """CSS must style the mic-btn class."""
        css = _read("styles.css")
        assert ".mic-btn" in css
        assert "border-radius: 50%" in css

    def test_recording_state(self):
        """CSS must have a recording state."""
        css = _read("styles.css")
        assert ".mic-btn.recording" in css

    def test_responsive(self):
        """CSS must include a responsive @media rule."""
        css = _read("styles.css")
        assert "@media" in css


class TestAppJs:
    def test_voice_agent_class(self):
        """JS must define a VoiceAgent class."""
        js = _read("app.js")
        assert "class VoiceAgent" in js

    def test_dom_ready_handler(self):
        """JS must register a DOMContentLoaded handler."""
        js = _read("app.js")
        assert "DOMContentLoaded" in js

    def test_websocket_used(self):
        """JS must use WebSocket."""
        js = _read("app.js")
        assert "WebSocket" in js

    def test_microphone_access(self):
        """JS must use getUserMedia for the microphone."""
        js = _read("app.js")
        assert "getUserMedia" in js

    def test_message_handler_switch(self):
        """JS must handle 'status' and 'turn_complete' messages."""
        js = _read("app.js")
        assert "turn_complete" in js
        assert "case 'status'" in js

    def test_syntax_check(self):
        """JS file should be syntactically valid (very basic check)."""
        js = _read("app.js")
        # Brace balance check
        assert js.count("{") == js.count("}"), "Unbalanced braces"
        assert js.count("(") == js.count(")"), "Unbalanced parens"
        # No obvious syntax issues
        assert "function" in js or "class" in js


# Run with: pytest tests/test_dashboard.py -v
