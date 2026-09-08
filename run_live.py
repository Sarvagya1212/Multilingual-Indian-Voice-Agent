"""Launch script for the Live Voice Agent.

This script wires the ConversationOrchestrator to the VoiceGateway and starts
both a WebSocket server for the audio streaming and an HTTP server for the UI.

Usage:
    python run_live.py

Opens:
    http://localhost:8080  — voice agent UI (press-and-hold to talk)
    ws://localhost:8765    — WebSocket gateway (audio streaming)
"""
import asyncio
import os
import sys
import webbrowser
from pathlib import Path

from aiohttp import web

from src.logger import setup_logger
from src.config import settings

# Fix Windows cp1252 console encoding for Hindi/Devanagari output
try:
    if sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ---------------------------------------------------------------------------
HTTP_PORT = int(os.getenv("HTTP_PORT", "8080"))
WS_PORT = int(os.getenv("WS_PORT", "8765"))
DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"


# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

def _check_ollama() -> bool:
    """Return True if Ollama is reachable on localhost:11434."""
    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def _check_internet() -> bool:
    """Return True if we can reach the gTTS endpoint."""
    import urllib.request
    try:
        req = urllib.request.Request("https://translate.google.com", method="HEAD")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


def check_prerequisites() -> dict:
    """Run prerequisite checks and print a summary.

    Returns a dict of boolean flags for each check.
    """
    print("\n" + "=" * 64)
    print("  Multilingual Indian Voice Agent — Prerequisite Check")
    print("=" * 64)

    checks = {}

    # -- LLM provider --
    if settings.llm_provider == "ollama":
        ok = _check_ollama()
        checks["ollama"] = ok
        tag = "[OK]" if ok else "[!!]"
        print(f"  {tag} Ollama (LLM)       — {'reachable' if ok else 'NOT reachable at localhost:11434'}")
        if not ok:
            print("       Fix: ollama serve  (in another terminal)")
    elif settings.llm_provider == "anthropic":
        ok = bool(settings.anthropic_api_key)
        checks["anthropic"] = ok
        tag = "[OK]" if ok else "[!!]"
        print(f"  {tag} Anthropic API key  — {'set' if ok else 'MISSING (set ANTHROPIC_API_KEY)'}")
    else:
        print(f"  [??] LLM provider: {settings.llm_provider}")

    # -- TTS provider --
    if settings.tts_provider == "gtts":
        ok = _check_internet()
        checks["network"] = ok
        tag = "[OK]" if ok else "[!!]"
        print(f"  {tag} Internet (gTTS)    — {'reachable' if ok else 'NO internet for Google Translate TTS'}")
    elif settings.tts_provider == "openai":
        ok = bool(settings.openai_api_key)
        checks["openai"] = ok
        tag = "[OK]" if ok else "[!!]"
        print(f"  {tag} OpenAI API key     — {'set' if ok else 'MISSING (set OPENAI_API_KEY)'}")

    # -- STT provider (Whisper runs locally, just check import) --
    try:
        import whisper  # noqa: F401
        checks["whisper"] = True
        print("  [OK] Whisper (STT)       — installed")
    except ImportError:
        checks["whisper"] = False
        print("  [!!] Whisper (STT)       — NOT installed (pip install openai-whisper)")

    # -- Dashboard files --
    html_ok = (DASHBOARD_DIR / "index.html").exists()
    checks["dashboard"] = html_ok
    tag = "[OK]" if html_ok else "[!!]"
    print(f"  {tag} Dashboard files    — {'found' if html_ok else 'MISSING dashboard/index.html'}")

    # -- Telemetry --
    print(f"  [OK] Telemetry           — enabled (logs/events.jsonl)")

    print("=" * 64)

    has_llm = checks.get("ollama", False) or checks.get("anthropic", False)
    if not has_llm:
        print("\n  WARNING: No LLM provider available.")
        print("  The server will start, but the pipeline will fail at runtime.")
        print("  Either run `ollama serve` or set ANTHROPIC_API_KEY.\n")
    else:
        print("\n  All critical checks passed!\n")

    return checks


# ---------------------------------------------------------------------------
# HTTP server for dashboard static files
# ---------------------------------------------------------------------------

async def start_http_server(port: int = 8080) -> web.AppRunner:
    """Start an aiohttp server to serve the dashboard UI."""
    app = web.Application()

    async def index(request: web.Request) -> web.FileResponse:
        return web.FileResponse(DASHBOARD_DIR / "index.html")

    app.router.add_get("/", index)
    app.router.add_static("/", DASHBOARD_DIR)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    return runner


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Start the HTTP server and WebSocket gateway concurrently."""
    from src.pipeline.orchestrator import ConversationOrchestrator
    from src.gateway.websocket_server import VoiceGateway

    print("\n" + "=" * 64)
    print("  Multilingual Indian Voice Agent — Live Demo")
    print("=" * 64 + "\n")

    # 1. Check prerequisites
    checks = check_prerequisites()

    # 2. Enable telemetry so the dashboard charts work
    os.environ["TELEMETRY_ENABLED"] = "true"
    logger.info("Telemetry enabled. Dashboard will show live metrics.")

    # 3. Create the Orchestrator
    logger.info("Initializing Conversation Orchestrator...")
    orchestrator = ConversationOrchestrator(
        stt_provider=settings.stt_provider,
        tts_provider=settings.tts_provider,
        llm_provider=settings.llm_provider,
    )
    # Pre-create a session so the first turn doesn't need setup.
    orchestrator.create_session()

    # 4. Start the WebSocket Voice Gateway
    logger.info(f"Starting WebSocket Voice Gateway on port {WS_PORT}...")
    gateway = VoiceGateway(host="0.0.0.0", port=WS_PORT, pipeline=orchestrator)
    ws_server = await gateway.start_in_thread()

    # 5. Start the HTTP server
    logger.info(f"Starting HTTP server on port {HTTP_PORT}...")
    http_runner = await start_http_server(port=HTTP_PORT)

    # 6. Ready banner
    print("\n" + "=" * 64)
    print("  READY!")
    print("=" * 64)
    print(f"  Dashboard : http://localhost:{HTTP_PORT}")
    print(f"  WebSocket : ws://localhost:{WS_PORT}")
    print(f"  LLM       : {settings.llm_provider} ({settings.llm_model})")
    print(f"  STT       : {settings.stt_provider} ({settings.whisper_model})")
    print(f"  TTS       : {settings.tts_provider}")
    print(f"  Telemetry : enabled -> logs/events.jsonl")
    print("=" * 64)
    print(f"\n  Open http://localhost:{HTTP_PORT} in your browser")
    print("  Press-and-hold the MIC button to talk.")
    print("\n  Press Ctrl+C to stop.\n")

    # Try to open the browser automatically
    try:
        webbrowser.open(f"http://localhost:{HTTP_PORT}")
    except Exception:
        pass

    # 7. Run forever until interrupted
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        print("\nShutting down...")
        await http_runner.cleanup()
        ws_server.close()
        await ws_server.wait_closed()
        print("Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
