"""WebSocket voice gateway for real-time audio communication.

Handles WebSocket connections, audio streaming, VAD-based turn detection,
and integration with the ConversationOrchestrator pipeline.
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Callable, Awaitable

import websockets
from websockets.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from src.logger import setup_logger
from src.pipeline.types import AgentState
from src.gateway.audio_utils import bytes_to_audio, audio_to_bytes
from src.gateway.vad import SimpleEnergyVAD, VADConfig

logger = setup_logger(__name__)


class GatewaySession:
    """Per-connection session state for the voice gateway."""

    def __init__(self, session_id: str, gateway: "VoiceGateway"):
        self.id = session_id
        self.gateway = gateway
        self.created_at = datetime.now()
        self.state = AgentState.IDLE
        self.audio_buffer: list[bytes] = []
        self.vad = SimpleEnergyVAD()
        self.is_streaming = False

    def add_audio(self, chunk: bytes) -> None:
        """Append an audio chunk to the session buffer."""
        self.audio_buffer.append(chunk)

    def get_buffer_bytes(self) -> bytes:
        """Return all buffered audio as a single bytes object."""
        return b"".join(self.audio_buffer)

    def clear_buffer(self) -> None:
        """Clear the audio buffer."""
        self.audio_buffer.clear()

    def reset(self) -> None:
        """Reset session for a new turn."""
        self.audio_buffer.clear()
        self.state = AgentState.IDLE
        self.is_streaming = False
        self.vad.reset()


# ---------------------------------------------------------------------------
# Gateway message protocol
# ---------------------------------------------------------------------------
#
# Client -> Server binary frame:
#   Raw PCM audio bytes (16-bit, mono, 16kHz)
#
# Client -> Server JSON:
#   { "type": "start" }              — Begin streaming
#   { "type": "stop" }               — End streaming (trigger pipeline)
#   { "type": "ping" }               — Health check
#   { "type": "interrupt" }          — User barge-in
#
# Server -> Client JSON:
#   { "type": "status", "state": "..." }           — State change
#   { "type": "turn_started" }                      — Pipeline triggered
#   { "type": "turn_complete", "language": "..." }  — Response ready
#   { "type": "pong" }                             — Ping response
#   { "type": "error", "message": "..." }          — Error
#
# Server -> Client binary frame:
#   TTS audio bytes (WAV)
# ---------------------------------------------------------------------------


class VoiceGateway:
    """WebSocket voice gateway for real-time communication.

    Manages WebSocket connections and orchestrates the audio pipeline:
    Audio In → VAD → STT → LLM → TTS → Audio Out

    Usage:
        gateway = VoiceGateway(pipeline=orchestrator)
        asyncio.run(gateway.start())
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        pipeline=None,
        vad_config: VADConfig | None = None,
    ):
        """Initialize the gateway.

        Args:
            host: Host to bind.
            port: Port to bind.
            pipeline: ConversationOrchestrator instance for processing turns.
            vad_config: VAD configuration.
        """
        self.host = host
        self.port = port
        self.pipeline = pipeline
        self.vad_config = vad_config or VADConfig()
        self.sessions: Dict[str, GatewaySession] = {}
        self._running = False

    async def handle_websocket(self, websocket: ServerConnection) -> None:
        """Handle a single WebSocket connection.

        Args:
            websocket: The WebSocket connection.
        """
        session_id = str(uuid.uuid4())
        session = GatewaySession(session_id, self)
        self.sessions[session_id] = session

        logger.info(f"[{session_id}] New WebSocket connection from {websocket.remote_address}")

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await self._handle_audio(websocket, session, message)
                else:
                    data = json.loads(message)
                    await self._handle_json(websocket, session, data)

        except ConnectionClosed:
            logger.info(f"[{session_id}] Connection closed")
        except Exception as e:
            logger.error(f"[{session_id}] Error: {e}", exc_info=True)
            try:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e),
                }))
            except Exception:
                pass
        finally:
            self.sessions.pop(session_id, None)

    async def _handle_audio(
        self,
        websocket: ServerConnection,
        session: GatewaySession,
        audio_chunk: bytes,
    ) -> None:
        """Process an incoming audio chunk.

        If streaming is active, buffer the audio. If not, apply VAD.

        Args:
            websocket: WebSocket for sending acknowledgements.
            session: Per-connection session state.
            audio_chunk: Raw PCM audio bytes.
        """
        if not session.is_streaming:
            # VAD mode: check for speech onset
            audio_np = bytes_to_audio(audio_chunk, self.vad_config.sample_rate)
            if session.vad.is_speech(audio_np):
                session.is_streaming = True
                session.audio_buffer.append(audio_chunk)
                logger.info(f"[{session.id}] Speech detected, buffering audio")
                await websocket.send(json.dumps({
                    "type": "speech_start",
                    "session_id": session.id,
                }))
        else:
            # In streaming mode, buffer everything
            session.audio_buffer.append(audio_chunk)

            # Check for silence (end of utterance)
            audio_np = bytes_to_audio(audio_chunk, self.vad_config.sample_rate)
            if not session.vad.is_speech(audio_np):
                # End of speech detected via silence
                session.is_streaming = False
                logger.info(f"[{session.id}] Silence detected, {len(session.get_buffer_bytes())} bytes buffered")
                await websocket.send(json.dumps({
                    "type": "speech_end",
                    "session_id": session.id,
                    "buffered_bytes": len(session.get_buffer_bytes()),
                }))

    async def _handle_json(
        self,
        websocket: ServerConnection,
        session: GatewaySession,
        data: dict,
    ) -> None:
        """Handle a JSON control message.

        Args:
            websocket: WebSocket for sending responses.
            session: Per-connection session state.
            data: Parsed JSON message.
        """
        msg_type = data.get("type")

        if msg_type == "ping":
            await websocket.send(json.dumps({"type": "pong"}))

        elif msg_type == "start":
            # Begin streaming mode (explicit start)
            session.is_streaming = True
            session.clear_buffer()
            session.state = AgentState.LISTENING
            await websocket.send(json.dumps({
                "type": "status",
                "state": session.state.value,
                "session_id": session.id,
            }))
            logger.info(f"[{session.id}] Streaming started")

        elif msg_type == "stop":
            # Explicit stop — process whatever is buffered
            session.is_streaming = False
            await self._process_turn(websocket, session)

        elif msg_type == "interrupt":
            # User barge-in
            if self.pipeline:
                self.pipeline.interrupt()
            session.reset()
            await websocket.send(json.dumps({
                "type": "status",
                "state": AgentState.INTERRUPTED.value,
                "session_id": session.id,
            }))
            logger.info(f"[{session.id}] User interruption handled")

        elif msg_type == "status":
            await websocket.send(json.dumps({
                "type": "status",
                "state": session.state.value,
                "session_id": session.id,
            }))

        else:
            logger.warning(f"[{session.id}] Unknown message type: {msg_type}")

    async def _process_turn(
        self,
        websocket: ServerConnection,
        session: GatewaySession,
    ) -> None:
        """Run the full pipeline on buffered audio.

        Args:
            websocket: WebSocket for sending results.
            session: Per-connection session state.
        """
        audio_bytes = session.get_buffer_bytes()
        if not audio_bytes:
            await websocket.send(json.dumps({
                "type": "turn_complete",
                "language": "en",
                "skipped": True,
            }))
            return

        if self.pipeline is None:
            logger.error(f"[{session.id}] No pipeline configured")
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Pipeline not configured",
            }))
            return

        session.state = AgentState.TRANSCRIBING
        await websocket.send(json.dumps({
            "type": "status",
            "state": session.state.value,
            "session_id": session.id,
        }))

        try:
            logger.info(f"[{session.id}] Processing turn ({len(audio_bytes)} bytes)")
            session.clear_buffer()

            # Run pipeline
            result = await self.pipeline.process_turn(audio_bytes)

            session.state = AgentState.SPEAKING
            await websocket.send(json.dumps({
                "type": "turn_started",
                "session_id": session.id,
            }))

            # Send audio response
            if result.audio:
                await websocket.send(result.audio)
                logger.info(
                    f"[{session.id}] Sent {len(result.audio)} bytes "
                    f"({result.duration:.1f}s audio)"
                )

            await websocket.send(json.dumps({
                "type": "turn_complete",
                "session_id": session.id,
                "duration": result.duration,
            }))

            session.state = AgentState.IDLE

        except Exception as e:
            logger.error(f"[{session.id}] Pipeline error: {e}", exc_info=True)
            session.state = AgentState.ERROR
            await websocket.send(json.dumps({
                "type": "error",
                "message": str(e),
            }))

    async def start(self) -> None:
        """Start the WebSocket server (runs forever)."""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        self._running = True
        async with websockets.serve(self.handle_websocket, self.host, self.port):
            await asyncio.Future()  # Run forever

    async def start_in_thread(self) -> websockets.Server:
        """Start the server and return the Server for later shutdown.

        Useful for testing.
        """
        server = await websockets.serve(
            self.handle_websocket,
            self.host,
            self.port,
        )
        self._running = True
        logger.info(f"WebSocket server started on {self.host}:{self.port}")
        return server


async def main() -> None:
    """Standalone entry point for the gateway."""
    gateway = VoiceGateway()
    await gateway.start()


if __name__ == "__main__":
    asyncio.run(main())
