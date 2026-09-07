"""Tests for VoiceGateway WebSocket server.

Uses a real local server bound to a random port to verify
JSON control message handling and message protocol.
"""
import asyncio
import json

import pytest
import websockets

from src.gateway.websocket_server import VoiceGateway, GatewaySession


@pytest.mark.asyncio
async def test_ping_pong():
    """Server should respond to ping with pong."""
    gateway = VoiceGateway(host="127.0.0.1", port=0)
    # `port=0` is interpreted by `serve` as random port; we need to bind manually
    # so we pick a high random port ourselves
    import random
    port = random.randint(20000, 30000)
    gateway.port = port

    server = await gateway.start_in_thread()
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"type": "ping"}))
            response = json.loads(await ws.recv())
            assert response["type"] == "pong"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_status():
    """Server should respond to status query with current state."""
    import random
    port = random.randint(20000, 30000)
    gateway = VoiceGateway(host="127.0.0.1", port=port)
    server = await gateway.start_in_thread()
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"type": "status"}))
            response = json.loads(await ws.recv())
            assert response["type"] == "status"
            assert response["state"] == "idle"
            assert "session_id" in response
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_start_message():
    """Server should switch to LISTENING on start."""
    import random
    port = random.randint(20000, 30000)
    gateway = VoiceGateway(host="127.0.0.1", port=port)
    server = await gateway.start_in_thread()
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"type": "start"}))
            response = json.loads(await ws.recv())
            assert response["type"] == "status"
            assert response["state"] == "listening"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_unknown_message_type():
    """Unknown message types should be tolerated, not crash."""
    import random
    port = random.randint(20000, 30000)
    gateway = VoiceGateway(host="127.0.0.1", port=port)
    server = await gateway.start_in_thread()
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"type": "frobnicate"}))
            # Server should still accept the next message
            await ws.send(json.dumps({"type": "ping"}))
            response = json.loads(await ws.recv())
            assert response["type"] == "pong"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_interrupt_without_pipeline():
    """Interrupt should work even if no pipeline is attached."""
    import random
    port = random.randint(20000, 30000)
    gateway = VoiceGateway(host="127.0.0.1", port=port)
    server = await gateway.start_in_thread()
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"type": "interrupt"}))
            response = json.loads(await ws.recv())
            assert response["type"] == "status"
            assert response["state"] == "interrupted"
    finally:
        server.close()
        await server.wait_closed()


class TestGatewaySession:
    def test_session_init(self):
        """Session should be in IDLE state with empty buffer."""
        gateway = VoiceGateway()
        session = GatewaySession("test-id", gateway)
        assert session.id == "test-id"
        assert session.state.value == "idle"
        assert session.audio_buffer == []
        assert session.is_streaming is False

    def test_add_and_get_audio(self):
        """Audio chunks should accumulate in the buffer."""
        gateway = VoiceGateway()
        session = GatewaySession("test-id", gateway)
        session.add_audio(b"\x00\x01")
        session.add_audio(b"\x02\x03")
        assert session.get_buffer_bytes() == b"\x00\x01\x02\x03"

    def test_clear_buffer(self):
        """clear_buffer should empty the audio buffer."""
        gateway = VoiceGateway()
        session = GatewaySession("test-id", gateway)
        session.add_audio(b"\x00\x01")
        session.clear_buffer()
        assert session.get_buffer_bytes() == b""

    def test_reset(self):
        """reset should clear buffer, reset state, and reset VAD."""
        gateway = VoiceGateway()
        session = GatewaySession("test-id", gateway)
        session.add_audio(b"\x00\x01")
        session.is_streaming = True
        session.reset()
        assert session.get_buffer_bytes() == b""
        assert session.state.value == "idle"
        assert session.is_streaming is False


# Run with: pytest tests/test_websocket_server.py -v
