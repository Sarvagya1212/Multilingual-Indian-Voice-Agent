"""Tests for the free local LLM (Ollama) and free TTS (gTTS) providers.

These providers are designed to run without paid API keys. Ollama tests
gracefully skip when no Ollama server is running; gTTS tests need network
access to Google's translate endpoint.
"""
from __future__ import annotations

import asyncio
import os
import socket

import pytest

from src.llm.base import Message, MessageRole
from src.llm.providers import (
    LLM_PROVIDERS,
    OllamaLLMProvider,
    get_llm_provider,
)
from src.tts.base import TTSResult
from src.tts.providers import (
    TTS_PROVIDERS,
    GttsTTSProvider,
    get_tts_provider,
)


# ---------------------------------------------------------------- helpers


def _ollama_server_reachable(host: str = "127.0.0.1", port: int = 11434) -> bool:
    """Return True if an Ollama server is listening on the default port."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _network_reachable(host: str = "8.8.8.8", port: int = 53, timeout: float = 1.0) -> bool:
    """Return True if the host has general internet connectivity."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


OLLAMA_AVAILABLE = _ollama_server_reachable()
NETWORK_AVAILABLE = _network_reachable()

skip_no_ollama = pytest.mark.skipif(
    not OLLAMA_AVAILABLE,
    reason="No Ollama server running on localhost:11434",
)
skip_no_network = pytest.mark.skipif(
    not NETWORK_AVAILABLE,
    reason="No network access for gTTS",
)


# --------------------------------------------------------------- LLM registry


class TestOllamaRegistry:
    def test_ollama_in_registry(self):
        assert "ollama" in LLM_PROVIDERS
        assert LLM_PROVIDERS["ollama"] is OllamaLLMProvider

    def test_llama_alias(self):
        assert LLM_PROVIDERS.get("llama") is OllamaLLMProvider

    def test_get_provider_by_name(self):
        provider = get_llm_provider("ollama")
        assert isinstance(provider, OllamaLLMProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider("not_a_real_provider")


class TestOllamaProviderInit:
    def test_default_model(self):
        p = OllamaLLMProvider()
        assert p.model == "llama3.1:latest"
        assert p.base_url == "http://localhost:11434"

    def test_custom_model(self):
        p = OllamaLLMProvider(model="mistral:7b")
        assert p.model == "mistral:7b"

    def test_custom_base_url_strips_slash(self):
        p = OllamaLLMProvider(base_url="http://example.com:11434/")
        assert p.base_url == "http://example.com:11434"

    def test_name_includes_model(self):
        p = OllamaLLMProvider(model="mistral:7b")
        assert p.name == "ollama-mistral:7b"

    def test_latency_starts_at_zero(self):
        p = OllamaLLMProvider()
        assert p.latency_ms == 0.0


# --------------------------------------------------------- LLM message shape


class TestOllamaMessageConversion:
    def test_simple_messages(self):
        p = OllamaLLMProvider()
        msgs = [
            Message(role=MessageRole.SYSTEM, content="Be brief."),
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi"),
        ]
        out = p._convert_messages(msgs)
        assert out[0] == {"role": "system", "content": "Be brief."}
        assert out[1] == {"role": "user", "content": "Hello"}
        assert out[2] == {"role": "assistant", "content": "Hi"}

    def test_tool_result_message(self):
        p = OllamaLLMProvider()
        msgs = [
            Message(
                role=MessageRole.TOOL_RESULT,
                content="Course found",
                tool_call_id="call_1",
            ),
        ]
        out = p._convert_messages(msgs)
        assert out[0]["role"] == "tool"
        assert out[0]["tool_call_id"] == "call_1"


# ----------------------------------------------- LLM live integration tests


@skip_no_ollama
@pytest.mark.asyncio
class TestOllamaLive:
    """Tests that hit a running Ollama server. Skip if not available."""

    async def test_chat_returns_content(self):
        p = OllamaLLMProvider()
        msgs = [
            Message(role=MessageRole.SYSTEM, content="Reply with one word: PONG."),
            Message(role=MessageRole.USER, content="PING"),
        ]
        resp = await p.chat(messages=msgs, max_tokens=16)
        assert isinstance(resp.content, str)
        assert len(resp.content) > 0
        assert resp.tool_calls == []
        assert p.latency_ms > 0

    async def test_stream_yields_chunks(self):
        p = OllamaLLMProvider()
        msgs = [
            Message(role=MessageRole.USER, content="Count to 3."),
        ]
        chunks: list[str] = []
        async for chunk in p.chat_stream(messages=msgs, max_tokens=20):
            chunks.append(chunk)
        joined = "".join(chunks)
        assert len(joined) > 0

    async def test_language_detection_english(self):
        p = OllamaLLMProvider()
        assert await p.detect_language("Hello, I want to join JEE course") == "en"

    async def test_language_detection_hindi_devanagari(self):
        p = OllamaLLMProvider()
        assert await p.detect_language("नमस्ते, JEE कोर्स के बारे में बताओ") == "hi"

    async def test_language_detection_hinglish(self):
        p = OllamaLLMProvider()
        lang = await p.detect_language("मैं JEE की preparation कर रहा हूं")
        assert lang in ("hi", "hinglish")

    async def test_language_detection_empty(self):
        p = OllamaLLMProvider()
        assert await p.detect_language("") == "en"


# --------------------------------------------------------------- TTS registry


class TestGttsRegistry:
    def test_gtts_in_registry(self):
        assert "gtts" in TTS_PROVIDERS
        assert TTS_PROVIDERS["gtts"] is GttsTTSProvider

    def test_google_tts_alias(self):
        assert TTS_PROVIDERS.get("google_tts") is GttsTTSProvider

    def test_get_provider_by_name(self):
        provider = get_tts_provider("gtts")
        assert isinstance(provider, GttsTTSProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            get_tts_provider("not_a_real_provider")


class TestGttsProviderInit:
    def test_defaults(self):
        p = GttsTTSProvider()
        assert p.language == "en"
        assert p.slow is False
        assert p.name == "gtts-en"

    def test_custom_language(self):
        p = GttsTTSProvider(language="hi")
        assert p.language == "hi"
        assert p.name == "gtts-hi"

    def test_slow_mode(self):
        p = GttsTTSProvider(slow=True)
        assert p.slow is True

    def test_available_voices(self):
        p = GttsTTSProvider()
        voices = p.available_voices
        assert "en" in voices
        assert "hi" in voices


# --------------------------------------------------------- TTS language map


class TestGttsLanguageMapping:
    def test_english_passthrough(self):
        p = GttsTTSProvider()
        assert p._language_for("en") == "en"

    def test_hindi_passthrough(self):
        p = GttsTTSProvider()
        assert p._language_for("hi") == "hi"

    def test_hinglish_maps_to_english(self):
        p = GttsTTSProvider()
        # gTTS has no Hinglish code; pick the dominant script
        assert p._language_for("hinglish") == "en"

    def test_none_falls_back_to_default(self):
        p = GttsTTSProvider(language="en")
        assert p._language_for(None) == "en"

    def test_unknown_code_passes_through(self):
        p = GttsTTSProvider()
        assert p._language_for("es") == "es"


# ----------------------------------------------- TTS live integration tests


@skip_no_network
@pytest.mark.asyncio
class TestGttsLive:
    """Tests that actually hit the Google TTS endpoint. Skip if offline."""

    async def test_synthesize_english(self):
        p = GttsTTSProvider()
        result = await p.synthesize("Hello, world.", language="en")
        assert isinstance(result, TTSResult)
        assert result.format == "mp3"
        assert len(result.audio) > 1000  # MP3 of one word is at least a few KB
        # MP3 frame sync bytes
        assert result.audio[:2] == b"\xff\xf3" or result.audio[:2] == b"\xff\xfb"

    async def test_synthesize_hindi(self):
        p = GttsTTSProvider()
        result = await p.synthesize("नमस्ते दोस्तों", language="hi")
        assert result.format == "mp3"
        assert len(result.audio) > 1000
        assert p.latency_ms > 0

    async def test_synthesize_hinglish_routes_to_english(self):
        p = GttsTTSProvider()
        result = await p.synthesize("JEE ke liye preparation", language="hinglish")
        assert result.format == "mp3"
        assert result.voice == "en"  # we routed it

    async def test_synthesize_stream_yields_bytes(self):
        p = GttsTTSProvider()
        total = bytearray()
        async for chunk in p.synthesize_stream("Hello, world.", language="en"):
            total.extend(chunk)
        assert len(total) > 1000
        assert bytes(total[:2]) == b"\xff\xf3" or bytes(total[:2]) == b"\xff\xfb"
