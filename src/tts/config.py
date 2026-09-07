"""TTS configuration."""
from pydantic import BaseModel
from typing import Optional


class TTSConfig(BaseModel):
    """Configuration for TTS providers."""

    # Provider selection
    provider: str = "gtts"  # gtts (free), openai (paid)

    # OpenAI TTS options
    voice: str = "alloy"  # alloy, echo, fable, nova, shimmer
    model: str = "tts-1"  # tts-1 or tts-1-hd
    response_format: str = "mp3"  # mp3, opus, aac, flac
    speed: float = 1.0  # 0.25 to 4.0

    # Streaming options
    chunk_size: int = 1024  # Bytes per chunk for streaming

    class Config:
        extra = "ignore"


# Voice mappings for different languages/accent
VOICE_LANGUAGE_MAP = {
    "en": "alloy",  # English
    "hi": "alloy",  # Hindi (use alloy for neutral)
    "hinglish": "alloy",  # Hinglish
    "ta": "alloy",  # Tamil
    "te": "alloy",  # Telugu
    "bn": "alloy",  # Bengali
}


def get_voice_for_language(language: str) -> str:
    """Get appropriate voice for a language."""
    return VOICE_LANGUAGE_MAP.get(language, "alloy")
