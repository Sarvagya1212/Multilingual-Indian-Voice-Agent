"""TTS (Text-to-Speech) module.

Provides a provider-agnostic abstraction for speech synthesis.

Example:
    from src.tts import get_tts_provider, TTSResult

    # Get default provider
    tts = get_tts_provider()

    # Synthesize speech
    result = await tts.synthesize("Course fees are ₹25,000", language="hinglish")
    print(f"Audio duration: {result.duration}s")

    # Stream audio
    async for chunk in tts.synthesize_stream("Hello, how can I help?"):
        play_audio(chunk)
"""

from src.tts.base import TTSProvider, TTSResult
from src.tts.providers import (
    OpenAITTSProvider,
    GttsTTSProvider,
    get_tts_provider,
    TTS_PROVIDERS,
)
from src.tts.config import TTSConfig, VOICE_LANGUAGE_MAP, get_voice_for_language
from src.tts.normalizer import TextNormalizer, NormalizedText, normalizer

__all__ = [
    # Core classes
    "TTSProvider",
    "TTSResult",
    # Implementations
    "OpenAITTSProvider",
    "GttsTTSProvider",
    "get_tts_provider",
    "TTS_PROVIDERS",
    # Config
    "TTSConfig",
    "VOICE_LANGUAGE_MAP",
    "get_voice_for_language",
    # Normalization
    "TextNormalizer",
    "NormalizedText",
    "normalizer",
]
