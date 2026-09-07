"""STT (Speech-to-Text) module.

Provides a provider-agnostic abstraction for speech recognition.

Example:
    from src.stt import get_stt_provider, STTResult

    # Get default provider
    stt = get_stt_provider()

    # Transcribe audio
    result = await stt.transcribe(audio_bytes, language="hi")
    print(f"Text: {result.text}, Language: {result.language}")

    # Detect language
    lang = await stt.detect_language(audio_bytes)
    print(f"Detected: {lang}")
"""

from src.stt.base import STTProvider, STTResult
from src.stt.providers import WhisperSTTProvider, get_stt_provider, STT_PROVIDERS
from src.stt.config import STTConfig, LANGUAGE_CODES

__all__ = [
    # Core classes
    "STTProvider",
    "STTResult",
    # Implementations
    "WhisperSTTProvider",
    "get_stt_provider",
    "STT_PROVIDERS",
    # Config
    "STTConfig",
    "LANGUAGE_CODES",
]
