"""TTS (Text-to-Speech) abstraction layer."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Optional


@dataclass
class TTSResult:
    """Result from text-to-speech synthesis."""
    audio: bytes
    duration: float  # Audio duration in seconds
    format: str = "mp3"  # Audio format (mp3, wav, pcm)
    sample_rate: int = 24000  # Default OpenAI TTS sample rate
    voice: str = "alloy"  # Voice used for synthesis


class TTSProvider(ABC):
    """Abstract base class for TTS providers.

    All TTS implementations must inherit from this class.
    """

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> TTSResult:
        """Synthesize text to speech, return full audio.

        Args:
            text: Text to synthesize (should be normalized)
            voice: Voice to use (provider-specific)
            language: Language hint

        Returns:
            TTSResult with audio bytes and metadata
        """
        pass

    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks as they are generated.

        Args:
            text: Text to synthesize
            voice: Voice to use
            language: Language hint

        Yields:
            Audio chunks (bytes) for streaming playback
        """
        pass

    @property
    @abstractmethod
    def latency_ms(self) -> float:
        """Average latency to first audio byte in milliseconds."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and identification."""
        pass

    @property
    @abstractmethod
    def available_voices(self) -> list:
        """List of available voice names for this provider."""
        pass
