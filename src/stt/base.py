"""STT (Speech-to-Text) abstraction layer."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional
import numpy as np


@dataclass
class STTResult:
    """Result from speech-to-text transcription."""
    text: str
    language: str
    confidence: float = 0.0
    duration: float = 0.0  # Audio duration in seconds
    words: list = field(default_factory=list)  # Optional word-level timestamps


class STTProvider(ABC):
    """Abstract base class for STT providers.

    All STT implementations must inherit from this class and implement
    the abstract methods. This allows for easy swapping between providers.
    """

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        language: Optional[str] = None,
    ) -> STTResult:
        """Transcribe audio bytes to text.

        Args:
            audio: Audio data as bytes (wav/pcm format)
            language: Optional language hint (e.g., 'en', 'hi', 'hinglish')

        Returns:
            STTResult with transcribed text, detected language, and confidence
        """
        pass

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream transcription yielding partial results.

        Args:
            audio_stream: Async generator of audio chunks
            language: Optional language hint

        Yields:
            Partial transcription results as they become available
        """
        pass

    @abstractmethod
    async def detect_language(self, audio: bytes) -> str:
        """Detect the primary language of the audio.

        Args:
            audio: Audio data as bytes

        Returns:
            Language code (e.g., 'en', 'hi', 'hinglish')
        """
        pass

    @property
    @abstractmethod
    def latency_ms(self) -> float:
        """Average transcription latency in milliseconds."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and identification."""
        pass
