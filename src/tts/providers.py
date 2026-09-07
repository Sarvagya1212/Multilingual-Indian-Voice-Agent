"""TTS provider implementations."""
import time
import io
import struct
import wave
from typing import AsyncGenerator, Optional

from src.tts.base import TTSProvider, TTSResult
from src.tts.config import TTSConfig, get_voice_for_language
from src.tts.normalizer import TextNormalizer, normalizer
from src.logger import setup_logger

logger = setup_logger(__name__)


class OpenAITTSProvider(TTSProvider):
    """OpenAI TTS provider.

    Supports streaming synthesis for real-time playback.
    """

    def __init__(
        self,
        voice: str = "alloy",
        model: str = "tts-1",
        api_key: Optional[str] = None,
        config: TTSConfig = None,
    ):
        """Initialize OpenAI TTS provider.

        Args:
            voice: Voice name (alloy, echo, fable, nova, shimmer)
            model: Model name (tts-1 or tts-1-hd)
            api_key: OpenAI API key (uses env var if not provided)
            config: Optional TTSConfig
        """
        self.config = config or TTSConfig()
        self.voice = voice or self.config.voice
        self.model = model or self.config.model
        self._latency_ms = 0.0
        self._client = None
        self._normalizer = TextNormalizer()

        logger.info(f"OpenAITTSProvider initialized: voice={self.voice}, model={self.model}")

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._get_api_key())
            except ImportError:
                logger.error("openai package not installed. Run: pip install openai")
                raise
        return self._client

    def _get_api_key(self) -> Optional[str]:
        """Get API key from env or parameter."""
        from src.config import settings
        return settings.openai_api_key

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> TTSResult:
        """Synthesize text to speech, return full audio.

        Args:
            text: Text to synthesize (will be normalized)
            voice: Override voice
            language: Language hint (for voice selection)

        Returns:
            TTSResult with audio bytes
        """
        start = time.time()

        # Normalize text
        normalized = self._normalizer.normalize(text)
        if normalized.substitutions:
            logger.debug(f"Normalized: {normalized.substitutions}")

        # Select voice
        voice = voice or self.config.voice
        if language and language in ("hi", "hinglish"):
            # Use slightly slower voice for Hindi
            pass

        # Generate audio
        try:
            response = self.client.audio.speech.create(
                model=self.model,
                voice=voice,
                input=normalized.normalized,
                response_format=self.config.response_format,
                speed=self.config.speed,
            )
            audio = response.content
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise

        self._latency_ms = (time.time() - start) * 1000

        # Estimate duration (rough: ~150 chars/min at normal speed)
        duration = len(normalized.normalized) / 150 * 60 / self.config.speed

        logger.info(f"TTS: '{normalized.normalized[:50]}...' ({self._latency_ms:.0f}ms, {duration:.1f}s)")

        return TTSResult(
            audio=audio,
            duration=duration,
            format=self.config.response_format,
            voice=voice,
        )

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks as they are generated.

        Args:
            text: Text to synthesize
            voice: Override voice
            language: Language hint

        Yields:
            Audio chunks
        """
        # Normalize text
        normalized = self._normalizer.normalize(text)

        voice = voice or self.config.voice

        # Stream from OpenAI
        try:
            with self.client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=voice,
                input=normalized.normalized,
                response_format=self.config.response_format,
                speed=self.config.speed,
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=self.config.chunk_size):
                    yield chunk
        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")
            raise

    @property
    def latency_ms(self) -> float:
        return self._latency_ms

    @property
    def name(self) -> str:
        return f"openai-tts-{self.model}"

    @property
    def available_voices(self) -> list:
        return ["alloy", "echo", "fable", "nova", "shimmer"]


class GttsTTSProvider(TTSProvider):
    """Google Translate TTS (gTTS) provider — free, no API key.

    Uses the public Google Translate text-to-speech endpoint. No authentication
    required, but quality is lower than paid TTS and network access is required
    at synthesis time.

    Notes:
    - Streams the full audio to memory; not chunk-streaming
    - Supports many languages including Hindi (`hi`) and English (`en`)
    - Audio format is always MP3
    """

    def __init__(
        self,
        language: str = "en",
        slow: bool = False,
        config: TTSConfig = None,
    ):
        """Initialize gTTS provider.

        Args:
            language: Default TLD language code (en, hi, etc.)
            slow: If True, speak more slowly
            config: Optional TTSConfig (used for response_format)
        """
        self.config = config or TTSConfig()
        self.language = language or "en"
        self.slow = slow
        self._latency_ms = 0.0
        self._normalizer = TextNormalizer()

        logger.info(f"GttsTTSProvider initialized: language={self.language}")

    def _language_for(self, language: Optional[str]) -> str:
        """Map our language codes to gTTS language codes."""
        if not language:
            return self.language
        # gTTS doesn't have a Hinglish code; pick the dominant script
        if language == "hinglish":
            # Hinglish is mostly Latin script → use English, the listener
            # will still hear the Hindi words pronounced by gTTS
            return "en"
        if language in ("en", "hi"):
            return language
        # Default: try the code directly
        return language

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> TTSResult:
        """Synthesize text to speech using gTTS.

        Args:
            text: Text to synthesize (will be normalized)
            voice: Ignored (gTTS doesn't expose voice selection)
            language: Language code (en, hi, hinglish)

        Returns:
            TTSResult with MP3 audio bytes
        """
        start = time.time()

        normalized = self._normalizer.normalize(text)
        lang = self._language_for(language)

        # gTTS is sync, so we run it in a thread to keep the pipeline async
        import asyncio
        try:
            from gtts import gTTS
            loop = asyncio.get_event_loop()
            audio_bytes = await loop.run_in_executor(
                None, self._synth_sync, normalized.normalized, lang
            )
        except ImportError:
            logger.error("gTTS not installed. Run: pip install gtts")
            raise
        except Exception as e:
            logger.error(f"gTTS synthesis failed: {e}")
            raise

        self._latency_ms = (time.time() - start) * 1000
        duration = len(normalized.normalized) / 150 * 60  # rough estimate

        logger.info(
            f"gTTS: '{normalized.normalized[:50]}...' "
            f"lang={lang} ({self._latency_ms:.0f}ms, {duration:.1f}s)"
        )

        return TTSResult(
            audio=audio_bytes,
            duration=duration,
            format="mp3",
            voice=lang,
        )

    def _synth_sync(self, text: str, lang: str) -> bytes:
        """Synchronous synthesis used by the executor."""
        import io
        from gtts import gTTS
        buf = io.BytesIO()
        tts = gTTS(text=text, lang=lang, slow=self.slow)
        tts.write_to_fp(buf)
        return buf.getvalue()

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks.

        gTTS doesn't truly stream — it returns the full MP3. We yield it in
        chunks so the interface matches the other providers.
        """
        result = await self.synthesize(text, voice=voice, language=language)
        chunk_size = self.config.chunk_size
        for i in range(0, len(result.audio), chunk_size):
            yield result.audio[i : i + chunk_size]

    @property
    def latency_ms(self) -> float:
        return self._latency_ms

    @property
    def name(self) -> str:
        return f"gtts-{self.language}"

    @property
    def available_voices(self) -> list:
        """gTTS doesn't have voices; we expose the language list instead."""
        return ["en", "hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa"]


# Provider registry
TTS_PROVIDERS = {
    "openai": OpenAITTSProvider,
    "openai_tts": OpenAITTSProvider,
    "gtts": GttsTTSProvider,
    "google_tts": GttsTTSProvider,
}


def get_tts_provider(provider_name: Optional[str] = None, **kwargs) -> TTSProvider:
    """Get a TTS provider by name.

    Args:
        provider_name: Provider name (default: from config)
        **kwargs: Provider-specific arguments

    Returns:
        TTSProvider instance
    """
    from src.config import settings

    name = provider_name or settings.tts_provider
    provider_class = TTS_PROVIDERS.get(name)

    if not provider_class:
        raise ValueError(
            f"Unknown TTS provider: {name}. "
            f"Available: {list(TTS_PROVIDERS.keys())}"
        )

    return provider_class(**kwargs)
