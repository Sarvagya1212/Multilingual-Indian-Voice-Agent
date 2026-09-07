"""STT provider implementations."""
import time
import io
import wave
from typing import AsyncGenerator, Optional
import numpy as np

from src.stt.base import STTProvider, STTResult
from src.stt.config import STTConfig
from src.logger import setup_logger

logger = setup_logger(__name__)


class WhisperSTTProvider(STTProvider):
    """OpenAI Whisper STT provider (local inference).

    Supports Whisper models from tiny to large.
    Handles English, Hindi, and code-switching (Hinglish) well.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        config: STTConfig = None,
    ):
        """Initialize Whisper STT provider.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: cpu or cuda
            config: Optional STTConfig for fine-tuning
        """
        self.config = config or STTConfig()
        self.model_name = model_name
        self.device = device
        self._latency_ms = 0.0
        self._model = None

        # Lazy-load model on first use to avoid startup delay
        logger.info(f"WhisperSTTProvider initialized: model={model_name}, device={device}")

    def _load_model(self):
        """Lazy-load the Whisper model."""
        if self._model is None:
            try:
                import whisper
                logger.info(f"Loading Whisper model: {self.model_name}")
                self._model = whisper.load_model(self.model_name, device=self.device)
                logger.info("Whisper model loaded successfully")
            except ImportError:
                logger.error("openai-whisper not installed. Run: pip install openai-whisper")
                raise
        return self._model

    async def transcribe(
        self,
        audio: bytes,
        language: Optional[str] = None,
    ) -> STTResult:
        """Transcribe audio bytes to text.

        Args:
            audio: Audio data as bytes (WAV format, 16kHz mono recommended)
            language: Optional language code (en, hi, etc.)

        Returns:
            STTResult with text, language, confidence
        """
        start = time.time()

        # Convert bytes to numpy array
        audio_np = self._bytes_to_audio(audio)

        if audio_np is None or len(audio_np) == 0:
            return STTResult(
                text="",
                language=language or "unknown",
                confidence=0.0,
                duration=0.0,
            )

        # Load model (lazy)
        model = self._load_model()

        # Transcribe options
        options = {
            "temperature": self.config.temperature,
            "beam_size": self.config.beam_size,
            "best_of": self.config.best_of,
            "patience": self.config.patience,
        }

        if language:
            options["language"] = language
        else:
            options["language"] = None  # Auto-detect

        if self.config.initial_prompt:
            options["initial_prompt"] = self.config.initial_prompt

        # Run transcription
        result = model.transcribe(audio_np, **options)

        self._latency_ms = (time.time() - start) * 1000

        # Calculate audio duration
        duration = len(audio_np) / 16000  # Assuming 16kHz

        # Build result
        segments = result.get("segments", [])
        confidence = self._calculate_confidence(segments)

        logger.info(
            f"STT: '{result['text'][:50]}...' "
            f"lang={result.get('language', 'unknown')} "
            f"({self._latency_ms:.0f}ms)"
        )

        return STTResult(
            text=result["text"].strip(),
            language=result.get("language", language or "en"),
            confidence=confidence,
            duration=duration,
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream transcription yielding partial results.

        For Whisper, this buffers chunks and transcribes incrementally.
        """
        buffer = b""
        async for chunk in audio_stream:
            buffer += chunk

        if buffer:
            result = await self.transcribe(buffer, language=language)
            yield result.text

    async def detect_language(self, audio: bytes) -> str:
        """Detect the primary language of the audio.

        Uses Whisper's built-in language detection.
        """
        audio_np = self._bytes_to_audio(audio)
        if audio_np is None or len(audio_np) < 16000:  # At least 1s
            return "en"

        model = self._load_model()

        # Whisper detect_language returns (lang, probability)
        # Need to handle both 2-tuple and 3-tuple returns across versions
        try:
            result = model.detect_language(audio_np)
            if isinstance(result, tuple):
                if len(result) == 2:
                    lang_code, _ = result
                else:
                    lang_code = result[0]
                return self._normalize_language_code(lang_code)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")

        return "en"

    def _calculate_confidence(self, segments: list) -> float:
        """Calculate average confidence from segments."""
        if not segments:
            return 0.0

        confidences = []
        for seg in segments:
            if "no_speech_prob" in seg:
                # Lower no_speech_prob = higher confidence
                confidences.append(1.0 - seg["no_speech_prob"])
            elif "avg_logprob" in seg:
                # Convert logprob to confidence (rough approximation)
                confidences.append(min(1.0, max(0.0, 1.0 + seg["avg_logprob"])))

        return sum(confidences) / len(confidences) if confidences else 0.0

    def _normalize_language_code(self, code: str) -> str:
        """Normalize language code to our standard format."""
        code = code.lower()

        if code in ("en", "english"):
            return "en"
        elif code in ("hi", "hindi"):
            return "hi"
        elif code in ("hi-en", "hinglish", "hinglish"):
            return "hinglish"
        else:
            return code

    def _bytes_to_audio(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """Convert audio bytes to numpy array (16kHz float32)."""
        if not audio_bytes or len(audio_bytes) < 44:
            return None

        try:
            # Check if WAV format
            if audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE':
                with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                    sample_rate = wf.getframerate()
                    n_frames = wf.getnframes()
                    n_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()

                    raw = wf.readframes(n_frames)
                    audio_np = np.frombuffer(raw, dtype=np.int16)
            else:
                # Assume raw 16-bit PCM, 16kHz, mono
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

            # Convert to float32, normalized
            audio_np = audio_np.astype(np.float32) / 32768.0

            # Resample to 16kHz if needed (simple linear interpolation)
            if 'sample_rate' in dir() and sample_rate != 16000:
                audio_np = self._resample(audio_np, sample_rate, 16000)

            return audio_np

        except Exception as e:
            logger.error(f"Error converting audio bytes: {e}")
            return None

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear interpolation resampling."""
        if orig_sr == target_sr or len(audio) == 0:
            return audio

        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    @property
    def latency_ms(self) -> float:
        return self._latency_ms

    @property
    def name(self) -> str:
        return f"whisper-{self.model_name}"


# Provider registry
STT_PROVIDERS = {
    "openai_whisper": WhisperSTTProvider,
    "whisper": WhisperSTTProvider,
}


def get_stt_provider(provider_name: Optional[str] = None, **kwargs) -> STTProvider:
    """Get an STT provider by name.

    Args:
        provider_name: Provider name (default: from config)
        **kwargs: Provider-specific arguments

    Returns:
        STTProvider instance

    Raises:
        ValueError: If provider name is unknown
    """
    from src.config import settings

    name = provider_name or settings.stt_provider
    provider_class = STT_PROVIDERS.get(name)

    if not provider_class:
        raise ValueError(
            f"Unknown STT provider: {name}. "
            f"Available: {list(STT_PROVIDERS.keys())}"
        )

    # Default kwargs from settings
    if not kwargs:
        kwargs = {
            "model_name": settings.whisper_model,
            "device": settings.whisper_device,
        }

    return provider_class(**kwargs)
