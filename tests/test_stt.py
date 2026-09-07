"""Tests for STT module."""
import pytest
import numpy as np
import wave
import io


# Create a simple test WAV file
def create_test_wav(duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Create a simple sine wave WAV file for testing."""
    import math

    frequency = 440  # A4 note
    samples = int(duration * sample_rate)
    audio_data = []

    for i in range(samples):
        value = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * i / sample_rate))
        audio_data.append(value)

    # Create WAV
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(np.array(audio_data, dtype=np.int16)))
    return buffer.getvalue()


class TestSTTConfig:
    """Tests for STTConfig."""

    def test_default_config(self):
        from src.stt.config import STTConfig
        config = STTConfig()
        assert config.provider == "openai_whisper"
        assert config.model_name == "base"
        assert config.device == "cpu"

    def test_custom_config(self):
        from src.stt.config import STTConfig
        config = STTConfig(model_name="small", temperature=0.5)
        assert config.model_name == "small"
        assert config.temperature == 0.5


class TestSTTResult:
    """Tests for STTResult."""

    def test_result_creation(self):
        from src.stt import STTResult
        result = STTResult(
            text="Hello world",
            language="en",
            confidence=0.95,
            duration=1.5,
        )
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.confidence == 0.95
        assert result.duration == 1.5
        assert result.words == []


class TestWhisperProvider:
    """Tests for WhisperSTTProvider."""

    def test_provider_init(self):
        from src.stt import WhisperSTTProvider
        provider = WhisperSTTProvider(model_name="tiny", device="cpu")
        assert provider.model_name == "tiny"
        assert provider.device == "cpu"
        assert provider.name == "whisper-tiny"

    def test_resample(self):
        from src.stt import WhisperSTTProvider
        provider = WhisperSTTProvider()

        # 44.1kHz audio resampled to 16kHz
        audio_44k = np.random.randn(44100).astype(np.float32)
        resampled = provider._resample(audio_44k, 44100, 16000)

        # Should be roughly 16/44.1 times the original length
        expected_length = int(len(audio_44k) * 16000 / 44100)
        assert abs(len(resampled) - expected_length) < 10

    def test_resample_same_rate(self):
        from src.stt import WhisperSTTProvider
        provider = WhisperSTTProvider()

        audio = np.random.randn(16000).astype(np.float32)
        resampled = provider._resample(audio, 16000, 16000)
        assert len(resampled) == len(audio)

    def test_normalize_language_code(self):
        from src.stt import WhisperSTTProvider
        provider = WhisperSTTProvider()

        assert provider._normalize_language_code("en") == "en"
        assert provider._normalize_language_code("hi") == "hi"
        assert provider._normalize_language_code("hi-en") == "hinglish"
        assert provider._normalize_language_code("unknown") == "unknown"

    def test_bytes_to_audio_wav(self):
        from src.stt import WhisperSTTProvider
        provider = WhisperSTTProvider()

        # Create test WAV
        wav_bytes = create_test_wav(duration=0.5)
        audio = provider._bytes_to_audio(wav_bytes)

        assert audio is not None
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        # Should be ~0.5 seconds at 16kHz
        assert len(audio) > 7000
        assert len(audio) < 9000

    def test_bytes_to_audio_empty(self):
        from src.stt import WhisperSTTProvider
        provider = WhisperSTTProvider()

        audio = provider._bytes_to_audio(b"")
        assert audio is None

        audio = provider._bytes_to_audio(b"short")
        assert audio is None


class TestProviderRegistry:
    """Tests for provider registry."""

    def test_get_stt_provider_default(self):
        from src.stt import get_stt_provider, WhisperSTTProvider
        provider = get_stt_provider()
        assert isinstance(provider, WhisperSTTProvider)

    def test_get_stt_provider_by_name(self):
        from src.stt import get_stt_provider, WhisperSTTProvider
        provider = get_stt_provider("openai_whisper")
        assert isinstance(provider, WhisperSTTProvider)

    def test_get_stt_provider_unknown(self):
        from src.stt import get_stt_provider
        with pytest.raises(ValueError, match="Unknown STT provider"):
            get_stt_provider("unknown_provider")


@pytest.mark.asyncio
class TestTranscribe:
    """Async tests for transcription."""

    async def test_transcribe_empty_audio(self):
        from src.stt import WhisperSTTProvider
        provider = WhisperSTTProvider(model_name="tiny")

        result = await provider.transcribe(b"", language="en")
        assert result.text == ""
        assert result.language == "en"
        assert result.confidence == 0.0

    async def test_transcribe_wav(self):
        from src.stt import WhisperSTTProvider
        provider = WhisperSTTProvider(model_name="tiny")

        # Create short silent WAV
        wav_bytes = create_test_wav(duration=1.0)
        result = await provider.transcribe(wav_bytes)

        # Should return some result (possibly empty for silent audio)
        assert isinstance(result, object)
        assert hasattr(result, "text")
        assert hasattr(result, "language")


@pytest.mark.asyncio
class TestLanguageDetection:
    """Tests for language detection."""

    async def test_detect_language_short_audio(self):
        from src.stt import WhisperSTTProvider
        provider = WhisperSTTProvider(model_name="tiny")

        # Very short audio should return default
        lang = await provider.detect_language(b"")
        assert lang == "en"

    async def test_detect_language_wav(self):
        from src.stt import WhisperSTTProvider
        provider = WhisperSTTProvider(model_name="tiny")

        wav_bytes = create_test_wav(duration=2.0)
        lang = await provider.detect_language(wav_bytes)

        # Should return a language code
        assert isinstance(lang, str)
        assert lang in ["en", "hi", "hinglish", "unknown"]


# Run tests with: pytest tests/test_stt.py -v
