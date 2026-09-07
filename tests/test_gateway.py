"""Tests for gateway module — audio_utils, VAD, and WebSocket server."""
import io
import wave
import struct

import pytest
import numpy as np

from src.gateway.audio_utils import (
    bytes_to_audio,
    audio_to_bytes,
    create_wav_header,
    resample_audio,
    mix_audio,
    compute_rms,
    trim_silence,
)
from src.gateway.vad import SimpleEnergyVAD, VADConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_wav_bytes(samples: list[int], sample_rate: int = 16000) -> bytes:
    """Create a valid WAV file from int16 samples."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(np.array(samples, dtype=np.int16)))
    return buffer.getvalue()


def make_pcm_bytes(samples: list[int]) -> bytes:
    """Create raw 16-bit PCM bytes."""
    return bytes(np.array(samples, dtype=np.int16))


# ---------------------------------------------------------------------------
# audio_utils — bytes_to_audio
# ---------------------------------------------------------------------------

class TestBytesToAudio:
    def test_wav_stereo_decodes(self):
        """Stereo WAV should decode (channel interleaved)."""
        # 1 second of 440 Hz sine wave, stereo (interleaved L,R)
        import math
        samples = [
            int(32767 * math.sin(2 * math.pi * 440 * i / 16000))
            for i in range(16000)
        ]
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(2)  # stereo
            wf.setsampwidth(2)
            wf.setframerate(16000)
            interleaved = []
            for s in samples:
                interleaved.extend([s, s])  # L + R identical
            wf.writeframes(bytes(np.array(interleaved, dtype=np.int16)))
        wav_bytes = buffer.getvalue()

        audio = bytes_to_audio(wav_bytes, sample_rate=16000)
        # Stereo: 2 samples per frame, so 2x the number of frames
        assert len(audio) == 32000
        assert audio.dtype == np.float32
        assert -1.0 <= audio[0] <= 1.0

    def test_wav_resample_24k_to_16k(self):
        """WAV at 24kHz should be resampled to 16kHz."""
        # 480 samples = 20ms at 24kHz
        samples = [0, 1000, -1000, 0] * 120
        wav_bytes = make_wav_bytes(samples, sample_rate=24000)
        audio = bytes_to_audio(wav_bytes, sample_rate=16000)
        # 480 samples at 24kHz / 48000 * 16000 = 16000 at 16kHz
        # Due to edge handling, result may be slightly different
        assert len(audio) > 0
        assert audio.dtype == np.float32

    def test_raw_pcm(self):
        """Raw int16 PCM should be decoded correctly."""
        samples = [0, 1000, -1000, 32767, -32768, 0]
        pcm = make_pcm_bytes(samples)
        audio = bytes_to_audio(pcm, sample_rate=16000)
        assert audio.dtype == np.float32
        assert len(audio) == 6
        # int16 / 32767 ≈ ±1.0
        assert audio[3] == pytest.approx(1.0, abs=0.001)
        assert audio[4] == pytest.approx(-1.0, abs=0.001)

    def test_empty_bytes(self):
        """Empty bytes should return empty array."""
        audio = bytes_to_audio(b"")
        assert len(audio) == 0
        assert audio.dtype == np.float32

    def test_invalid_wav_falls_back_to_pcm(self):
        """Garbage that isn't valid WAV should fall back to raw PCM."""
        # Even length, so np.frombuffer succeeds; RIFF header absent -> PCM path
        garbage = b"\x00\x01\x02\x03\x04\x05"
        audio = bytes_to_audio(garbage, sample_rate=16000)
        assert len(audio) == 3  # 6 bytes / 2 = 3 samples
        assert audio.dtype == np.float32


# ---------------------------------------------------------------------------
# audio_utils — audio_to_bytes
# ---------------------------------------------------------------------------

class TestAudioToBytes:
    def test_roundtrip_float32_wav(self):
        """Float32 [-1, 1] -> WAV -> back should preserve values."""
        original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        wav = audio_to_bytes(original, as_wav=True)
        assert wav[:4] == b"RIFF"
        audio = bytes_to_audio(wav, sample_rate=16000)
        # Roundtrip preserves only the actual signal — no padding
        assert len(audio) == 5
        assert audio[0] == pytest.approx(0.0, abs=0.01)
        assert audio[1] == pytest.approx(0.5, abs=0.01)
        assert audio[2] == pytest.approx(-0.5, abs=0.01)
        assert audio[3] == pytest.approx(1.0, abs=0.01)

    def test_roundtrip_pcm(self):
        """Float32 -> raw PCM -> back should be lossless."""
        original = np.array([0, 1000, -1000, 32767, -32768], dtype=np.float32)
        # Scale to [-1, 1] first
        scaled = original / 32767.0
        pcm = audio_to_bytes(scaled, as_wav=False)
        back = np.frombuffer(pcm, dtype=np.int16)
        assert list(back[:3]) == [0, 1000, -1000]

    def test_clipping(self):
        """Values beyond [-1.0, 1.0] should be clipped to int16 range."""
        original = np.array([0.0, 2.0, -3.0], dtype=np.float32)
        pcm = audio_to_bytes(original, as_wav=False)
        back = np.frombuffer(pcm, dtype=np.int16)
        # np.clip(x, -1, 1) => 2.0->1.0, -3.0->-1.0
        # int16(1.0*32767)=32767, int16(-1.0*32767)=-32767
        assert back[0] == 0
        assert back[1] == 32767
        assert back[2] == -32767


# ---------------------------------------------------------------------------
# audio_utils — create_wav_header
# ---------------------------------------------------------------------------

class TestWavHeader:
    def test_header_size(self):
        """WAV header should be exactly 44 bytes."""
        header = create_wav_header(num_frames=16000, sample_rate=16000)
        assert len(header) == 44

    def test_header_contains_RIFF(self):
        """Header should start with RIFF."""
        header = create_wav_header(num_frames=16000)
        assert header[:4] == b"RIFF"
        assert header[8:12] == b"WAVE"

    def test_header_data_size(self):
        """Header should reflect the correct data size."""
        header = create_wav_header(num_frames=16000, num_channels=1, sample_rate=16000)
        # bytes 4-7: file size - 8 (little-endian u32)
        file_size = struct.unpack("<I", header[4:8])[0]
        assert file_size == 36 + 16000 * 2  # header + data


# ---------------------------------------------------------------------------
# audio_utils — resample_audio
# ---------------------------------------------------------------------------

class TestResampleAudio:
    def test_no_resample_same_rate(self):
        """Same rate should return the same array."""
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        result = resample_audio(audio, 16000, 16000)
        assert np.array_equal(result, audio)

    def test_downsample(self):
        """44.1kHz to 16kHz should produce fewer samples."""
        audio = np.random.randn(44100).astype(np.float32) * 0.1
        result = resample_audio(audio, 44100, 16000)
        assert len(result) < len(audio)
        assert len(result) > 0

    def test_upsample(self):
        """8kHz to 16kHz should double the sample count."""
        audio = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)
        result = resample_audio(audio, 8000, 16000)
        assert len(result) > len(audio)

    def test_empty_audio(self):
        """Empty audio should return empty array."""
        audio = np.array([], dtype=np.float32)
        result = resample_audio(audio, 16000, 8000)
        assert len(result) == 0

    def test_single_sample(self):
        """Single sample should not crash."""
        audio = np.array([0.5], dtype=np.float32)
        result = resample_audio(audio, 16000, 8000)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# audio_utils — mix_audio
# ---------------------------------------------------------------------------

class TestMixAudio:
    def test_single_chunk(self):
        """Single chunk should pass through unchanged."""
        audio = np.array([0.1, 0.2, -0.1], dtype=np.float32)
        result = mix_audio([audio])
        assert np.allclose(result, audio)

    def test_two_chunks_same_length(self):
        """Two equal-length chunks should be summed."""
        a = np.array([0.5, 0.0], dtype=np.float32)
        b = np.array([0.5, 0.0], dtype=np.float32)
        result = mix_audio([a, b])
        assert np.allclose(result, [1.0, 0.0])

    def test_different_length_padding(self):
        """Chunks of different lengths should be padded before summing."""
        a = np.array([0.5, 0.0], dtype=np.float32)
        b = np.array([0.5], dtype=np.float32)
        result = mix_audio([a, b])
        assert len(result) == 2
        assert result[0] == pytest.approx(1.0, abs=0.001)

    def test_normalisation_prevents_clipping(self):
        """Sum > 1.0 should be normalised."""
        a = np.array([0.8], dtype=np.float32)
        b = np.array([0.8], dtype=np.float32)
        result = mix_audio([a, b])
        assert abs(result[0]) <= 1.0

    def test_empty_list(self):
        """Empty list should return empty array."""
        result = mix_audio([])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# audio_utils — compute_rms
# ---------------------------------------------------------------------------

class TestComputeRms:
    def test_silence(self):
        """Near-silence should have RMS ~0."""
        audio = np.zeros(1600, dtype=np.float32)
        rms = compute_rms(audio)
        assert rms == 0.0

    def test_constant_signal(self):
        """Constant signal of 0.5 should have RMS 0.5."""
        audio = np.full(1600, 0.5, dtype=np.float32)
        rms = compute_rms(audio)
        assert rms == pytest.approx(0.5, abs=0.001)

    def test_sine_wave(self):
        """Sine wave of amplitude 1.0 should have RMS ~0.707."""
        import math
        audio = np.array(
            [math.sin(2 * math.pi * 440 * i / 16000) for i in range(1600)],
            dtype=np.float32,
        )
        rms = compute_rms(audio)
        assert rms == pytest.approx(0.707, abs=0.01)


# ---------------------------------------------------------------------------
# audio_utils — trim_silence
# ---------------------------------------------------------------------------

class TestTrimSilence:
    def test_trim_leading(self):
        """Leading silence should be removed."""
        audio = np.concatenate([
            np.zeros(8000, dtype=np.float32),  # 0.5s silence
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.zeros(8000, dtype=np.float32),
        ])
        result = trim_silence(audio, sample_rate=16000, threshold=0.01)
        assert len(result) == 3

    def test_trim_all_silence(self):
        """Audio that is all silence should return empty array."""
        audio = np.zeros(1600, dtype=np.float32)
        result = trim_silence(audio, sample_rate=16000, threshold=0.01)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# VAD — SimpleEnergyVAD
# ---------------------------------------------------------------------------

class TestVADConfig:
    def test_default_config(self):
        """Default config should have sensible values."""
        config = VADConfig()
        assert config.sample_rate == 16000
        assert config.frame_duration_ms == 30
        assert config.energy_threshold == 0.02
        assert config.min_speech_duration_ms == 250
        assert config.min_silence_duration_ms == 500
        assert config.speech_pad_ms == 300


class TestSimpleEnergyVAD:
    def test_speech_detection_loud(self):
        """Loud audio should be detected as speech."""
        vad = SimpleEnergyVAD()
        audio = np.full(vad.frame_size, 0.1, dtype=np.float32)
        assert vad.is_speech(audio) is True

    def test_speech_detection_silent(self):
        """Near-silence should not be detected as speech."""
        vad = SimpleEnergyVAD()
        audio = np.zeros(vad.frame_size, dtype=np.float32)
        assert vad.is_speech(audio) is False

    def test_hysteresis_prevents_flapping(self):
        """Hysteresis should prevent rapid state transitions."""
        vad = SimpleEnergyVAD()
        # Start with loud frame (should trigger onset)
        loud = np.full(vad.frame_size, 0.1, dtype=np.float32)
        # Follow with quiet frame just above offset threshold (stay in speech)
        quiet = np.full(vad.frame_size, 0.015, dtype=np.float32)

        assert vad.is_speech_with_hysteresis(loud) is True
        # Quiet frame while in speech: lower offset threshold keeps speech alive
        assert vad.is_speech_with_hysteresis(quiet) is True
        # Second quiet frame: offset threshold exits speech
        vad.is_speech_with_hysteresis(quiet)  # second quiet → exit
        vad.reset()
        # After reset, quiet frame falls below onset threshold → no speech
        assert vad.is_speech_with_hysteresis(quiet) is False

    def test_detect_speech_segments(self):
        """Should return (start, end) sample indices for speech."""
        vad = SimpleEnergyVAD()
        # 1 second audio: speech from 0.3s to 0.7s
        speech_start = int(0.3 * vad.config.sample_rate)
        speech_end = int(0.7 * vad.config.sample_rate)
        audio = np.zeros(vad.config.sample_rate, dtype=np.float32)
        audio[speech_start:speech_end] = 0.1  # speech region

        segments = vad.detect_speech_segments(audio, use_hysteresis=False)
        assert len(segments) >= 1
        start_sample, end_sample = segments[0]
        assert start_sample < speech_end
        assert end_sample > speech_start

    def test_no_speech_returns_empty(self):
        """Audio with no speech should return empty list."""
        vad = SimpleEnergyVAD()
        audio = np.zeros(vad.config.sample_rate, dtype=np.float32)  # 1s silence
        segments = vad.detect_speech_segments(audio)
        assert segments == []

    def test_reset(self):
        """Reset should clear hysteresis state."""
        vad = SimpleEnergyVAD()
        loud = np.full(vad.frame_size, 0.1, dtype=np.float32)
        vad.is_speech_with_hysteresis(loud)
        vad.reset()
        # After reset, hysteresis starts from silence
        quiet = np.full(vad.frame_size, 0.01, dtype=np.float32)
        assert vad.is_speech_with_hysteresis(quiet) is False

    def test_short_audio(self):
        """Audio shorter than one frame should not crash."""
        vad = SimpleEnergyVAD()
        audio = np.array([0.1, 0.2], dtype=np.float32)
        segments = vad.detect_speech_segments(audio)
        assert segments == []


# Run with: pytest tests/test_gateway.py -v
