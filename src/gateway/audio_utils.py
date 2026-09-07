"""Audio utilities for the voice gateway.

Handles conversion between bytes, numpy arrays, WAV headers, and resampling.
"""
import struct
import io
import wave
import numpy as np
import numpy.typing as npt
from typing import Tuple

from src.logger import setup_logger

logger = setup_logger(__name__)

# Allowed sample rates for WAV encoding
SUPPORTED_SAMPLE_RATES = (8000, 16000, 24000, 44100, 48000)


def bytes_to_audio(audio_bytes: bytes, sample_rate: int = 16000) -> npt.NDArray[np.float32]:
    """Convert audio bytes to float32 numpy array (normalised to [-1, 1]).

    Handles both WAV-wrapped and raw PCM input. WAV headers are stripped
    automatically; raw PCM is decoded as 16-bit signed little-endian.

    Args:
        audio_bytes: Raw audio bytes.
        sample_rate: Sample rate hint (used only for raw PCM; WAV header drives it).

    Returns:
        Float32 audio array normalised to [-1.0, 1.0].
    """
    if not audio_bytes:
        return np.array([], dtype=np.float32)

    # Try WAV first
    if audio_bytes[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                actual_sr = wf.getframerate()
                # WAV sampwidth is in bytes; 1=u8, 2=int16, 3=int24, 4=int32
                dtype = {
                    1: np.uint8,
                    2: np.int16,
                    3: np.int32,  # 24-bit stored in int32
                    4: np.int32,
                }.get(wf.getsampwidth(), np.int16)
                raw = wf.readframes(wf.getnframes())
                float_audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
                if dtype == np.int16:
                    float_audio /= 32767.0
                elif dtype == np.int32:
                    float_audio /= 2147483647.0
                elif dtype == np.uint8:
                    float_audio = (float_audio - 128.0) / 128.0
                # Resample if needed
                if actual_sr != sample_rate:
                    float_audio = resample_audio(float_audio, actual_sr, sample_rate)
                return float_audio
        except Exception as e:
            logger.warning(f"Failed to parse WAV header: {e}, treating as raw PCM")

    # Raw PCM — 16-bit signed little-endian
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    return audio_np


def audio_to_bytes(
    audio_np: npt.NDArray[np.float32],
    sample_rate: int = 16000,
    as_wav: bool = True,
) -> bytes:
    """Convert float32 numpy array to bytes.

    Args:
        audio_np: Float32 audio array (expected range [-1.0, 1.0]).
        sample_rate: Sample rate for WAV header.
        as_wav: If True, prepend WAV header; if False, return raw PCM.

    Returns:
        Audio bytes.
    """
    # Clip to valid range, then convert to int16. Scale by 32767 (not 32768) to
    # stay within the int16 range even at full-scale.
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767.0).astype(np.int16)

    if as_wav:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return buffer.getvalue()
    return audio_int16.tobytes()


def create_wav_header(
    num_frames: int,
    num_channels: int = 1,
    sample_rate: int = 16000,
) -> bytes:
    """Create a manual WAV RIFF header.

    Args:
        num_frames: Number of audio frames.
        num_channels: Number of channels.
        sample_rate: Sample rate in Hz.

    Returns:
        44-byte WAV header.
    """
    byte_rate = sample_rate * num_channels * 2
    block_align = num_channels * 2
    data_size = num_frames * num_channels * 2

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        16,  # bits per sample
        b"data",
        data_size,
    )
    return header


def resample_audio(
    audio: npt.NDArray[np.float32],
    orig_sr: int,
    target_sr: int,
) -> npt.NDArray[np.float32]:
    """Resample audio using linear interpolation.

    Args:
        audio: Input audio array.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio array.
    """
    if orig_sr == target_sr or orig_sr <= 0 or target_sr <= 0:
        return audio

    # Prevent degenerate cases
    if len(audio) < 2:
        return audio

    duration = len(audio) / orig_sr
    target_length = max(1, int(duration * target_sr))
    indices = np.linspace(0, len(audio) - 1, target_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def mix_audio(
    chunks: list[npt.NDArray[np.float32]],
) -> npt.NDArray[np.float32]:
    """Mix multiple audio chunks into a single array.

    All chunks are padded/trimmed to the longest length and summed.

    Args:
        chunks: List of audio arrays.

    Returns:
        Mixed audio array.
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    max_len = max(len(c) for c in chunks)
    padded = [np.pad(c, (0, max_len - len(c))) for c in chunks]
    mixed = np.sum(padded, axis=0)
    # Normalise to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed /= max_val
    return mixed.astype(np.float32)


def compute_rms(audio: npt.NDArray[np.float32]) -> float:
    """Compute RMS energy of an audio array.

    Args:
        audio: Float32 audio array.

    Returns:
        RMS value.
    """
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio**2)))


def trim_silence(
    audio: npt.NDArray[np.float32],
    sample_rate: int = 16000,
    threshold: float = 0.01,
    frame_duration_ms: int = 10,
) -> npt.NDArray[np.float32]:
    """Trim leading and trailing silence from audio.

    Args:
        audio: Float32 audio array.
        sample_rate: Sample rate.
        threshold: Energy threshold below which is considered silence.
        frame_duration_ms: Frame size for energy computation.

    Returns:
        Trimmed audio (or empty array if all silence).
    """
    if len(audio) == 0:
        return audio

    frame_size = int(sample_rate * frame_duration_ms / 1000)
    if frame_size < 1:
        frame_size = 1

    # Find first non-silent frame
    start = None
    for i in range(0, len(audio) - frame_size, frame_size):
        if compute_rms(audio[i : i + frame_size]) > threshold:
            start = i
            break

    # If no speech found, return empty
    if start is None:
        return np.array([], dtype=np.float32)

    # Find last non-silent frame
    end = None
    for i in range(len(audio) - frame_size, 0, -frame_size):
        if compute_rms(audio[i : i + frame_size]) > threshold:
            end = i + frame_size
            break

    if end is None:
        end = len(audio)

    return audio[start:end]
