"""Voice Activity Detection (VAD) for the voice gateway.

Provides energy-based VAD with configurable thresholds and segment merging.
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from src.logger import setup_logger
from src.gateway.audio_utils import compute_rms

logger = setup_logger(__name__)


@dataclass
class VADConfig:
    """Configuration for SimpleEnergyVAD."""

    sample_rate: int = 16000
    frame_duration_ms: int = 30
    energy_threshold: float = 0.02
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 300


class SimpleEnergyVAD:
    """Energy-based voice activity detection.

    Detects speech segments in raw PCM audio by computing frame-level RMS energy
    and applying hysteresis (higher threshold for speech onset, lower for offset).
    """

    def __init__(self, config: VADConfig | None = None):
        """Initialize VAD.

        Args:
            config: VAD configuration. Uses defaults if not provided.
        """
        self.config = config or VADConfig()
        self.frame_size = int(
            self.config.sample_rate * self.config.frame_duration_ms / 1000
        )
        # Hysteresis thresholds
        self._onset_threshold = self.config.energy_threshold
        self._offset_threshold = self.config.energy_threshold * 0.5
        self._in_speech = False

    def is_speech(self, audio: npt.NDArray[np.float32]) -> bool:
        """Detect if an audio frame contains speech.

        Args:
            audio: Single frame of audio.

        Returns:
            True if the frame contains speech.
        """
        energy = compute_rms(audio)
        return energy > self.config.energy_threshold

    def is_speech_with_hysteresis(self, audio: npt.NDArray[np.float32]) -> bool:
        """Detect speech with hysteresis to reduce flapping.

        Uses a higher threshold to START speech and a lower threshold to END it.
        This prevents rapid on/off transitions at the boundary.

        Args:
            audio: Single frame of audio.

        Returns:
            True if the frame contains speech.
        """
        energy = compute_rms(audio)
        if self._in_speech:
            self._in_speech = energy > self._offset_threshold
        else:
            self._in_speech = energy > self._onset_threshold
        return self._in_speech

    def detect_speech_segments(
        self,
        audio: npt.NDArray[np.float32],
        use_hysteresis: bool = True,
    ) -> List[Tuple[int, int]]:
        """Detect speech segments in audio.

        Args:
            audio: Full audio array.
            use_hysteresis: If True, use hysteresis for onset/offset detection.

        Returns:
            List of (start_sample, end_sample) tuples for speech segments.
        """
        if len(audio) < self.frame_size:
            return []

        num_frames = len(audio) // self.frame_size
        is_speech_fn = self.is_speech_with_hysteresis if use_hysteresis else self.is_speech
        speech_frames = [
            i for i in range(num_frames)
            if is_speech_fn(audio[i * self.frame_size:(i + 1) * self.frame_size])
        ]

        if not speech_frames:
            return []

        # Merge nearby speech frames (within min_silence gap)
        min_speech_frames = int(
            self.config.min_speech_duration_ms / self.config.frame_duration_ms
        )
        min_gap_frames = int(
            self.config.min_silence_duration_ms / self.config.frame_duration_ms
        )

        segments: List[Tuple[int, int]] = []
        start = speech_frames[0]
        prev = speech_frames[0]

        for frame in speech_frames[1:]:
            if frame - prev <= min_gap_frames:
                prev = frame
            else:
                if prev - start + 1 >= min_speech_frames:
                    segments.append((start, prev))
                start = frame
                prev = frame

        # Last segment
        if prev - start + 1 >= min_speech_frames:
            segments.append((start, prev))

        # Convert frame indices to sample indices
        pad_frames = int(self.config.speech_pad_ms / self.config.frame_duration_ms)
        padded_segments: List[Tuple[int, int]] = []
        for start_frame, end_frame in segments:
            start_sample = max(0, (start_frame - pad_frames) * self.frame_size)
            end_sample = min(
                len(audio), (end_frame + 1 + pad_frames) * self.frame_size
            )
            padded_segments.append((start_sample, end_sample))

        return padded_segments

    def reset(self) -> None:
        """Reset VAD state (call between sessions)."""
        self._in_speech = False
