"""Shared helpers for the demo scripts.

Demos must work without API keys so they're usable in CI and by
portfolio reviewers. When a key is present, the demo will exercise
the real provider; when missing, it falls back to deterministic stubs
that simulate the same shape of response.
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

# Fix Windows cp1252 console encoding for Hindi/Devanagari output.
# Without this, print() raises UnicodeEncodeError for Devanagari chars.
try:
    if sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover — reconfigure fails in some IDE terminals
    pass


# ----------------------------------------------------------------------
# Display helpers (ASCII-only to survive Windows cp1252)
# ----------------------------------------------------------------------


def header(title: str) -> None:
    """Print a centred banner with the demo title."""
    bar = "=" * 64
    print()
    print(bar)
    print(f"  {title}")
    print(bar)
    print()


def step(num: int, total: int, label: str) -> None:
    """Print a numbered step indicator."""
    print(f"[{num}/{total}] {label}")


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def miss(msg: str) -> None:
    """Marker for an API-key-gated step that is being simulated."""
    print(f"  [SIM] {msg}")


def has_api_key(env_var: str) -> bool:
    """Check whether the named environment variable is set and non-empty."""
    val = os.getenv(env_var, "")
    return bool(val and val.strip())


# ----------------------------------------------------------------------
# Synthetic audio
# ----------------------------------------------------------------------


def create_synthetic_wav(
    duration: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> bytes:
    """Create a small WAV byte string of a sine wave (simulated speech).

    This is the same generator the baseline test uses; demo scripts
    reuse it so reviewers can run demos without a microphone.
    """
    import math
    import numpy as np

    samples = int(duration * sample_rate)
    t = np.arange(samples) / sample_rate
    envelope = 0.3 + 0.2 * np.sin(2 * np.pi * 2 * t)
    wave_data = (32767 * envelope * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wave_data.tobytes())
    return buffer.getvalue()


def fake_tts_audio(text: str) -> bytes:
    """Generate a placeholder TTS payload for a text response.

    The size scales with the response length so latency estimates are
    plausible. The bytes are valid WAV so anything that tries to
    decode them won't crash.
    """
    duration = max(0.5, min(4.0, len(text) * 0.04))
    return create_synthetic_wav(duration=duration, frequency=320.0)


# ----------------------------------------------------------------------
# Deterministic simulation
# ----------------------------------------------------------------------


@dataclass
class SimTurnResult:
    """What a simulated turn looks like — mirrors TTSResult shape."""

    transcript: str
    response_text: str
    detected_language: str
    audio: bytes
    duration: float
    latency_ms: float
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    state: str = "IDLE"
    grounded: bool = True
    notes: List[str] = field(default_factory=list)


@dataclass
class SimulatedAgent:
    """Drop-in agent stub for demos without API keys.

    Mirrors the public surface of `AgentOrchestrator` (create_session,
    process_turn, interrupt) but returns deterministic responses based
    on the input transcript.
    """

    session_id: str = ""
    state: str = "IDLE"
    turns: int = 0
    history: List[SimTurnResult] = field(default_factory=list)

    def create_session(self) -> "SimulatedAgent":
        self.session_id = f"demo-{int(time.time() * 1000)}"
        self.state = "IDLE"
        return self

    async def process_turn(self, transcript: str, language: str = "en") -> SimTurnResult:
        """Produce a deterministic response for a given transcript."""
        self.turns += 1
        start = time.perf_counter()
        self.state = "TRANSCRIBING"
        self.state = "THINKING"
        # Scripted responses per demo (overridden via subclass or hook)
        response = scripted_response(transcript, language)
        self.state = "GENERATING"
        await asyncio.sleep(0.05)  # simulate LLM round-trip
        audio = fake_tts_audio(response)
        self.state = "SPEAKING"
        await asyncio.sleep(0.02)  # simulate first-byte latency
        self.state = "IDLE"
        elapsed_ms = (time.perf_counter() - start) * 1000
        result = SimTurnResult(
            transcript=transcript,
            response_text=response,
            detected_language=language,
            audio=audio,
            duration=len(audio) / 16000 / 2,  # 16-bit mono at 16kHz
            latency_ms=elapsed_ms,
            state=self.state,
        )
        self.history.append(result)
        return result

    def interrupt(self) -> bool:
        if self.state in ("SPEAKING", "GENERATING"):
            self.state = "INTERRUPTED"
            return True
        return False


# ----------------------------------------------------------------------
# Scripted responses per language
# ----------------------------------------------------------------------


_SCRIPTED: Dict[str, Dict[str, str]] = {
    "en": {
        "default": "We offer JEE, NEET, and CBSE courses. What are you interested in?",
    },
    "hi": {
        "default": "हम JEE, NEET और CBSE कोर्स ऑफर करते हैं। आपकी रुचि किसमें है?",
    },
    "hinglish": {
        "default": "Hum JEE, NEET aur CBSE courses offer karte hain. Aapki interest kisme hai?",
    },
}


def scripted_response(transcript: str, language: str) -> str:
    """Look up a scripted response based on language and transcript cues."""
    bucket = _SCRIPTED.get(language, _SCRIPTED["en"])
    t = transcript.lower()
    if "fee" in t or "fees" in t or "कीमत" in t or "kitna" in t:
        return {
            "en": "JEE course fees are INR 1,50,000 with EMI options.",
            "hi": "JEE कोर्स की फीस 1,50,000 रुपये है, EMI उपलब्ध है।",
            "hinglish": "JEE course ki fees INR 1,50,000 hai, EMI option available hai.",
        }[language]
    if "eligibility" in t or "eligible" in t or "योग्यता" in t or "eligible" in t:
        return {
            "en": "Class 11 students with 75%+ in Class 10 are eligible.",
            "hi": "कक्षा 11 के छात्र जिन्होंने कक्षा 10 में 75% से अधिक अंक प्राप्त किए हैं, पात्र हैं।",
            "hinglish": "Class 11 ke students jo Class 10 mein 75%+ score kiya hai, eligible hain.",
        }[language]
    if "demo" in t or "डेमो" in t:
        return {
            "en": "I have scheduled a demo for tomorrow at 10 AM. You will receive a confirmation SMS.",
            "hi": "मैंने कल सुबह 10 बजे के लिए डेमो शेड्यूल किया है। आपको पुष्टि एसएमएस मिलेगा।",
            "hinglish": "Maine kal subah 10 baje ke liye demo schedule kiya hai. Aapko confirmation SMS milega.",
        }[language]
    if "hostel" in t or "छात्रावास" in t:
        return {
            "en": "Hostel facility is available at our Hyderabad and Bangalore centres.",
            "hi": "हमारे हैदराबाद और बैंगलोर केंद्रों पर छात्रावास सुविधा उपलब्ध है।",
            "hinglish": "Humare Hyderabad aur Bangalore centres par hostel facility available hai.",
        }[language]
    if "scholarship" in t or "स्कॉलरशिप" in t:
        return {
            "en": "Scholarships are available for students with 90%+ in their previous class.",
            "hi": "पिछली कक्षा में 90%+ अंक प्राप्त करने वाले छात्रों के लिए स्कॉलरशिप उपलब्ध है।",
            "hinglish": "Pichli class mein 90%+ score karne wale students ke liye scholarship available hai.",
        }[language]
    return bucket["default"]


# ----------------------------------------------------------------------
# Async runner helper
# ----------------------------------------------------------------------


def run(coro_factory: Callable[[], Awaitable[None]]) -> None:
    """Run an async demo main, surfacing exceptions cleanly."""
    try:
        asyncio.run(coro_factory())
    except KeyboardInterrupt:
        print()
        print("Demo interrupted by user.")
        sys.exit(0)
