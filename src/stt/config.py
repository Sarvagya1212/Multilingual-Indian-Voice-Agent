"""STT configuration."""
from pydantic import BaseModel
from typing import Optional


class STTConfig(BaseModel):
    """Configuration for STT providers."""

    # Provider selection
    provider: str = "openai_whisper"

    # Whisper-specific options
    model_name: str = "base"  # tiny, base, small, medium, large
    device: str = "cpu"  # cpu, cuda
    language: Optional[str] = None  # Force language

    # Transcription options
    temperature: float = 0.0
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    initial_prompt: Optional[str] = None  # Prompt to guide transcription

    # Streaming options
    chunk_length_s: float = 30.0  # Max chunk length for streaming
    condition_on_previous_text: bool = True  # Use previous text as context

    class Config:
        """Pydantic config."""
        extra = "ignore"


# Language code mappings for STT providers
LANGUAGE_CODES = {
    "english": "en",
    "hindi": "hi",
    "hinglish": "hi-en",
    "bilingual": "hi-en",
    "tamil": "ta",
    "telugu": "te",
    "bengali": "bn",
    "marathi": "mr",
    "gujarati": "gu",
    "kannada": "kn",
    "malayalam": "ml",
    "punjabi": "pa",
}
