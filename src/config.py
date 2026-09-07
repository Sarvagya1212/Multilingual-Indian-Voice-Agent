from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # STT Configuration
    stt_provider: str = "openai_whisper"
    whisper_model: str = "base"
    whisper_device: str = "cpu"

    # TTS Configuration
    tts_provider: str = "gtts"   # free (gTTS); use "openai" for paid
    tts_voice: str = "alloy"

    # LLM Configuration
    llm_provider: str = "ollama"  # free (local); use "anthropic" for paid
    llm_model: str = "llama3.1:latest"  # change to "claude-sonnet-4-20250514" for Anthropic

    # Vector Database
    vector_db_provider: str = "chromadb"
    chromadb_path: str = "./chromadb_data"

    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Logging
    log_level: str = "INFO"

    # Audio
    sample_rate: int = 16000
    channels: int = 1

    # Application
    session_timeout_minutes: int = 30
    max_tool_call_retries: int = 3

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()