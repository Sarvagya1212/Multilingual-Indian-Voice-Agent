from src.logger import setup_logger
from src.config import settings

logger = setup_logger(__name__)

def main():
    logger.info("Starting Multilingual Indian Voice Agent")
    logger.info(f"Model: {settings.llm_model}")
    logger.info(f"STT: {settings.stt_provider} ({settings.whisper_model})")
    logger.info(f"TTS: {settings.tts_provider} ({settings.tts_voice})")
    logger.info("Voice agent ready. Import and use pipeline modules.")

if __name__ == "__main__":
    main()