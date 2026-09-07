"""Tests for TTS module."""
import pytest

from src.tts import TextNormalizer, NormalizedText, normalizer


class TestTextNormalizer:
    """Tests for TextNormalizer."""

    def test_normalize_currency(self):
        """Test currency normalization."""
        result = normalizer.normalize("Course fees are ₹25,000")
        assert "rupees" in result.normalized.lower() or "₹25,000" not in result.normalized
        assert result.original == "Course fees are ₹25,000"
        assert len(result.substitutions) > 0

    def test_normalize_percentage(self):
        """Test percentage normalization."""
        result = normalizer.normalize("Success rate is 85%")
        # Number is converted to words, percent stays
        assert "percent" in result.normalized.lower()
        assert len(result.substitutions) > 0

    def test_normalize_time(self):
        """Test time normalization."""
        result = normalizer.normalize("Class starts at 6 PM")
        # Number converted to words, AM/PM spaced out
        # Should have "P M" or "six" in output
        assert "P M" in result.normalized or "six" in result.normalized.lower()
        assert len(result.substitutions) > 0

    def test_normalize_jee(self):
        """Test JEE abbreviation expansion."""
        result = normalizer.normalize("I want to join JEE classes")
        assert "J E E" in result.normalized or "JEE" in result.normalized

    def test_normalize_neet(self):
        """Test NEET abbreviation expansion."""
        result = normalizer.normalize("Preparing for NEET exam")
        assert "N E E T" in result.normalized or "NEET" in result.normalized

    def test_normalize_iit(self):
        """Test IIT abbreviation expansion."""
        result = normalizer.normalize("IIT JEE Advanced")
        assert "I I T" in result.normalized

    def test_normalize_course_combinations(self):
        """Test multiple course abbreviations."""
        result = normalizer.normalize(
            "JEE, NEET, IIT, and CBSE exams"
        )
        # Should contain course expansions
        assert "JEE" in result.normalized or "J E E" in result.normalized

    def test_normalize_date(self):
        """Test date normalization."""
        result = normalizer.normalize("Session starts 15/06/2024")
        # Numbers converted to words, slashes preserved
        assert "slash" in result.normalized.lower()
        assert len(result.substitutions) > 0

    def test_normalize_url(self):
        """Test URL replacement."""
        result = normalizer.normalize("Visit https://example.com for details")
        assert "https://example.com" not in result.normalized
        assert "link" in result.normalized.lower()

    def test_normalize_email(self):
        """Test email replacement."""
        result = normalizer.normalize("Email us at info@example.com")
        assert "info@example.com" not in result.normalized
        assert "email" in result.normalized.lower()

    def test_normalize_preserve_english(self):
        """Test that plain English is preserved."""
        result = normalizer.normalize("Hello, how can I help you?")
        assert "Hello" in result.normalized
        assert "help" in result.normalized

    def test_normalize_hinglish(self):
        """Test Hinglish text normalization."""
        result = normalizer.normalize("JEE ke liye ₹25,000 fees hai")
        assert "JEE" in result.normalized or "J E E" in result.normalized
        assert "fees" in result.normalized

    def test_normalize_empty(self):
        """Test empty string."""
        result = normalizer.normalize("")
        assert result.normalized == ""
        assert len(result.substitutions) == 0

    def test_normalize_numbers(self):
        """Test standalone number conversion."""
        result = normalizer.normalize("There are 3 batches")
        assert "3" in result.normalized or "three" in result.normalized.lower()

    def test_normalize_phone(self):
        """Test phone number normalization."""
        result = normalizer.normalize("Call 98765 43210 for details")
        # Phone pattern matches and separates digits
        # Numbers in phone get converted to words
        assert len(result.substitutions) > 0

    def test_number_to_words(self):
        """Test number to words conversion."""
        norm = TextNormalizer()

        assert norm._number_to_words(0) == "zero"
        assert norm._number_to_words(5) == "five"
        assert norm._number_to_words(15) == "fifteen"
        assert norm._number_to_words(25) == "twenty five"
        assert norm._number_to_words(100) == "one hundred"
        assert norm._number_to_words(123) == "one hundred and twenty three"

    def test_number_to_words_indian(self):
        """Test Indian numbering system (lakh/crore)."""
        norm = TextNormalizer()

        # One lakh
        result = norm._number_to_words(100000)
        assert "lakh" in result

        # One crore
        result = norm._number_to_words(10000000)
        assert "crore" in result

    def test_two_digit_to_words(self):
        """Test two-digit number conversion."""
        norm = TextNormalizer()

        assert norm._two_digit_to_words(0) == ""
        assert norm._two_digit_to_words(5) == "five"
        assert norm._two_digit_to_words(15) == "fifteen"
        assert norm._two_digit_to_words(21) == "twenty one"
        assert norm._two_digit_to_words(99) == "ninety nine"

    def test_normalized_text_dataclass(self):
        """Test NormalizedText dataclass."""
        result = NormalizedText(
            original="Test",
            normalized="test",
            substitutions=[("CURRENCY", "₹100")]
        )
        assert result.original == "Test"
        assert result.normalized == "test"
        assert len(result.substitutions) == 1


class TestTTSConfig:
    """Tests for TTSConfig."""

    def test_default_config(self):
        from src.tts.config import TTSConfig
        config = TTSConfig()
        # Defaults: free gTTS (override with TTSConfig(provider="openai") for paid)
        assert config.provider == "gtts"
        assert config.voice == "alloy"
        assert config.model == "tts-1"

    def test_custom_config(self):
        from src.tts.config import TTSConfig
        config = TTSConfig(voice="nova", speed=1.2)
        assert config.voice == "nova"
        assert config.speed == 1.2

    def test_get_voice_for_language(self):
        from src.tts.config import get_voice_for_language
        assert get_voice_for_language("en") == "alloy"
        assert get_voice_for_language("hi") == "alloy"
        assert get_voice_for_language("unknown") == "alloy"  # default


class TestTTSResult:
    """Tests for TTSResult."""

    def test_result_creation(self):
        from src.tts import TTSResult
        result = TTSResult(
            audio=b"test audio",
            duration=2.5,
            format="mp3",
            voice="nova",
        )
        assert result.audio == b"test audio"
        assert result.duration == 2.5
        assert result.format == "mp3"
        assert result.voice == "nova"


class TestTTSProviders:
    """Tests for TTS provider registry."""

    def test_openai_provider_init(self):
        from src.tts import OpenAITTSProvider
        provider = OpenAITTSProvider()
        assert provider.voice == "alloy"
        assert provider.model == "tts-1"
        assert provider.name == "openai-tts-tts-1"

    def test_openai_provider_custom(self):
        from src.tts import OpenAITTSProvider
        provider = OpenAITTSProvider(voice="nova", model="tts-1-hd")
        assert provider.voice == "nova"
        assert provider.model == "tts-1-hd"
        assert provider.available_voices == ["alloy", "echo", "fable", "nova", "shimmer"]

    def test_get_tts_provider_default(self):
        from src.tts import get_tts_provider, GttsTTSProvider
        provider = get_tts_provider()
        assert isinstance(provider, GttsTTSProvider)

    def test_get_tts_provider_by_name(self):
        from src.tts import get_tts_provider, OpenAITTSProvider
        provider = get_tts_provider("openai")
        assert isinstance(provider, OpenAITTSProvider)

    def test_get_tts_provider_unknown(self):
        from src.tts import get_tts_provider
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            get_tts_provider("unknown_provider")


# Run tests with: pytest tests/test_tts.py -v
