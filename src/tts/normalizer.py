"""Text normalization for TTS.

Handles:
- Currency (₹25,000, Rs. 1000)
- Percentages (25%)
- Times (6 PM, 10:30 AM)
- Indian exam abbreviations (JEE, NEET, IIT)
- Phone numbers, dates, URLs
- Numbers in context
"""
import re
from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass
class NormalizedText:
    """Result of text normalization."""
    original: str
    normalized: str
    substitutions: List[Tuple[str, str]] = field(default_factory=list)


class TextNormalizer:
    """Normalize text for TTS synthesis.

    Converts written forms into spoken forms:
    - ₹25,000 -> "twenty five thousand rupees"
    - 25% -> "twenty five percent"
    - 6 PM -> "six P M"
    - JEE -> "J E E"
    """

    # Number words for Indian numbering system
    UNITS = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
             "sixteen", "seventeen", "eighteen", "nineteen"]
    TENS = ["", "", "twenty", "thirty", "forty", "fifty",
            "sixty", "seventy", "eighty", "ninety"]

    # Currency patterns - convert to spoken form
    CURRENCY_PATTERNS = [
        # ₹25,000 or ₹25000 -> "twenty five thousand rupees"
        (r'₹\s*(\d{1,3}(?:,\d{2,3})*(?:\.\d+)?)', 'rupees'),
        # Rs. 25,000 or Rs 25000
        (r'Rs\.?\s*(\d{1,3}(?:,\d{2,3})*(?:\.\d+)?)', 'rupees'),
        # INR 25,000
        (r'INR\s*(\d{1,3}(?:,\d{2,3})*(?:\.\d+)?)', 'rupees'),
    ]

    # Percentage patterns
    PERCENTAGE_PATTERNS = [
        (r'(\d+(?:\.\d+)?)\s*%', 'percent'),
        (r'(\d+(?:\.\d+)?)\s*per\s*cent', 'percent'),
    ]

    # Time patterns
    TIME_PATTERNS = [
        (r'(\d{1,2}):(\d{2})\s*(AM|PM)', r'\1 \3'),
        (r'(\d{1,2})\s*AM\b', r'\1 A M'),
        (r'(\d{1,2})\s*PM\b', r'\1 P M'),
        (r'(\d{1,2})\s*a\.?m\.?', r'\1 A M'),
        (r'(\d{1,2})\s*p\.?m\.?', r'\1 P M'),
    ]

    # Indian exam/course abbreviations
    COURSE_PATTERNS = [
        (r'\bIIT\s*JEE\b', 'I I T J E E'),
        (r'\bJEE\s*Main\b', 'J E E Main'),
        (r'\bJEE\s*Advanced\b', 'J E E Advanced'),
        (r'\bJEE\b', 'J E E'),
        (r'\bNEET\b', 'N E E T'),
        (r'\bIIT\b', 'I I T'),
        (r'\bNIT\b', 'N I T'),
        (r'\bAIIMS\b', 'A I I M S'),
        (r'\bCBSE\b', 'C B S E'),
        (r'\bICSE\b', 'I C S E'),
        (r'\bNDA\b', 'N D A'),
        (r'\bCAT\b', 'C A T'),
        (r'\bGATE\b', 'G A T E'),
        (r'\bAI\s*/\s*ML\b', 'A I slash M L'),
        (r'\bML\b', 'M L'),
        (r'\bAI\b', 'A I'),
        (r'\bUG\b', 'U G'),
        (r'\bPG\b', 'P G'),
        (r'\bPhD\b', 'P h D'),
        (r'\bB\.?Tech\b', 'B Tech'),
        (r'\bM\.?Tech\b', 'M Tech'),
        (r'\bMBA\b', 'M B A'),
    ]

    # Date patterns
    DATE_PATTERNS = [
        (r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b', r'\1 slash \2 slash \3'),
    ]

    # Phone number patterns (10-digit Indian numbers)
    PHONE_PATTERNS = [
        (r'\+91[-\s]?(\d{5})[-\s]?(\d{5})\b', r'+ nine one \1 \2'),
        (r'\b(\d{5})[-\s]?(\d{5})\b', r'\1 \2'),
    ]

    # URL/email patterns
    URL_PATTERNS = [
        (r'https?://[^\s]+', 'link'),
        (r'www\.[^\s]+', 'link'),
        (r'\b[\w.-]+@[\w.-]+\.\w+\b', 'email address'),
    ]

    def normalize(self, text: str) -> NormalizedText:
        """Normalize text for TTS synthesis.

        Args:
            text: Input text

        Returns:
            NormalizedText with original, normalized, and list of substitutions
        """
        normalized = text
        substitutions = []

        # Order matters - apply specific patterns first
        pattern_groups = [
            ("URL", self.URL_PATTERNS),
            ("PHONE", self.PHONE_PATTERNS),
            ("CURRENCY", self.CURRENCY_PATTERNS),
            ("PERCENTAGE", self.PERCENTAGE_PATTERNS),
            ("TIME", self.TIME_PATTERNS),
            ("DATE", self.DATE_PATTERNS),
            ("COURSE", self.COURSE_PATTERNS),
        ]

        for group_name, patterns in pattern_groups:
            for pattern, replacement in patterns:
                new_text, count = re.subn(pattern, replacement, normalized)
                if count > 0:
                    normalized = new_text
                    substitutions.append((group_name, f"{count}x {pattern}"))

        # Handle numbers in context (Indian numbering)
        normalized = self._normalize_numbers(normalized, substitutions)

        # Handle punctuation
        normalized = self._normalize_punctuation(normalized)

        # Clean up extra spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return NormalizedText(
            original=text,
            normalized=normalized,
            substitutions=substitutions,
        )

    def _normalize_numbers(self, text: str, substitutions: list) -> str:
        """Convert standalone numbers to words."""
        # Find standalone numbers (not part of code, dates, etc.)
        def replace_number(match):
            num_str = match.group(0).replace(',', '')
            try:
                num = int(num_str)
                if 0 <= num <= 999999:
                    return self._number_to_words(num)
            except (ValueError, IndexError):
                pass
            return match.group(0)

        # Only match standalone numbers (with word boundaries)
        text = re.sub(r'\b\d{1,6}\b', replace_number, text)
        return text

    def _number_to_words(self, n: int) -> str:
        """Convert number to Indian English words.

        Uses lakh/crore system:
        - 1,00,000 = one lakh
        - 1,00,00,000 = one crore
        """
        if n == 0:
            return "zero"

        parts = []

        # Crores (1,00,00,000+)
        if n >= 10000000:
            crores = n // 10000000
            parts.append(self._two_digit_to_words(crores))
            parts.append("crore")
            n %= 10000000

        # Lakhs (1,00,000+)
        if n >= 100000:
            lakhs = n // 100000
            parts.append(self._two_digit_to_words(lakhs))
            parts.append("lakh")
            n %= 100000

        # Thousands (1,000+)
        if n >= 1000:
            thousands = n // 1000
            parts.append(self._two_digit_to_words(thousands))
            parts.append("thousand")
            n %= 1000

        # Hundreds
        if n >= 100:
            hundreds = n // 100
            parts.append(self.UNITS[hundreds])
            parts.append("hundred")
            n %= 100

        # Tens and units
        if n > 0:
            if parts:
                parts.append("and")
            parts.append(self._two_digit_to_words(n))

        return " ".join(parts)

    def _two_digit_to_words(self, n: int) -> str:
        """Convert 0-99 to words."""
        if n < 10:
            return self.UNITS[n]
        elif n < 20:
            return self.TEENS[n - 10]
        else:
            tens = self.TENS[n // 10]
            units = self.UNITS[n % 10]
            return f"{tens} {units}".strip()

    def _normalize_punctuation(self, text: str) -> str:
        """Handle punctuation for better TTS pronunciation."""
        # Add spaces after punctuation
        text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
        # Handle ellipsis
        text = text.replace('...', ' ')
        # Handle em/en dashes
        text = text.replace('—', ' - ').replace('–', ' - ')
        return text


# Singleton instance
normalizer = TextNormalizer()
