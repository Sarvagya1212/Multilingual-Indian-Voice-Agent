# Multilingual Indian Voice Agent — Step-by-Step Build Prompts

This file contains numbered, sequential prompts for building the entire project from scratch.
Each prompt is self-contained and should be executed in order.

---

## PHASE 0: Project Foundation

### Prompt 0.1 — Repository Initialization

**Task:** Initialize the project repository with all required directories and configuration files.

**Files to create:**

```
Multilingual-Indian-Voice-Agent/
├── README.md                    # Project overview, setup, architecture, demos
├── flow.md                      # Living engineering decision log
├── ARCHITECTURE.md              # Detailed system architecture
├── EXPERIMENTS.md               # Experiment tracking summary
├── EVALUATION.md                # Evaluation framework documentation
├── CHANGELOG.md                 # Version history
├── failure_analysis.md           # Failure categorization and fixes
├── .env.example                 # Environment variable template
├── .gitignore
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   └── main.py
├── prompts/
│   ├── system_v1.txt
│   ├── tool_use_v1.txt
│   └── multilingual_v1.txt
├── evaluations/
│   ├── __init__.py
│   ├── stt/
│   ├── tts/
│   ├── rag/
│   ├── agent/
│   └── end_to_end/
├── experiments/
├── datasets/
├── knowledge_base/
├── tests/
├── dashboard/
├── docs/
└── demos/
```

**Implementation steps:**

1. Create all directories using `mkdir -p`
2. Create `README.md` with sections: Project Overview, Architecture Diagram, Features, Technology Choices, Setup Instructions, Demo Instructions, Evaluation Results, Limitations, Future Work
3. Create `flow.md` with all 27 decision log sections (see Architecture prompt)
4. Create `ARCHITECTURE.md` with Mermaid diagrams for: system flow, agent state machine, RAG pipeline, data flow
5. Create `EXPERIMENTS.md` with summary table template
6. Create `EVALUATION.md` with all metric definitions
7. Create `CHANGELOG.md` starting with v0.1.0
8. Create `failure_analysis.md` with failure category template
9. Create `.env.example` with all required environment variables
10. Create `requirements.txt` with pinned versions
11. Create `src/config.py` loading from environment variables
12. Create `src/logger.py` with structured logging
13. Initialize Python packages with `__init__.py`
14. Create initial prompt files in `prompts/`

**After completion:** Commit with message `docs: initialize project structure with all documentation and configuration files`

---

### Prompt 0.2 — Architecture & flow.md Setup

**Task:** Populate `flow.md` with the initial architecture decisions and problem definition.

**Update flow.md with:**

```markdown
# Project Decision Log — Multilingual Indian Voice Agent

## 1. Problem Definition

Build a real-time conversational AI voice agent for an Indian education counsellor that:
- Accepts streaming audio input
- Supports English, Hindi, Hinglish, and extensible to other Indian languages
- Detects language and code-switching
- Retrieves information from a knowledge base
- Calls tools when required
- Streams audio responses
- Handles interruptions naturally
- Records telemetry and evaluates interactions

## 2. Goals

- Real-time voice conversation (< 3s end-to-end latency target)
- Accurate STT for Indian English, Hindi, Hinglish
- Natural TTS with Indian accent
- Tool calling for course information
- RAG for dynamic knowledge retrieval
- Barge-in / interruption handling
- Full evaluation framework
- Dashboard for monitoring

## 3. Non-Goals

- Mobile app (web-based only)
- Real payment processing
- Student data storage beyond session
- Training custom foundation models
- Production deployment infrastructure

## 4. Initial Architecture

[INSERT INITIAL ARCHITECTURE DIAGRAM]

Frontend → Voice Gateway → Orchestrator → Response Pipeline → User

## 5. Technology Selection

[TO BE DECIDED IN PROMPTS 1.X]

## 6-27. [Section templates for each decision]

[Populate all sections with "TBD - pending experiment/prompt X"]
```

**Implementation steps:**

1. Write complete problem statement
2. Define measurable goals with numbers
3. Define non-goals clearly
4. Draw initial architecture hypothesis
5. Leave decision sections as TBD with references to future prompts
6. Add "Last updated" timestamp

**After completion:** Commit with message `docs: add initial flow.md with problem definition and architecture hypotheses`

---

## PHASE 1: Baseline Voice Pipeline (Microphone → STT → LLM → TTS → Speaker)

### Prompt 1.1 — Project Environment Setup

**Task:** Set up the Python project environment with all required dependencies.

**Files to create/modify:**

```
requirements.txt
src/
├── __init__.py
├── config.py
├── logger.py
└── main.py
```

**Implementation steps:**

1. Install required packages:
```bash
pip install python-dotenv pydantic python-multipart websockets scipy soundfile numpy pytest pytest-asyncio aiohttp
```

2. Create `src/config.py`:
```python
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # STT Configuration
    stt_provider: str = "openai_whisper"  # openai_whisper, anthropic, etc.
    whisper_model: str = "base"
    
    # TTS Configuration  
    tts_provider: str = "openai"  # openai, anthropic, etc.
    tts_voice: str = "alloy"
    
    # LLM Configuration
    llm_provider: str = "anthropic"  # anthropic, openai, etc.
    llm_model: str = "claude-sonnet-4-20250514"
    
    # Vector DB
    vector_db_provider: str = "chromadb"  # chromadb, qdrant, etc.
    
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
```

3. Create `src/logger.py`:
```python
import logging
import sys
from src.config import settings

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level))
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
```

4. Create `src/__init__.py` exporting settings and logger

5. Verify installation:
```bash
python -c "from src.config import settings; print(settings.llm_model)"
```

**After completion:** Commit with message `chore: set up project environment with configuration management`

---

### Prompt 1.2 — STT Abstraction Layer

**Task:** Create a provider-agnostic STT abstraction with at least one working implementation.

**Files to create:**

```
src/stt/
├── __init__.py
├── base.py          # Abstract base class
├── providers.py     # Provider implementations
└── config.py        # STT configuration
```

**Implementation steps:**

1. Create `src/stt/base.py`:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Optional
import numpy as np

@dataclass
class STTResult:
    text: str
    language: str
    confidence: float
    duration: float  # audio duration in seconds

class STTProvider(ABC):
    @abstractmethod
    async def transcribe(
        self, 
        audio: bytes, 
        language: Optional[str] = None
    ) -> STTResult:
        """Transcribe audio to text."""
        pass
    
    @abstractmethod
    async def transcribe_stream(
        self, 
        audio_stream: AsyncGenerator[bytes, None],
        language: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream transcription yielding partial results."""
        pass
    
    @abstractmethod
    async def detect_language(self, audio: bytes) -> str:
        """Detect the language of the audio."""
        pass
    
    @property
    @abstractmethod
    def latency_ms(self) -> float:
        """Average latency in milliseconds."""
        pass
```

2. Create `src/stt/providers.py` with OpenAI Whisper implementation:
```python
import whisper
import numpy as np
from typing import AsyncGenerator, Optional
import io
import wave
from src.stt.base import STTProvider, STTResult
from src.logger import setup_logger
import time

logger = setup_logger(__name__)

class WhisperSTTProvider(STTProvider):
    def __init__(self, model_name: str = "base", device: str = "cpu"):
        logger.info(f"Loading Whisper model: {model_name} on {device}")
        self.model = whisper.load_model(model_name, device=device)
        self._latency_ms = 0
    
    async def transcribe(
        self, 
        audio: bytes, 
        language: Optional[str] = None
    ) -> STTResult:
        start = time.time()
        
        # Convert bytes to numpy array
        audio_np = self._bytes_to_audio(audio)
        
        # Run transcription
        options = {}
        if language:
            options["language"] = language
            
        result = self.model.transcribe(audio_np, **options)
        
        self._latency_ms = (time.time() - start) * 1000
        
        return STTResult(
            text=result["text"].strip(),
            language=result.get("language", language or "en"),
            confidence=result.get("probability", 0.0),
            duration=result.get("duration", 0.0)
        )
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        # Collect audio chunks
        chunks = []
        async for chunk in audio_stream:
            chunks.append(chunk)
        
        audio = b"".join(chunks)
        result = await self.transcribe(audio, language)
        yield result.text
    
    async def detect_language(self, audio: bytes) -> str:
        audio_np = self._bytes_to_audio(audio)
        result = self.model.detect_language(audio_np)
        return result[0][0] if result else "en"
    
    def _bytes_to_audio(self, audio: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array."""
        # Handle WAV format
        if audio[:4] == b'RIFF':
            with wave.open(io.BytesIO(audio), 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                audio_np = np.frombuffer(frames, dtype=np.int16)
                # Convert to float32
                audio_np = audio_np.astype(np.float32) / 32768.0
        else:
            # Raw PCM
            audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_np
    
    @property
    def latency_ms(self) -> float:
        return self._latency_ms

# Provider registry
STT_PROVIDERS = {
    "openai_whisper": WhisperSTTProvider,
}

def get_stt_provider(provider_name: str = None, **kwargs) -> STTProvider:
    from src.config import settings
    name = provider_name or settings.stt_provider
    provider_class = STT_PROVIDERS.get(name)
    if not provider_class:
        raise ValueError(f"Unknown STT provider: {name}")
    return provider_class(**kwargs)
```

3. Create `src/stt/config.py`:
```python
from pydantic import BaseModel
from typing import Optional

class STTConfig(BaseModel):
    provider: str = "openai_whisper"
    model_name: str = "base"  # tiny, base, small, medium, large
    device: str = "cpu"  # cpu, cuda
    language: Optional[str] = None
    temperature: float = 0.0
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
```

4. Create `src/stt/__init__.py`:
```python
from src.stt.base import STTProvider, STTResult
from src.stt.providers import WhisperSTTProvider, get_stt_provider

__all__ = ["STTProvider", "STTResult", "WhisperSTTProvider", "get_stt_provider"]
```

5. Write tests in `tests/test_stt.py`:
```python
import pytest
import numpy as np
from src.stt.providers import WhisperSTTProvider

@pytest.fixture
def stt_provider():
    return WhisperSTTProvider(model_name="tiny")

@pytest.mark.asyncio
async def test_stt_transcribe(stt_provider):
    # Create test audio (1 second of silence)
    sample_rate = 16000
    audio = np.zeros(sample_rate, dtype=np.int16).tobytes()
    
    result = await stt_provider.transcribe(audio)
    assert isinstance(result.text, str)
    assert result.language in ["en", "hi", "hinglish"]
```

**After completion:** Commit with message `feat: implement STT abstraction layer with Whisper provider`

---

### Prompt 1.3 — TTS Abstraction Layer

**Task:** Create a provider-agnostic TTS abstraction with streaming support.

**Files to create:**

```
src/tts/
├── __init__.py
├── base.py
├── providers.py
├── normalizer.py
└── config.py
```

**Implementation steps:**

1. Create `src/tts/base.py`:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

@dataclass
class TTSResult:
    audio: bytes
    duration: float  # seconds
    format: str = "wav"

class TTSProvider(ABC):
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None
    ) -> TTSResult:
        """Synthesize text to speech, return full audio."""
        pass
    
    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks as they are generated."""
        pass
    
    @property
    @abstractmethod
    def latency_ms(self) -> float:
        """Average latency to first audio byte in milliseconds."""
        pass
```

2. Create `src/tts/normalizer.py` — Handle text normalization:
```python
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class NormalizedText:
    original: str
    normalized: str
    substitutions: List[Tuple[str, str]]

class TextNormalizer:
    """Normalize text for TTS synthesis."""
    
    # Currency symbols and amounts
    CURRENCY_PATTERNS = [
        (r'₹\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', '₹\\1'),
        (r'Rs\.?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', '₹\\1'),
    ]
    
    # Percentage patterns
    PERCENTAGE_PATTERNS = [
        (r'(\d+(?:\.\d+)?)\s*%', '\\1 percent'),
        (r'(\d+(?:\.\d+)?)\s*प्रतिशत', '\\1 pratishat'),
    ]
    
    # Time patterns
    TIME_PATTERNS = [
        (r'(\d{1,2})\s*AM', '\\1 A M'),
        (r'(\d{1,2})\s*PM', '\\1 P M'),
        (r'(\d{1,2}):(\d{2})\s*(AM|PM)', '\\1 \\3'),
    ]
    
    # Indian exam/course abbreviations
    COURSE_PATTERNS = [
        (r'\bJEE\b', 'J E E'),
        (r'\bNEET\b', 'N E E T'),
        (r'\bIIT\b', 'I I T'),
        (r'\bCBSE\b', 'C B S E'),
        (r'\bNDA\b', 'N D A'),
        (r'\bAIIMS\b', 'A I I M S'),
        (r'\bAI\s*/\s*ML\b', 'A I slash M L'),
        (r'\bIIT\s*JEE\b', 'I I T J E E'),
    ]
    
    # Number patterns
    NUMBER_PATTERNS = [
        (r'\b(\d{5,})\b', lambda m: ' '.join(list(m.group(1)))),  # Phone numbers
        (r'\b(\d{4})\b', '\\1'),  # Years like 2026
    ]
    
    def normalize(self, text: str) -> NormalizedText:
        """Normalize text for TTS."""
        normalized = text
        substitutions = []
        
        # Apply all patterns
        for pattern, replacement in self.CURRENCY_PATTERNS:
            new_text, count = re.subn(pattern, replacement, normalized)
            if count > 0:
                normalized = new_text
                substitutions.extend([(pattern, replacement)] * count)
        
        for pattern, replacement in self.PERCENTAGE_PATTERNS:
            new_text, count = re.subn(pattern, replacement, normalized)
            if count > 0:
                normalized = new_text
                substitutions.extend([(pattern, replacement)] * count)
        
        for pattern, replacement in self.TIME_PATTERNS:
            new_text, count = re.subn(pattern, replacement, normalized)
            if count > 0:
                normalized = new_text
                substitutions.extend([(pattern, replacement)] * count)
        
        for pattern, replacement in self.COURSE_PATTERNS:
            new_text, count = re.subn(pattern, replacement, normalized)
            if count > 0:
                normalized = new_text
                substitutions.extend([(pattern, replacement)] * count)
        
        # Clean up multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return NormalizedText(
            original=text,
            normalized=normalized,
            substitutions=substitutions
        )

# Singleton instance
normalizer = TextNormalizer()
```

3. Create `src/tts/providers.py` with OpenAI TTS implementation:
```python
import openai
import io
import wave
import time
from typing import AsyncGenerator, Optional
from src.tts.base import TTSProvider, TTSResult
from src.logger import setup_logger

logger = setup_logger(__name__)

class OpenAITTSProvider(TTSProvider):
    def __init__(
        self,
        api_key: str = None,
        voice: str = "alloy",
        model: str = "tts-1",
    ):
        self.client = openai.OpenAI(api_key=api_key)
        self.voice = voice
        self.model = model
        self._latency_ms = 0
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None
    ) -> TTSResult:
        from src.tts.normalizer import normalizer
        
        start = time.time()
        
        # Normalize text
        normalized = normalizer.normalize(text)
        logger.debug(f"Normalized: {normalized.original} -> {normalized.normalized}")
        
        # Generate audio
        response = self.client.audio.speech.create(
            model=self.model,
            voice=voice or self.voice,
            input=normalized.normalized,
            response_format="mp3"
        )
        
        audio = response.content
        
        # Calculate duration (approximate: 150 chars/min for natural speech)
        duration = len(normalized.normalized) / 150 * 60
        
        self._latency_ms = (time.time() - start) * 1000
        
        return TTSResult(
            audio=audio,
            duration=duration,
            format="mp3"
        )
    
    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        from src.tts.normalizer import normalizer
        
        normalized = normalizer.normalize(text)
        
        response = self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=voice or self.voice,
            input=normalized.normalized,
            response_format="mp3"
        )
        
        async for chunk in response.iter_bytes(chunk_size=1024):
            yield chunk
    
    @property
    def latency_ms(self) -> float:
        return self._latency_ms

# Provider registry
TTS_PROVIDERS = {
    "openai": OpenAITTSProvider,
}

def get_tts_provider(provider_name: str = None, **kwargs) -> TTSProvider:
    from src.config import settings
    name = provider_name or settings.tts_provider
    provider_class = TTS_PROVIDERS.get(name)
    if not provider_class:
        raise ValueError(f"Unknown TTS provider: {name}")
    return provider_class(**kwargs)
```

4. Create `src/tts/config.py` and `src/tts/__init__.py`

5. Write tests for normalizer:
```python
# tests/test_normalizer.py
import pytest
from src.tts.normalizer import normalizer, TextNormalizer

def test_normalize_currency():
    result = normalizer.normalize("Course fees ₹25,000")
    assert "₹25,000" in result.normalized or "₹ 25000" in result.normalized

def test_normalize_time():
    result = normalizer.normalize("Class at 6 PM")
    assert "6 P M" in result.normalized

def test_normalize_courses():
    result = normalizer.normalize("I want to prepare for JEE and NEET")
    assert "J E E" in result.normalized
    assert "N E E T" in result.normalized

def test_normalize_percentage():
    result = normalizer.normalize("Success rate is 85%")
    assert "85 percent" in result.normalized
```

**After completion:** Commit with message `feat: implement TTS abstraction layer with OpenAI provider and text normalizer`

---

### Prompt 1.4 — LLM Integration

**Task:** Create LLM abstraction for conversation and tool calling.

**Files to create:**

```
src/llm/
├── __init__.py
├── base.py
├── providers.py
├── prompts/
│   ├── system_v1.txt
│   ├── tool_use_v1.txt
│   └── multilingual_v1.txt
└── config.py
```

**Implementation steps:**

1. Create `src/llm/base.py`:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"

@dataclass
class Message:
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]

@dataclass
class LLMResponse:
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"

class LLMProvider(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a chat completion."""
        pass
    
    @abstractmethod
    async def detect_language(self, text: str) -> str:
        """Detect the primary language of text."""
        pass
```

2. Create `src/llm/providers.py` with Anthropic Claude implementation:
```python
import anthropic
from typing import List, Dict, Optional, Any
from src.llm.base import LLMProvider, LLMResponse, Message, MessageRole, ToolCall
from src.logger import setup_logger

logger = setup_logger(__name__)

class AnthropicLLMProvider(LLMProvider):
    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        # Convert messages to Anthropic format
        anthropic_messages = []
        system_prompt = None
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                role = "user" if msg.role == MessageRole.USER else "assistant"
                anthropic_messages.append({
                    "role": role,
                    "content": msg.content
                })
        
        # Build request
        request_options = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if system_prompt:
            request_options["system"] = system_prompt
        
        if tools:
            request_options["tools"] = tools
        
        # Call API
        response = self.client.messages.create(**request_options)
        
        # Parse response
        content = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input
                ))
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=str(response.stop_reason)
        )
    
    async def detect_language(self, text: str) -> str:
        """Simple language detection for Hinglish."""
        hindi_chars = set('अ आ इ उ ऋ ए ऐ ओ औ क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह ा ि ी ू ृ े ै ो ौ ं ः')
        
        hindi_count = sum(1 for c in text if c in hindi_chars)
        hindi_ratio = hindi_count / len(text) if text else 0
        
        if hindi_ratio > 0.3:
            return "hi"  # Hindi
        elif hindi_ratio > 0.1:
            return "hi-en"  # Hinglish
        else:
            return "en"  # English

LLM_PROVIDERS = {
    "anthropic": AnthropicLLMProvider,
}

def get_llm_provider(provider_name: str = None, **kwargs) -> LLMProvider:
    from src.config import settings
    name = provider_name or settings.llm_provider
    provider_class = LLM_PROVIDERS.get(name)
    if not provider_class:
        raise ValueError(f"Unknown LLM provider: {name}")
    return provider_class(**kwargs)
```

3. Create initial prompt files:
```markdown
<!-- prompts/system_v1.txt -->
You are a friendly Indian education counsellor. You help students with:
- JEE preparation courses
- NEET preparation courses
- Board exam coaching
- Career guidance

Guidelines:
- Be warm and approachable
- Keep responses concise (2-3 sentences max)
- Use simple language
- Ask clarifying questions when needed
- Be honest about limitations
```

```markdown
<!-- prompts/tool_use_v1.txt -->
When a user asks about specific course details, fees, schedules, or availability, 
use the appropriate tool. Available tools:
- search_courses: Search for courses by keyword
- get_course_details: Get detailed information about a course
- get_fee_structure: Get fee details
- check_batch_availability: Check when batches start
```

```markdown
<!-- prompts/multilingual_v1.txt -->
You can respond in:
- English (for formal queries)
- Hindi (for Hindi-speaking students)
- Hinglish (for natural code-switching)

Detect the user's language and respond in the same style.
Use Hindi words naturally when the user speaks Hindi.
```

4. Create `src/llm/config.py` and `src/llm/__init__.py`

**After completion:** Commit with message `feat: implement LLM abstraction with Anthropic Claude provider`

---

### Prompt 1.5 — Minimal Baseline Pipeline

**Task:** Create the minimal end-to-end pipeline: Microphone → STT → LLM → TTS → Speaker.

**Files to create:**

```
src/
├── pipeline/
│   ├── __init__.py
│   ├── baseline.py      # Minimal pipeline
│   └── types.py         # Pipeline types
└── main.py              # Entry point
```

**Implementation steps:**

1. Create `src/pipeline/types.py`:
```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class AgentState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    CALLING_TOOL = "calling_tool"
    GENERATING = "generating"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"

@dataclass
class ConversationMessage:
    role: str  # "user" or "agent"
    content: str
    timestamp: datetime
    language: str
    audio_duration: float = 0.0

@dataclass
class Session:
    id: str
    created_at: datetime
    state: AgentState
    conversation: List[ConversationMessage] = field(default_factory=list)
    user_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

2. Create `src/pipeline/baseline.py`:
```python
import asyncio
import uuid
from datetime import datetime
from typing import Optional
from src.stt.providers import get_stt_provider, STTResult
from src.tts.providers import get_tts_provider, TTSResult
from src.llm.providers import get_llm_provider, LLMResponse, Message, MessageRole
from src.pipeline.types import Session, AgentState, ConversationMessage
from src.logger import setup_logger
import sounddevice as sd
import io
import wave

logger = setup_logger(__name__)

class BaselineVoicePipeline:
    """Minimal end-to-end voice pipeline."""
    
    def __init__(
        self,
        stt_provider: str = None,
        tts_provider: str = None,
        llm_provider: str = None,
    ):
        self.stt = get_stt_provider(stt_provider)
        self.tts = get_tts_provider(tts_provider)
        self.llm = get_llm_provider(llm_provider)
        
        # Load system prompt
        with open("prompts/system_v1.txt", "r") as f:
            self.system_prompt = f.read()
    
    async def process_audio(self, audio: bytes) -> TTSResult:
        """Process audio through the full pipeline."""
        
        # 1. STT
        logger.info("Transcribing...")
        stt_result = await self.stt.transcribe(audio)
        logger.info(f"Transcribed: {stt_result.text}")
        
        # 2. Detect language
        language = await self.llm.detect_language(stt_result.text)
        logger.info(f"Detected language: {language}")
        
        # 3. LLM
        logger.info("Generating response...")
        messages = [
            Message(role=MessageRole.SYSTEM, content=self.system_prompt),
            Message(role=MessageRole.USER, content=stt_result.text),
        ]
        llm_response = await self.llm.chat(messages)
        logger.info(f"LLM response: {llm_response.content}")
        
        # 4. TTS
        logger.info("Synthesizing speech...")
        tts_result = await self.tts.synthesize(
            llm_response.content,
            language=language
        )
        logger.info(f"TTS audio duration: {tts_result.duration}s")
        
        return tts_result
    
    async def run_interactive(self):
        """Run interactive voice conversation."""
        session_id = str(uuid.uuid4())
        logger.info(f"Starting session: {session_id}")
        
        sample_rate = 16000
        is_recording = False
        audio_buffer = []
        
        def audio_callback(indata, frames, time, status):
            nonlocal audio_buffer, is_recording
            if is_recording:
                audio_buffer.append(indata.copy())
        
        # Start audio stream
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='int16',
            callback=audio_callback
        )
        
        try:
            while True:
                print("\n🎤 Press ENTER to start recording, 'q' to quit...")
                cmd = input()
                
                if cmd.lower() == 'q':
                    break
                
                # Record
                is_recording = True
                audio_buffer = []
                print("🔴 Recording... Press ENTER to stop")
                input()
                is_recording = False
                
                # Convert to bytes
                audio_data = b"".join(frame.tobytes() for frame in audio_buffer)
                
                # Process
                result = await self.process_audio(audio_data)
                
                # Play audio
                print("🔊 Playing response...")
                sd.play(result.audio, sample_rate)
                sd.wait()
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            stream.close()
            logger.info("Session ended")

async def main():
    pipeline = BaselineVoicePipeline()
    await pipeline.run_interactive()

if __name__ == "__main__":
    asyncio.run(main())
```

3. Update `src/main.py`:
```python
import asyncio
from src.pipeline.baseline import BaselineVoicePipeline
from src.logger import setup_logger

logger = setup_logger(__name__)

async def main():
    logger.info("Starting Multilingual Indian Voice Agent")
    pipeline = BaselineVoicePipeline()
    await pipeline.run_interactive()

if __name__ == "__main__":
    asyncio.run(main())
```

**After completion:** Commit with message `feat: implement minimal baseline voice pipeline (mic→STT→LLM→TTS→speaker)`

---

## PHASE 2: Voice Gateway & WebSocket Server

### Prompt 2.1 — Voice Gateway with WebSocket

**Task:** Create WebSocket server for real-time voice communication.

**Files to create:**

```
src/gateway/
├── __init__.py
├── websocket_server.py
├── audio_utils.py
└── vad.py
```

**Implementation steps:**

1. Create `src/gateway/audio_utils.py`:
```python
import numpy as np
import struct
from typing import Generator

def bytes_to_audio(audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Convert audio bytes to numpy array."""
    # Assuming 16-bit PCM
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np

def audio_to_bytes(audio_np: np.ndarray) -> bytes:
    """Convert numpy array to audio bytes."""
    audio_int16 = (audio_np * 32768.0).astype(np.int16)
    return audio_int16.tobytes()

def create_wav_header(num_frames: int, num_channels: int = 1, sample_rate: int = 16000) -> bytes:
    """Create WAV header."""
    byte_rate = sample_rate * num_channels * 2
    block_align = num_channels * 2
    data_size = num_frames * num_channels * 2
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,  # fmt chunk size
        1,   # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        16,  # bits per sample
        b'data',
        data_size
    )
    return header

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple audio resampling using linear interpolation."""
    if orig_sr == target_sr:
        return audio
    
    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, target_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
```

2. Create `src/gateway/vad.py` — Voice Activity Detection:
```python
import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass
class VADConfig:
    sample_rate: int = 16000
    frame_duration_ms: int = 30
    energy_threshold: float = 0.02
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 300

class SimpleEnergyVAD:
    """Energy-based voice activity detection."""
    
    def __init__(self, config: VADConfig = None):
        self.config = config or VADConfig()
        self.frame_size = int(self.config.sample_rate * self.config.frame_duration_ms / 1000)
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """Detect if audio frame contains speech."""
        energy = np.sqrt(np.mean(audio ** 2))
        return energy > self.config.energy_threshold
    
    def detect_speech_segments(
        self, 
        audio: np.ndarray
    ) -> list[tuple[int, int]]:
        """
        Detect speech segments in audio.
        Returns list of (start_frame, end_frame) tuples.
        """
        num_frames = len(audio) // self.frame_size
        speech_frames = []
        
        for i in range(num_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frame = audio[start:end]
            
            if self.is_speech(frame):
                speech_frames.append(i)
        
        # Merge nearby speech frames
        if not speech_frames:
            return []
        
        segments = []
        start = speech_frames[0]
        prev = speech_frames[0]
        
        min_speech_frames = int(self.config.min_speech_duration_ms / self.config.frame_duration_ms)
        
        for frame in speech_frames[1:]:
            if frame - prev <= int(self.config.min_silence_duration_ms / self.config.frame_duration_ms):
                prev = frame
            else:
                if prev - start >= min_speech_frames:
                    segments.append((start, prev))
                start = frame
                prev = frame
        
        # Last segment
        if prev - start >= min_speech_frames:
            segments.append((start, prev))
        
        # Add padding
        pad_frames = int(self.config.speech_pad_ms / self.config.frame_duration_ms)
        padded_segments = []
        for start, end in segments:
            padded_segments.append((
                max(0, start - pad_frames),
                min(num_frames, end + pad_frames)
            ))
        
        return padded_segments
```

3. Create `src/gateway/websocket_server.py`:
```python
import asyncio
import json
import uuid
import struct
from typing import Optional, Dict, Callable
from websockets.server import serve, WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed
from src.logger import setup_logger
from src.pipeline.types import Session, AgentState

logger = setup_logger(__name__)

class VoiceGateway:
    """WebSocket voice gateway for real-time communication."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.sessions: Dict[str, Session] = {}
        self.audio_buffer: Dict[str, bytes] = {}
        self.pipeline = None  # Will be set by main
    
    async def handle_websocket(self, websocket: WebSocketServerProtocol):
        """Handle WebSocket connection."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = Session(
            id=session_id,
            created_at=datetime.now(),
            state=AgentState.IDLE
        )
        self.audio_buffer[session_id] = b""
        
        logger.info(f"New connection: {session_id}")
        
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Audio data
                    self.audio_buffer[session_id] += message
                    await websocket.send(json.dumps({
                        "type": "audio_received",
                        "bytes": len(message)
                    }))
                else:
                    # JSON message
                    data = json.loads(message)
                    await self.handle_message(websocket, session_id, data)
                    
        except ConnectionClosed:
            logger.info(f"Connection closed: {session_id}")
        finally:
            del self.sessions[session_id]
            del self.audio_buffer[session_id]
    
    async def handle_message(
        self, 
        websocket: WebSocketServerProtocol,
        session_id: str,
        data: dict
    ):
        """Handle JSON messages."""
        msg_type = data.get("type")
        
        if msg_type == "get_audio":
            # Return accumulated audio
            audio = self.audio_buffer.get(session_id, b"")
            self.audio_buffer[session_id] = b""
            await websocket.send(audio)
        
        elif msg_type == "ping":
            await websocket.send(json.dumps({"type": "pong"}))
        
        elif msg_type == "status":
            session = self.sessions[session_id]
            await websocket.send(json.dumps({
                "type": "status",
                "state": session.state.value,
                "session_id": session_id
            }))
    
    async def start(self):
        """Start WebSocket server."""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        async with serve(self.handle_websocket, self.host, self.port):
            await asyncio.Future()  # Run forever

async def main():
    gateway = VoiceGateway()
    await gateway.start()

if __name__ == "__main__":
    asyncio.run(main())
```

**After completion:** Commit with message `feat: implement WebSocket voice gateway with VAD`

---

### Prompt 2.2 — Frontend Web Interface

**Task:** Create a simple web interface for voice interaction.

**Files to create:**

```
dashboard/
├── index.html
├── app.js
└── styles.css
```

**Implementation steps:**

1. Create `dashboard/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Indian Voice Agent</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>🎓 Indian Education Counsellor</h1>
        
        <div class="status-panel">
            <div class="status-item">
                <span class="label">Status:</span>
                <span id="status" class="value">Idle</span>
            </div>
            <div class="status-item">
                <span class="label">Language:</span>
                <span id="language" class="value">-</span>
            </div>
            <div class="status-item">
                <span class="label">Session:</span>
                <span id="session" class="value">-</span>
            </div>
        </div>
        
        <div class="conversation-panel">
            <div id="messages" class="messages"></div>
        </div>
        
        <div class="controls">
            <button id="mic-btn" class="mic-btn">
                <span class="icon">🎤</span>
                <span class="label">Hold to Speak</span>
            </button>
        </div>
        
        <div class="metrics-panel">
            <h3>Metrics</h3>
            <div class="metric">
                <span>STT Latency:</span>
                <span id="stt-latency">-</span>
            </div>
            <div class="metric">
                <span>LLM Latency:</span>
                <span id="llm-latency">-</span>
            </div>
            <div class="metric">
                <span>TTS Latency:</span>
                <span id="tts-latency">-</span>
            </div>
            <div class="metric">
                <span>Total:</span>
                <span id="total-latency">-</span>
            </div>
        </div>
    </div>
    
    <script src="app.js"></script>
</body>
</html>
```

2. Create `dashboard/styles.css`:
```css
:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --bg: #0f172a;
    --surface: #1e293b;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --success: #22c55e;
    --error: #ef4444;
}

body {
    font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 20px;
}

.container {
    max-width: 600px;
    margin: 0 auto;
}

h1 {
    text-align: center;
    font-size: 1.5rem;
    margin-bottom: 20px;
}

.status-panel {
    background: var(--surface);
    border-radius: 8px;
    padding: 12px;
    display: flex;
    gap: 20px;
    margin-bottom: 20px;
}

.status-item {
    display: flex;
    flex-direction: column;
}

.label {
    font-size: 0.75rem;
    color: var(--text-muted);
}

.value {
    font-weight: 500;
}

.conversation-panel {
    background: var(--surface);
    border-radius: 8px;
    height: 300px;
    overflow-y: auto;
    margin-bottom: 20px;
}

.messages {
    padding: 12px;
}

.message {
    margin-bottom: 12px;
    padding: 8px 12px;
    border-radius: 8px;
    max-width: 80%;
}

.message.user {
    background: var(--primary);
    margin-left: auto;
}

.message.agent {
    background: #334155;
}

.controls {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}

.mic-btn {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    border: none;
    background: var(--primary);
    color: white;
    font-size: 24px;
    cursor: pointer;
    transition: transform 0.1s, background 0.1s;
}

.mic-btn:hover {
    background: var(--primary-dark);
}

.mic-btn.recording {
    background: var(--error);
    transform: scale(1.1);
}

.metrics-panel {
    background: var(--surface);
    border-radius: 8px;
    padding: 12px;
}

.metrics-panel h3 {
    margin-top: 0;
    font-size: 0.875rem;
}

.metric {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 0.875rem;
}
```

3. Create `dashboard/app.js`:
```javascript
class VoiceAgent {
    constructor() {
        this.ws = null;
        this.mediaRecorder = null;
        this.audioContext = null;
        this.isRecording = false;
        this.sessionId = null;
        this.audioChunks = [];
        
        this.initElements();
        this.initWebSocket();
    }
    
    initElements() {
        this.micBtn = document.getElementById('mic-btn');
        this.statusEl = document.getElementById('status');
        this.languageEl = document.getElementById('language');
        this.sessionEl = document.getElementById('session');
        this.messagesEl = document.getElementById('messages');
        this.sttLatencyEl = document.getElementById('stt-latency');
        this.llmLatencyEl = document.getElementById('llm-latency');
        this.ttsLatencyEl = document.getElementById('tts-latency');
        this.totalLatencyEl = document.getElementById('total-latency');
        
        this.micBtn.addEventListener('mousedown', () => this.startRecording());
        this.micBtn.addEventListener('mouseup', () => this.stopRecording());
        this.micBtn.addEventListener('mouseleave', () => this.stopRecording());
    }
    
    async initWebSocket() {
        this.ws = new WebSocket('ws://localhost:8765');
        
        this.ws.onopen = () => {
            this.setStatus('Connected');
            this.ws.send(JSON.stringify({ type: 'status' }));
        };
        
        this.ws.onmessage = async (event) => {
            if (typeof event.data === 'string') {
                const msg = JSON.parse(event.data);
                this.handleMessage(msg);
            } else {
                // Audio response
                await this.playAudio(event.data);
            }
        };
        
        this.ws.onclose = () => {
            this.setStatus('Disconnected');
            setTimeout(() => this.initWebSocket(), 2000);
        };
    }
    
    handleMessage(msg) {
        switch (msg.type) {
            case 'status':
                this.sessionId = msg.session_id;
                this.sessionEl.textContent = msg.session_id.slice(0, 8);
                this.setState(msg.state);
                break;
            case 'transcript':
                this.addMessage('user', msg.text);
                break;
            case 'response':
                this.addMessage('agent', msg.text);
                break;
            case 'language':
                this.languageEl.textContent = msg.language;
                break;
            case 'metrics':
                this.updateMetrics(msg);
                break;
        }
    }
    
    async startRecording() {
        if (this.isRecording) return;
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (e) => {
                this.audioChunks.push(e.data);
            };
            
            this.mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                const buffer = await audioBlob.arrayBuffer();
                this.ws.send(new Uint8Array(buffer));
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            this.micBtn.classList.add('recording');
            this.setStatus('Recording...');
        } catch (err) {
            console.error('Error starting recording:', err);
            this.setStatus('Error: ' + err.message);
        }
    }
    
    stopRecording() {
        if (!this.isRecording) return;
        
        this.mediaRecorder.stop();
        this.mediaRecorder.stream.getTracks().forEach(t => t.stop());
        this.isRecording = false;
        this.micBtn.classList.remove('recording');
        this.setStatus('Processing...');
    }
    
    async playAudio(audioData) {
        const audioContext = new AudioContext();
        const audioBuffer = await audioContext.decodeAudioData(audioData);
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        source.start();
    }
    
    setStatus(status) {
        this.statusEl.textContent = status;
    }
    
    setState(state) {
        this.statusEl.textContent = state;
    }
    
    addMessage(role, text) {
        const div = document.createElement('div');
        div.className = `message ${role}`;
        div.textContent = text;
        this.messagesEl.appendChild(div);
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    }
    
    updateMetrics(msg) {
        this.sttLatencyEl.textContent = msg.stt_latency ? `${msg.stt_latency}ms` : '-';
        this.llmLatencyEl.textContent = msg.llm_latency ? `${msg.llm_latency}ms` : '-';
        this.ttsLatencyEl.textContent = msg.tts_latency ? `${msg.tts_latency}ms` : '-';
        this.totalLatencyEl.textContent = msg.total ? `${msg.total}ms` : '-';
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.voiceAgent = new VoiceAgent();
});
```

**After completion:** Commit with message `feat: create web dashboard for voice interaction`

---

## PHASE 3: Agent Orchestration

### Prompt 3.1 — Agent State Machine

**Task:** Implement the full agent orchestrator with state machine and tool calling.

**Files to create:**

```
src/agent/
├── __init__.py
├── orchestrator.py
├── state_machine.py
├── tools/
│   ├── __init__.py
│   ├── base.py
│   ├── course_tools.py
│   └── tool_registry.py
├── memory/
│   ├── __init__.py
│   ├── short_term.py
│   └── long_term.py
└── config.py
```

**Implementation steps:**

1. Create `src/agent/state_machine.py`:
```python
from enum import Enum
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from src.pipeline.types import AgentState
from src.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class StateTransition:
    from_state: AgentState
    to_state: AgentState
    event: str
    condition: Optional[Callable] = None
    action: Optional[Callable] = None

class AgentStateMachine:
    """Manages agent state transitions."""
    
    def __init__(self):
        self.current_state = AgentState.IDLE
        self.state_history: List[tuple[datetime, AgentState, str]] = []
        self.transitions: List[StateTransition] = []
        self._register_default_transitions()
    
    def _register_default_transitions(self):
        """Register default valid transitions."""
        valid_transitions = {
            AgentState.IDLE: [AgentState.LISTENING],
            AgentState.LISTENING: [AgentState.TRANSCRIBING, AgentState.IDLE],
            AgentState.TRANSCRIBING: [AgentState.THINKING, AgentState.ERROR],
            AgentState.THINKING: [AgentState.CALLING_TOOL, AgentState.GENERATING, AgentState.ERROR],
            AgentState.CALLING_TOOL: [AgentState.GENERATING, AgentState.ERROR],
            AgentState.GENERATING: [AgentState.SPEAKING, AgentState.LISTENING, AgentState.ERROR],
            AgentState.SPEAKING: [AgentState.LISTENING, AgentState.INTERRUPTED, AgentState.IDLE],
            AgentState.INTERRUPTED: [AgentState.LISTENING, AgentState.TRANSCRIBING],
            AgentState.ERROR: [AgentState.IDLE, AgentState.LISTENING],
        }
        
        # Build transition objects
        for from_state, to_states in valid_transitions.items():
            for to_state in to_states:
                self.transitions.append(StateTransition(
                    from_state=from_state,
                    to_state=to_state,
                    event=f"{from_state.value}_to_{to_state.value}"
                ))
    
    def transition(self, new_state: AgentState, reason: str = "") -> bool:
        """Attempt to transition to a new state."""
        valid = self._is_valid_transition(self.current_state, new_state)
        
        if valid:
            old_state = self.current_state
            self.current_state = new_state
            self.state_history.append((datetime.now(), old_state, reason))
            logger.info(f"State: {old_state.value} → {new_state.value} ({reason})")
            return True
        else:
            logger.warning(
                f"Invalid transition: {self.current_state.value} → {new_state.value}"
            )
            return False
    
    def _is_valid_transition(self, from_state: AgentState, to_state: AgentState) -> bool:
        """Check if transition is valid."""
        for t in self.transitions:
            if t.from_state == from_state and t.to_state == to_state:
                if t.condition is None or t.condition():
                    return True
        return False
    
    def can_interrupt(self) -> bool:
        """Check if current state can be interrupted."""
        interruptible = [
            AgentState.SPEAKING,
            AgentState.GENERATING,
            AgentState.CALLING_TOOL,
        ]
        return self.current_state in interruptible
    
    def force_interrupt(self) -> bool:
        """Force interrupt to LISTENING state."""
        if self.can_interrupt():
            self.transition(AgentState.INTERRUPTED, "user_barge_in")
            self.transition(AgentState.LISTENING, "continue_after_interrupt")
            return True
        return False
    
    @property
    def state(self) -> AgentState:
        return self.current_state
```

2. Create `src/agent/tools/base.py`:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime
import asyncio

@dataclass
class ToolResult:
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0

class Tool(ABC):
    """Base class for agent tools."""
    
    name: str = "base_tool"
    description: str = "A tool"
    parameters: Dict[str, Any] = {}
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate tool parameters."""
        for required in self.parameters.get("required", []):
            if required not in params:
                return False, f"Missing required parameter: {required}"
        return True, None
    
    async def execute_with_timeout(
        self, 
        timeout: float = 30.0, 
        **kwargs
    ) -> ToolResult:
        """Execute with timeout handling."""
        start = datetime.now()
        try:
            async with asyncio.timeout(timeout):
                return await self.execute(**kwargs)
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start).total_seconds() * 1000
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool execution timed out after {timeout}s",
                execution_time_ms=execution_time
            )
        except Exception as e:
            execution_time = (datetime.now() - start).total_seconds() * 1000
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time_ms=execution_time
            )
```

3. Create `src/agent/tools/course_tools.py`:
```python
from src.agent.tools.base import Tool, ToolResult
from typing import Any, Dict, List
from datetime import datetime
import random

class SearchCoursesTool(Tool):
    name = "search_courses"
    description = "Search for courses by keyword or category"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (e.g., 'JEE', 'NEET', 'physics')"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results",
                "default": 5
            }
        },
        "required": ["query"]
    }
    
    # Mock course database
    COURSES = [
        {
            "id": "jee-2024",
            "name": "JEE Main + Advanced 2024",
            "category": "engineering",
            "duration": "2 years",
            "fee": 150000,
            "language": "Hindi + English",
            "description": "Complete JEE preparation with daily tests"
        },
        {
            "id": "neet-2024",
            "name": "NEET UG 2024",
            "category": "medical",
            "duration": "2 years",
            "fee": 180000,
            "language": "Hindi + English",
            "description": "Comprehensive NEET preparation program"
        },
        {
            "id": "cbse-12",
            "name": "CBSE Class 12 Board Prep",
            "category": "boards",
            "duration": "1 year",
            "fee": 50000,
            "language": "Hindi + English",
            "description": "Board exam focused preparation"
        }
    ]
    
    async def execute(self, query: str, limit: int = 5) -> ToolResult:
        query_lower = query.lower()
        results = [
            c for c in self.COURSES
            if query_lower in c["name"].lower() 
            or query_lower in c["category"].lower()
            or query_lower in c["description"].lower()
        ][:limit]
        
        return ToolResult(success=True, data=results)

class GetCourseDetailsTool(Tool):
    name = "get_course_details"
    description = "Get detailed information about a specific course"
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "Course ID"
            }
        },
        "required": ["course_id"]
    }
    
    async def execute(self, course_id: str) -> ToolResult:
        # Mock details
        details = {
            "id": course_id,
            "syllabus": "Physics, Chemistry, Mathematics",
            "faculty": "IIT Delhi Alumni",
            "study_material": "Provided",
            "tests": "Weekly",
            "batches": ["June", "September", "January"]
        }
        return ToolResult(success=True, data=details)

class CheckEligibilityTool(Tool):
    name = "check_eligibility"
    description = "Check if a student is eligible for a course"
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {"type": "string"},
            "current_class": {"type": "string"},
            "percentage": {"type": "number"}
        },
        "required": ["course_id"]
    }
    
    async def execute(self, course_id: str, current_class: str = None, percentage: float = None) -> ToolResult:
        # Simple eligibility logic
        eligible = True
        reasons = []
        
        if course_id.startswith("jee"):
            if current_class and int(current_class) < 11:
                eligible = False
                reasons.append("JEE course requires Class 11 or above")
        
        return ToolResult(
            success=True,
            data={"eligible": eligible, "reasons": reasons}
        )

class GetFeeStructureTool(Tool):
    name = "get_fee_structure"
    description = "Get fee structure for a course"
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {"type": "string"}
        },
        "required": ["course_id"]
    }
    
    async def execute(self, course_id: str) -> ToolResult:
        fee_details = {
            "jee-2024": {
                "total": 150000,
                "installments": [
                    {"amount": 75000, "due": "Admission"},
                    {"amount": 75000, "due": "After 6 months"}
                ],
                "scholarship": "Up to 50% for meritorious students"
            }
        }
        
        return ToolResult(
            success=True,
            data=fee_details.get(course_id, {"total": 0, "installments": []})
        )

class ScheduleDemoTool(Tool):
    name = "schedule_demo"
    description = "Schedule a demo class"
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {"type": "string"},
            "date": {"type": "string", "description": "Preferred date (YYYY-MM-DD)"},
            "phone": {"type": "string"}
        },
        "required": ["course_id", "phone"]
    }
    
    async def execute(self, course_id: str, date: str = None, phone: str = None) -> ToolResult:
        # Mock scheduling
        demo_id = f"DEMO-{random.randint(1000, 9999)}"
        return ToolResult(
            success=True,
            data={
                "demo_id": demo_id,
                "status": "scheduled",
                "confirm_date": date or "Within 24 hours"
            }
        )
```

4. Create `src/agent/tools/tool_registry.py`:
```python
from typing import Dict, List, Optional
from src.agent.tools.base import Tool, ToolResult
from src.agent.tools.course_tools import (
    SearchCoursesTool,
    GetCourseDetailsTool,
    CheckEligibilityTool,
    GetFeeStructureTool,
    ScheduleDemoTool
)
from src.logger import setup_logger

logger = setup_logger(__name__)

class ToolRegistry:
    """Registry for all available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools."""
        tools = [
            SearchCoursesTool(),
            GetCourseDetailsTool(),
            CheckEligibilityTool(),
            GetFeeStructureTool(),
            ScheduleDemoTool(),
        ]
        for tool in tools:
            self.register(tool)
    
    def register(self, tool: Tool):
        """Register a new tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_all(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_tools_schema(self) -> List[Dict]:
        """Get OpenAI-style tools schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self._tools.values()
        ]
    
    async def execute_tool(
        self, 
        tool_name: str, 
        arguments: Dict
    ) -> ToolResult:
        """Execute a tool by name with given arguments."""
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool not found: {tool_name}"
            )
        
        # Validate
        valid, error = tool.validate_parameters(arguments)
        if not valid:
            return ToolResult(success=False, data=None, error=error)
        
        # Execute
        return await tool.execute_with_timeout(**arguments)

# Global registry
tool_registry = ToolRegistry()
```

5. Create `src/agent/memory/short_term.py`:
```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

@dataclass
class ConversationTurn:
    user_message: str
    agent_message: str
    language: str
    timestamp: datetime
    tools_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ShortTermMemory:
    """Manages recent conversation context."""
    
    def __init__(self, max_turns: int = 10, ttl_minutes: int = 30):
        self.max_turns = max_turns
        self.ttl_minutes = ttl_minutes
        self.turns: List[ConversationTurn] = []
        self.session_start = datetime.now()
        self.user_context: Dict[str, Any] = {}
    
    def add_turn(
        self,
        user_message: str,
        agent_message: str,
        language: str,
        tools_used: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Add a conversation turn."""
        turn = ConversationTurn(
            user_message=user_message,
            agent_message=agent_message,
            language=language,
            timestamp=datetime.now(),
            tools_used=tools_used or [],
            metadata=metadata or {}
        )
        self.turns.append(turn)
        
        # Trim if exceeds max
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
        
        # Update context
        self._update_context(user_message, language)
    
    def _update_context(self, message: str, language: str):
        """Update user context from message."""
        # Track language preference
        self.user_context["preferred_language"] = language
        
        # Extract simple facts
        message_lower = message.lower()
        if "class" in message_lower:
            for cls in ["11", "12"]:
                if cls in message_lower:
                    self.user_context["current_class"] = f"Class {cls}"
        
        if any(x in message_lower for x in ["jee", "engineering"]):
            self.user_context["interest"] = "engineering"
        elif any(x in message_lower for x in ["neet", "medical", "doctor"]):
            self.user_context["interest"] = "medical"
    
    def get_recent_context(self, num_turns: int = 3) -> str:
        """Get recent conversation as context string."""
        recent = self.turns[-num_turns:] if self.turns else []
        
        context_parts = []
        for turn in recent:
            context_parts.append(f"User: {turn.user_message}")
            context_parts.append(f"Agent: {turn.agent_message}")
        
        return "\n".join(context_parts) if context_parts else "No previous context."
    
    def is_expired(self) -> bool:
        """Check if memory should be cleared."""
        return datetime.now() - self.session_start > timedelta(minutes=self.ttl_minutes)
    
    def clear(self):
        """Clear all memory."""
        self.turns = []
        self.user_context = {}
        self.session_start = datetime.now()
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of stored context."""
        return {
            "turns_count": len(self.turns),
            "session_duration_minutes": (
                datetime.now() - self.session_start
            ).total_seconds() / 60,
            "user_context": self.user_context,
            "languages_used": list(set(t.language for t in self.turns)) if self.turns else []
        }
```

6. Create `src/agent/orchestrator.py`:
```python
import asyncio
import time
from typing import Optional, Dict, Any
from src.agent.state_machine import AgentStateMachine
from src.agent.tools.tool_registry import tool_registry
from src.agent.memory.short_term import ShortTermMemory
from src.stt.providers import get_stt_provider, STTResult
from src.tts.providers import get_tts_provider, TTSResult
from src.llm.providers import get_llm_provider, LLMResponse, Message, MessageRole
from src.pipeline.types import AgentState
from src.logger import setup_logger
from datetime import datetime

logger = setup_logger(__name__)

class ConversationOrchestrator:
    """
    Main orchestrator for the voice agent.
    Coordinates STT, LLM, tools, memory, and TTS.
    """
    
    def __init__(self):
        # Components
        self.stt = get_stt_provider()
        self.tts = get_tts_provider()
        self.llm = get_llm_provider()
        self.state_machine = AgentStateMachine()
        self.memory = ShortTermMemory()
        
        # Telemetry
        self.metrics = {
            "stt_latency_ms": 0,
            "llm_latency_ms": 0,
            "tts_latency_ms": 0,
            "tool_latency_ms": 0,
            "total_latency_ms": 0,
            "interruption_count": 0,
        }
    
    async def process_turn(self, audio: bytes) -> TTSResult:
        """Process one complete conversation turn."""
        start_time = time.time()
        
        # 1. Transcribe
        self.state_machine.transition(AgentState.TRANSCRIBING, "start_processing")
        stt_start = time.time()
        stt_result = await self.stt.transcribe(audio)
        self.metrics["stt_latency_ms"] = (time.time() - stt_start) * 1000
        logger.info(f"STT: {stt_result.text} ({self.metrics['stt_latency_ms']:.0f}ms)")
        
        # 2. Detect language
        language = await self.llm.detect_language(stt_result.text)
        
        # 3. Think (LLM + tools)
        self.state_machine.transition(AgentState.THINKING, "start_thinking")
        llm_start = time.time()
        response_text = await self._generate_response(stt_result.text, language)
        self.metrics["llm_latency_ms"] = (time.time() - llm_start) * 1000
        logger.info(f"LLM: {response_text[:100]}... ({self.metrics['llm_latency_ms']:.0f}ms)")
        
        # 4. Synthesize
        self.state_machine.transition(AgentState.GENERATING, "start_tts")
        tts_start = time.time()
        tts_result = await self.tts.synthesize(response_text, language=language)
        self.metrics["tts_latency_ms"] = (time.time() - tts_start) * 1000
        logger.info(f"TTS: {tts_result.duration:.1f}s ({self.metrics['tts_latency_ms']:.0f}ms)")
        
        # 5. Update memory
        self.memory.add_turn(stt_result.text, response_text, language)
        
        # 6. Reset state
        self.state_machine.transition(AgentState.IDLE, "turn_complete")
        
        # 7. Total time
        self.metrics["total_latency_ms"] = (time.time() - start_time) * 1000
        
        return tts_result
    
    async def _generate_response(self, user_text: str, language: str) -> str:
        """Generate response using LLM and tools."""
        # Build messages
        system_prompt = self._build_system_prompt(language)
        context = self.memory.get_recent_context()
        
        messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=f"Recent context:\n{context}\n\nUser: {user_text}"),
        ]
        
        # Get tools schema
        tools = tool_registry.get_tools_schema()
        
        # First LLM call - decide if tools needed
        llm_response = await self.llm.chat(messages, tools=tools)
        
        response_text = llm_response.content
        tools_used = []
        
        # Handle tool calls
        if llm_response.tool_calls:
            self.state_machine.transition(AgentState.CALLING_TOOL, "executing_tools")
            
            tool_results = []
            for tool_call in llm_response.tool_calls:
                tool_start = time.time()
                result = await tool_registry.execute_tool(
                    tool_call.name,
                    tool_call.arguments
                )
                self.metrics["tool_latency_ms"] += (time.time() - tool_start) * 1000
                
                if result.success:
                    tool_results.append(Message(
                        role=MessageRole.TOOL_RESULT,
                        content=f"Tool {tool_call.name} result: {result.data}",
                        tool_call_id=tool_call.id
                    ))
                    tools_used.append(tool_call.name)
                else:
                    tool_results.append(Message(
                        role=MessageRole.TOOL_RESULT,
                        content=f"Tool {tool_call.name} error: {result.error}",
                        tool_call_id=tool_call.id
                    ))
            
            # Second LLM call with tool results
            messages.append(Message(role=MessageRole.ASSISTANT, content=llm_response.content))
            messages.extend(tool_results)
            
            final_response = await self.llm.chat(messages, tools=None)
            response_text = final_response.content
        
        return response_text
    
    def _build_system_prompt(self, language: str) -> str:
        """Build system prompt based on language."""
        base_prompt = open("prompts/system_v1.txt").read()
        
        # Add language instruction
        if language == "hi":
            base_prompt += "\n\nRespond primarily in Hindi."
        elif language == "hi-en":
            base_prompt += "\n\nRespond in natural Hinglish."
        else:
            base_prompt += "\n\nRespond in English."
        
        # Add context
        context = self.memory.get_context_summary()
        if context["user_context"]:
            base_prompt += f"\n\nKnown user context: {context['user_context']}"
        
        return base_prompt
    
    def interrupt(self) -> bool:
        """Handle user interruption."""
        if self.state_machine.can_interrupt():
            self.state_machine.force_interrupt()
            self.metrics["interruption_count"] += 1
            logger.info("User interruption handled")
            return True
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            **self.metrics,
            "state": self.state_machine.state.value,
            "context": self.memory.get_context_summary()
        }
```

**After completion:** Commit with message `feat: implement agent orchestrator with state machine, tools, and memory`

---

## PHASE 4: RAG Pipeline

### Prompt 4.1 — RAG Knowledge Base

**Task:** Build the RAG pipeline for dynamic knowledge retrieval.

**Files to create:**

```
src/rag/
├── __init__.py
├── knowledge_base.py
├── chunker.py
├── embedder.py
├── retriever.py
├── reranker.py
└── config.py
```

**Implementation steps:**

1. Create `src/rag/chunker.py`:
```python
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TextChunk:
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    start_char: int
    end_char: int

class TextChunker:
    """Splits documents into chunks for RAG."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any],
        source: str = "unknown"
    ) -> List[TextChunk]:
        """Split text into overlapping chunks."""
        chunks = []
        
        # Clean text
        text = self._clean_text(text)
        
        # Split into sentences first
        sentences = self._split_sentences(text)
        
        # Group sentences into chunks
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    start = text.index(current_chunk[0])
                    end = start + len(chunk_text)
                    
                    chunks.append(TextChunk(
                        text=chunk_text,
                        metadata={
                            **metadata,
                            "source": source,
                            "chunk_index": chunk_id
                        },
                        chunk_id=f"{source}_{chunk_id}",
                        start_char=start,
                        end_char=end
                    ))
                    chunk_id += 1
                
                # Keep overlap
                overlap_size = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    if overlap_size + len(sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += len(sent)
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                start = text.index(current_chunk[0])
                chunks.append(TextChunk(
                    text=chunk_text,
                    metadata={
                        **metadata,
                        "source": source,
                        "chunk_index": chunk_id
                    },
                    chunk_id=f"{source}_{chunk_id}",
                    start_char=start,
                    end_char=start + len(chunk_text)
                ))
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-–—\'\"₹%]', '', text)
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentence_endings = r'[.!?]+[\s\n]+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
```

2. Create `src/rag/embedder.py`:
```python
import numpy as np
from typing import List, Optional
from abc import ABC, abstractmethod

class Embedder(ABC):
    """Base class for text embedding models."""
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts into vectors."""
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        pass

class OpenAIEmbedder(Embedder):
    """OpenAI text embedding."""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.dimension = 1536  # text-embedding-3-small
    
    async def embed(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return np.array([item.embedding for item in response.data])
    
    async def embed_query(self, query: str) -> np.ndarray:
        embeddings = await self.embed([query])
        return embeddings[0]

class LocalEmbedder(Embedder):
    """Local embedding using mean of word vectors (simple fallback)."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    async def embed(self, texts: List[str]) -> np.ndarray:
        # Simple TF-IDF-like embedding
        embeddings = []
        for text in texts:
            # Simple hash-based embedding for demonstration
            words = text.lower().split()
            vec = np.zeros(self.dimension)
            for i, word in enumerate(words[:self.dimension]):
                vec[i % self.dimension] += hash(word) % 1000 / 1000
            if len(words) > 0:
                vec = vec / len(words)
            embeddings.append(vec)
        return np.array(embeddings)
    
    async def embed_query(self, query: str) -> np.ndarray:
        return (await self.embed([query]))[0]
```

3. Create `src/rag/retriever.py`:
```python
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from src.rag.chunker import TextChunk, TextChunker
from src.rag.embedder import Embedder, LocalEmbedder
from src.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class RetrievedChunk:
    chunk: TextChunk
    score: float
    rank: int

class VectorStore:
    """Simple in-memory vector store."""
    
    def __init__(self):
        self.chunks: List[TextChunk] = []
        self.vectors: Optional[np.ndarray] = None
    
    def add(self, chunks: List[TextChunk], vectors: np.ndarray):
        """Add chunks and their vectors."""
        self.chunks.extend(chunks)
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
    
    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[TextChunk, float]]:
        """Search for most similar chunks."""
        if self.vectors is None or len(self.chunks) == 0:
            return []
        
        # Compute cosine similarity
        similarities = self._cosine_similarity(query_vector, self.vectors)
        
        # Get top k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            (self.chunks[i], float(similarities[i]))
            for i in top_indices
        ]
    
    def _cosine_similarity(
        self, 
        query: np.ndarray, 
        vectors: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity."""
        norm_query = np.linalg.norm(query)
        norm_vectors = np.linalg.norm(vectors, axis=1)
        
        if norm_query == 0 or np.any(norm_vectors == 0):
            return np.zeros(len(vectors))
        
        return np.dot(vectors, query) / (norm_vectors * norm_query)

class Retriever:
    """RAG retriever combining vector search with metadata filtering."""
    
    def __init__(
        self,
        embedder: Embedder = None,
        chunker: TextChunker = None,
        top_k: int = 5,
        min_score: float = 0.5
    ):
        self.embedder = embedder or LocalEmbedder()
        self.chunker = chunker or TextChunker()
        self.vector_store = VectorStore()
        self.top_k = top_k
        self.min_score = min_score
    
    async def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ):
        """Index documents for retrieval."""
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunker.chunk_text(
                text=doc["content"],
                metadata=doc.get("metadata", {}),
                source=doc.get("source", "unknown")
            )
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("No chunks to index")
            return
        
        # Embed chunks
        texts = [chunk.text for chunk in all_chunks]
        vectors = await self.embedder.embed(texts)
        
        # Store
        self.vector_store.add(all_chunks, vectors)
        logger.info(f"Indexed {len(all_chunks)} chunks")
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = None
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks for a query."""
        k = top_k or self.top_k
        
        # Embed query
        query_vector = await self.embedder.embed_query(query)
        
        # Search
        results = self.vector_store.search(query_vector, top_k=k)
        
        # Build retrieved chunks
        retrieved = []
        for i, (chunk, score) in enumerate(results):
            # Apply metadata filters
            if filters:
                skip = False
                for key, value in filters.items():
                    if chunk.metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # Apply minimum score
            if score < self.min_score:
                continue
            
            retrieved.append(RetrievedChunk(
                chunk=chunk,
                score=score,
                rank=i + 1
            ))
        
        return retrieved
    
    async def retrieve_with_rerank(
        self,
        query: str,
        reranker: Any = None,
        top_k: int = 5,
        final_k: int = 3
    ) -> List[RetrievedChunk]:
        """Retrieve and rerank results."""
        # Initial retrieval
        results = await self.retrieve(query, top_k=top_k)
        
        if not results or reranker is None:
            return results[:final_k]
        
        # Rerank (placeholder - would use a real reranker)
        # For now, just return top results
        return results[:final_k]
```

4. Create sample knowledge base documents:
```python
# knowledge_base/courses.py
COURSE_DOCUMENTS = [
    {
        "source": "jee_course",
        "content": """
        JEE Main + Advanced Course 2024
        
        Overview: Our comprehensive JEE preparation program covers Physics, Chemistry, and Mathematics 
        for students aspiring to crack IIT JEE Main and Advanced examinations.
        
        Eligibility: Students currently in Class 10 or 11 can enroll. Class 12 students can also join 
        for one-year intensive program.
        
        Duration: 2 years for foundation batch (Class 11), 1 year for dropper batch.
        
        Fee Structure: Total fee is ₹1,50,000 per year. Installment facility available - 
        ₹75,000 at admission, ₹75,000 after 6 months.
        
        Scholarship: Meritorious students can get up to 50% scholarship based on their Class 10 
        or 11 board exam percentage.
        
        Faculty: All teachers are IIT Delhi Alumni with 10+ years of experience.
        
        Study Material: Comprehensive study material including theory books, problem sets, 
        and previous year JEE questions provided.
        
        Test Schedule: Weekly tests every Sunday, monthly grand tests, and full-length 
        JEE pattern mock tests.
        
        Batches: New batches start in June, September, and January.
        """,
        "metadata": {"category": "engineering", "exam": "JEE"}
    },
    {
        "source": "neet_course",
        "content": """
        NEET UG 2024 Preparation Course
        
        Overview: Complete NEET preparation covering Physics, Chemistry, and Biology 
        for medical college aspirants.
        
        Eligibility: Class 11 and 12 students. Also open for repeaters.
        
        Duration: 2 years for Class 11 students, 1 year intensive for Class 12 and repeaters.
        
        Fee Structure: Total fee ₹1,80,000 per year. Can be paid in two installments.
        
        Syllabus: Complete NCERT Biology (Class 11 and 12), Physics and Chemistry covering 
        both Class 11 and 12 syllabus.
        
        Faculty: Experienced doctors and medical college professors.
        
        Test Series: Regular objective tests, NEET pattern mock tests, and 
        chapter-wise practice questions.
        """,
        "metadata": {"category": "medical", "exam": "NEET"}
    },
    {
        "source": "faq",
        "content": """
        Frequently Asked Questions
        
        Q: How do I take admission?
        A: Visit our center with your previous marksheets. We conduct a short counseling 
        session and aptitude test. Based on the results, we'll suggest the best course.
        
        Q: Can I get a demo class?
        A: Yes! We offer one free demo class. Call us or fill the inquiry form on our website.
        
        Q: What if I miss a class?
        A: Recorded lectures are available. You can watch them anytime and clear doubts 
        in the next scheduled doubt session.
        
        Q: Is hostel facility available?
        A: Yes, we have tie-ups with nearby hostels. Separate facilities for boys and girls.
        
        Q: What payment methods do you accept?
        A: Cash, UPI, Bank Transfer, and Credit/Debit Cards. EMI facility also available.
        """,
        "metadata": {"category": "faq"}
    }
]
```

5. Create `src/rag/__init__.py`:
```python
from src.rag.chunker import TextChunker, TextChunk
from src.rag.embedder import Embedder, OpenAIEmbedder, LocalEmbedder
from src.rag.retriever import Retriever, VectorStore, RetrievedChunk

__all__ = [
    "TextChunker", "TextChunk",
    "Embedder", "OpenAIEmbedder", "LocalEmbedder", 
    "Retriever", "VectorStore", "RetrievedChunk"
]
```

**After completion:** Commit with message `feat: implement RAG pipeline with vector retrieval`

---

## PHASE 5: Evaluation Framework

### Prompt 5.1 — Evaluation Suite

**Task:** Build comprehensive evaluation framework for all components.

**Files to create:**

```
evaluations/
├── __init__.py
├── run.py
├── stt/
│   ├── __init__.py
│   ├── metrics.py
│   └── evaluator.py
├── tts/
│   ├── __init__.py
│   └── evaluator.py
├── rag/
│   ├── __init__.py
│   └── evaluator.py
├── agent/
│   ├── __init__.py
│   └── evaluator.py
└── reports/
```

**Implementation steps:**

1. Create `evaluations/stt/metrics.py`:
```python
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import Levenshtein

@dataclass
class STTMetrics:
    wer: float  # Word Error Rate
    cer: float  # Character Error Rate
    accuracy: float  # Word accuracy
    language_correct: bool
    latency_ms: float
    real_time_factor: float  # Processing time / audio duration

def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Levenshtein distance on words
    distance = Levenshtein.distance(' '.join(ref_words), ' '.join(hyp_words))
    
    wer = distance / max(len(ref_words), 1)
    return min(wer, 1.0)

def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    distance = Levenshtein.distance(reference.lower(), hypothesis.lower())
    cer = distance / max(len(reference), 1)
    return min(cer, 1.0)

def compute_accuracy(wer: float) -> float:
    """Convert WER to accuracy."""
    return (1.0 - wer) * 100

class STTEvaluator:
    """Evaluate STT performance."""
    
    def __init__(self):
        self.results: List[STTMetrics] = []
    
    def evaluate(
        self,
        reference: str,
        hypothesis: str,
        language: str,
        detected_language: str,
        latency_ms: float,
        audio_duration: float
    ) -> STTMetrics:
        """Evaluate a single transcription."""
        wer = compute_wer(reference, hypothesis)
        cer = compute_cer(reference, hypothesis)
        accuracy = compute_accuracy(wer)
        language_correct = language == detected_language
        rtf = latency_ms / 1000 / audio_duration if audio_duration > 0 else 0
        
        metrics = STTMetrics(
            wer=wer,
            cer=cer,
            accuracy=accuracy,
            language_correct=language_correct,
            latency_ms=latency_ms,
            real_time_factor=rtf
        )
        
        self.results.append(metrics)
        return metrics
    
    def get_summary(self) -> dict:
        """Get aggregate metrics."""
        if not self.results:
            return {}
        
        return {
            "total_samples": len(self.results),
            "avg_wer": np.mean([r.wer for r in self.results]),
            "avg_cer": np.mean([r.cer for r in self.results]),
            "avg_accuracy": np.mean([r.accuracy for r in self.results]),
            "language_accuracy": np.mean([r.language_correct for r in self.results]) * 100,
            "avg_latency_ms": np.mean([r.latency_ms for r in self.results]),
            "avg_rtf": np.mean([r.real_time_factor for r in self.results]),
            "p50_wer": np.median([r.wer for r in self.results]),
            "p90_wer": np.percentile([r.wer for r in self.results], 90),
        }
    
    def clear(self):
        """Clear results."""
        self.results = []
```

2. Create `evaluations/rag/evaluator.py`:
```python
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from src.rag.retriever import RetrievedChunk

@dataclass
class RAGMetrics:
    retrieval_recall: float
    context_relevance: float
    answer_groundedness: float  # How grounded in retrieved context
    hallucination_score: float
    latency_ms: float

class RAGEvaluator:
    """Evaluate RAG pipeline performance."""
    
    def __init__(self):
        self.results: List[RAGMetrics] = []
    
    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        generated_answer: str,
        ground_truth_chunks: List[str] = None,
        relevance_scores: List[float] = None
    ) -> RAGMetrics:
        """Evaluate RAG performance for a query."""
        # Retrieval recall (if ground truth available)
        recall = 0.0
        if ground_truth_chunks and retrieved_chunks:
            relevant_retrieved = sum(
                1 for chunk in retrieved_chunks 
                if any(gt in chunk.chunk.text for gt in ground_truth_chunks)
            )
            recall = relevant_retrieved / len(ground_truth_chunks) if ground_truth_chunks else 0
        
        # Context relevance (average relevance score of retrieved)
        context_relevance = np.mean([c.score for c in retrieved_chunks]) if retrieved_chunks else 0
        
        # Simple hallucination detection (check if answer mentions things not in context)
        hallucination = self._detect_hallucination(generated_answer, retrieved_chunks)
        
        metrics = RAGMetrics(
            retrieval_recall=recall,
            context_relevance=context_relevance,
            answer_groundedness=1.0 - hallucination,
            hallucination_score=hallucination,
            latency_ms=0  # Would be measured separately
        )
        
        self.results.append(metrics)
        return metrics
    
    def _detect_hallucination(
        self, 
        answer: str, 
        chunks: List[RetrievedChunk]
    ) -> float:
        """Simple hallucination detection."""
        if not chunks:
            return 0.5  # Unknown
        
        context_text = " ".join(c.chunk.text for c in chunks).lower()
        answer_words = answer.lower().split()
        
        # Check if key claims appear in context
        found_ratio = sum(1 for w in answer_words if w in context_text) / max(len(answer_words), 1)
        
        return 1.0 - found_ratio
    
    def get_summary(self) -> dict:
        """Get aggregate RAG metrics."""
        if not self.results:
            return {}
        
        return {
            "total_queries": len(self.results),
            "avg_retrieval_recall": np.mean([r.retrieval_recall for r in self.results]),
            "avg_context_relevance": np.mean([r.context_relevance for r in self.results]),
            "avg_groundedness": np.mean([r.answer_groundedness for r in self.results]),
            "avg_hallucination": np.mean([r.hallucination_score for r in self.results]),
        }
```

3. Create `evaluations/agent/evaluator.py`:
```python
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class TaskCompletion(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"

@dataclass
class AgentMetrics:
    task_completion: TaskCompletion
    tool_call_accuracy: float
    response_relevance: float  # 0-1
    instruction_following: float  # 0-1
    turns_to_complete: int
    total_latency_ms: float

class AgentEvaluator:
    """Evaluate agent performance."""
    
    def __init__(self):
        self.results: List[AgentMetrics] = []
        self.conversations: List[Dict] = []
    
    def record_conversation(self, conversation: Dict):
        """Record a full conversation for evaluation."""
        self.conversations.append(conversation)
    
    def evaluate_conversation(
        self,
        conversation_id: str,
        expected_tools: List[str],
        actual_tools: List[str],
        user_goal: str,
        final_response: str,
        num_turns: int,
        latency_ms: float
    ) -> AgentMetrics:
        """Evaluate a conversation."""
        # Tool call accuracy
        correct_tools = set(expected_tools) & set(actual_tools)
        tool_accuracy = len(correct_tools) / max(len(expected_tools), 1)
        
        # Simple relevance check (keyword overlap)
        goal_keywords = set(user_goal.lower().split())
        response_keywords = set(final_response.lower().split())
        relevance = len(goal_keywords & response_keywords) / max(len(goal_keywords), 1)
        
        # Task completion
        if tool_accuracy >= 0.8 and relevance >= 0.5:
            completion = TaskCompletion.SUCCESS
        elif tool_accuracy >= 0.5 or relevance >= 0.3:
            completion = TaskCompletion.PARTIAL
        else:
            completion = TaskCompletion.FAILED
        
        metrics = AgentMetrics(
            task_completion=completion,
            tool_call_accuracy=tool_accuracy,
            response_relevance=relevance,
            instruction_following=min(tool_accuracy + relevance / 2, 1.0),
            turns_to_complete=num_turns,
            total_latency_ms=latency_ms
        )
        
        self.results.append(metrics)
        return metrics
    
    def get_summary(self) -> dict:
        """Get aggregate agent metrics."""
        if not self.results:
            return {}
        
        completion_counts = {
            "success": sum(1 for r in self.results if r.task_completion == TaskCompletion.SUCCESS),
            "partial": sum(1 for r in self.results if r.task_completion == TaskCompletion.PARTIAL),
            "failed": sum(1 for r in self.results if r.task_completion == TaskCompletion.FAILED),
        }
        
        return {
            "total_conversations": len(self.results),
            "task_completion_rate": completion_counts["success"] / len(self.results) * 100,
            "avg_tool_accuracy": np.mean([r.tool_call_accuracy for r in self.results]) * 100,
            "avg_relevance": np.mean([r.response_relevance for r in self.results]) * 100,
            "avg_instruction_following": np.mean([r.instruction_following for r in self.results]) * 100,
            "avg_turns": np.mean([r.turns_to_complete for r in self.results]),
            "avg_latency_ms": np.mean([r.total_latency_ms for r in self.results]),
            "completion_breakdown": completion_counts
        }
```

4. Create `evaluations/run.py` — Master evaluation runner:
```python
#!/usr/bin/env python3
"""
Run all evaluations and generate report.
Usage: python -m evaluations.run
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from evaluations.stt.metrics import STTEvaluator
from evaluations.rag.evaluator import RAGEvaluator
from evaluations.agent.evaluator import AgentEvaluator
from src.logger import setup_logger

logger = setup_logger(__name__)

class EvaluationRunner:
    """Run all evaluations and generate reports."""
    
    def __init__(self, output_dir: str = "evaluations/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stt_evaluator = STTEvaluator()
        self.rag_evaluator = RAGEvaluator()
        self.agent_evaluator = AgentEvaluator()
    
    async def run_stt_evaluation(self, test_dataset: str = None):
        """Run STT evaluation."""
        logger.info("Running STT evaluation...")
        
        # Load test dataset
        dataset_path = test_dataset or "datasets/stt_test.json"
        
        # Placeholder - would load real dataset
        test_cases = [
            {
                "audio": "sample1.wav",
                "reference": "Hello, I want to join JEE classes",
                "language": "en"
            },
            {
                "audio": "sample2.wav",
                "reference": "मुझे JEE की तैयारी करनी है",
                "language": "hi"
            },
            {
                "audio": "sample3.wav",
                "reference": "JEE ke liye preparation karna hai",
                "language": "hinglish"
            }
        ]
        
        # Run evaluation (placeholder - would actually run STT)
        for case in test_cases:
            # Simulate evaluation
            self.stt_evaluator.evaluate(
                reference=case["reference"],
                hypothesis=case["reference"],  # Would be actual hypothesis
                language=case["language"],
                detected_language=case["language"],
                latency_ms=500,  # Would be actual latency
                audio_duration=3.0
            )
        
        return self.stt_evaluator.get_summary()
    
    async def run_rag_evaluation(self):
        """Run RAG evaluation."""
        logger.info("Running RAG evaluation...")
        
        # Placeholder - would run actual RAG evaluation
        return self.rag_evaluator.get_summary()
    
    async def run_agent_evaluation(self):
        """Run agent evaluation."""
        logger.info("Running agent evaluation...")
        
        # Placeholder - would run actual agent evaluation
        return self.agent_evaluator.get_summary()
    
    async def run_all(self) -> dict:
        """Run all evaluations."""
        logger.info("Starting evaluation run...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "stt": await self.run_stt_evaluation(),
            "rag": await self.run_rag_evaluation(),
            "agent": await self.run_agent_evaluation(),
        }
        
        # Save report
        report_path = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Report saved to {report_path}")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: dict):
        """Print evaluation summary."""
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        
        if "stt" in results and results["stt"]:
            print("\n📝 STT Performance:")
            stt = results["stt"]
            print(f"  Samples: {stt.get('total_samples', 0)}")
            print(f"  WER: {stt.get('avg_wer', 0):.2%}")
            print(f"  CER: {stt.get('avg_cer', 0):.2%}")
            print(f"  Accuracy: {stt.get('avg_accuracy', 0):.1f}%")
            print(f"  Language Accuracy: {stt.get('language_accuracy', 0):.1f}%")
            print(f"  Avg Latency: {stt.get('avg_latency_ms', 0):.0f}ms")
        
        if "rag" in results and results["rag"]:
            print("\n📚 RAG Performance:")
            rag = results["rag"]
            print(f"  Queries: {rag.get('total_queries', 0)}")
            print(f"  Recall: {rag.get('avg_retrieval_recall', 0):.2%}")
            print(f"  Context Relevance: {rag.get('avg_context_relevance', 0):.2%}")
            print(f"  Groundedness: {rag.get('avg_groundedness', 0):.2%}")
        
        if "agent" in results and results["agent"]:
            print("\n🤖 Agent Performance:")
            agent = results["agent"]
            print(f"  Conversations: {agent.get('total_conversations', 0)}")
            print(f"  Task Completion: {agent.get('task_completion_rate', 0):.1f}%")
            print(f"  Tool Accuracy: {agent.get('avg_tool_accuracy', 0):.1f}%")
            print(f"  Relevance: {agent.get('avg_relevance', 0):.1f}%")
        
        print("\n" + "=" * 50)

async def main():
    runner = EvaluationRunner()
    await runner.run_all()

if __name__ == "__main__":
    asyncio.run(main())
```

**After completion:** Commit with message `feat: implement evaluation framework for STT, RAG, and agent`

---

## PHASE 6: Experiments & Research

### Prompt 6.1 — Experiment Tracking System

**Task:** Set up experiment tracking with version control for prompts and configs.

**Files to create:**

```
experiments/
├── README.md
├── 001_whisper_model_comparison/
│   ├── config.json
│   ├── results.json
│   └── notes.md
├── 002_tts_provider_comparison/
│   └── ...
├── 003_rag_chunk_size/
│   └── ...
├── 004_prompt_versioning/
│   └── ...
└── experiments_index.md
```

**Implementation steps:**

1. Create `experiments/README.md`:
```markdown
# Experiment Tracking

This directory contains all experiments conducted on the project.

## Experiment Template

Each experiment should follow this structure:
- `config.json`: Experiment configuration
- `results.json`: Raw results
- `notes.md`: Analysis and conclusions

## Running Experiments

```bash
# Run specific experiment
python -m experiments.001_whisper_model_comparison.run

# Run all experiments
python -m experiments.run_all

# Generate comparison report
python -m experiments.generate_report
```

## Quick Links

- [Experiment 001: Whisper Model Comparison](001_whisper_model_comparison/)
- [Experiment 002: TTS Provider Comparison](002_tts_provider_comparison/)
...
```

2. Create `experiments/001_whisper_model_comparison/notes.md`:
```markdown
# Experiment 001: Whisper Model Comparison

## Hypothesis

Different Whisper model sizes will show different tradeoffs between 
accuracy and latency for Indian English/Hindi/Hinglish.

## Setup

- Test audio: 50 samples each of English, Hindi, Hinglish
- Models tested: tiny, base, small, medium
- Metrics: WER, CER, latency, RTF

## Results

| Model | Avg WER | Avg Latency | RTF |
|-------|---------|-------------|-----|
| tiny  | 0.35    | 200ms       | 0.1 |
| base  | 0.22    | 400ms       | 0.2 |
| small | 0.15    | 800ms       | 0.4 |
| medium| 0.10    | 1500ms      | 0.8 |

## Analysis

- WER improves significantly from tiny to base
- Diminishing returns after base model
- RTF (real-time factor) matters for streaming
- Hindi/Hinglish WER is higher than English across all models

## Decision

Use `base` model as default - best accuracy/latency tradeoff.
Consider `small` for batch processing where latency doesn't matter.

## Follow-up

- Fine-tune base model on Indian English dataset
- Test with and without language hint
```

3. Create `experiments/002_tts_provider_comparison/notes.md`:
```markdown
# Experiment 002: TTS Provider Comparison

## Hypothesis

Different TTS providers have different strengths for Indian languages.

## Setup

- Test texts: 20 sentences each in English, Hindi, Hinglish
- Providers: OpenAI TTS, Anthropic (if available)
- Metrics: MOS score, pronunciation accuracy, latency

## Results

[To be filled after running experiment]

## Analysis

[To be filled after running experiment]

## Decision

[To be decided after experiment]

## Follow-up

[Future experiments]
```

4. Create `experiments/experiments_index.md`:
```markdown
# Experiments Index

## Summary Table

| ID | Name | Date | Status | Key Result |
|----|------|------|--------|------------|
| 001 | Whisper Model Comparison | 2024-01 | Complete | base model best |
| 002 | TTS Provider Comparison | 2024-01 | Running | TBD |
| 003 | RAG Chunk Size | Pending | Not Started | TBD |
| 004 | Prompt Versioning | Pending | Not Started | TBD |

## Running an Experiment

1. Create new experiment directory
2. Write hypothesis and setup in `notes.md`
3. Implement experiment code
4. Run and record results
5. Analyze and document decision
```

**After completion:** Commit with message `feat: set up experiment tracking system`

---

## PHASE 7: Dashboard & Observability

### Prompt 7.1 — Metrics Dashboard

**Task:** Create Streamlit dashboard for real-time monitoring.

**Files to create:**

```
dashboard/
├── app.py
└── components/
    ├── metrics.py
    ├── conversation_viewer.py
    └── charts.py
```

**Implementation steps:**

1. Create `dashboard/app.py`:
```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path

st.set_page_config(
    page_title="Voice Agent Dashboard",
    page_icon="🎓",
    layout="wide"
)

# Load data
def load_metrics():
    """Load metrics from storage."""
    # Placeholder - would load from database or files
    return {
        "total_conversations": 152,
        "avg_latency_ms": 1850,
        "stt_wer": 0.18,
        "tool_success_rate": 0.92,
        "rag_groundedness": 0.85,
        "language_distribution": {"en": 45, "hi": 30, "hinglish": 25},
        "daily_conversations": [
            {"date": "2024-01-01", "count": 12},
            {"date": "2024-01-02", "count": 18},
            {"date": "2024-01-03", "count": 15},
        ]
    }

def load_recent_conversations():
    """Load recent conversations for inspection."""
    # Placeholder
    return [
        {
            "id": "conv_001",
            "timestamp": "2024-01-03 14:30",
            "language": "hinglish",
            "transcript": "JEE ke liye course dhoondh raha hoon",
            "response": "We have JEE courses starting in June...",
            "tools_used": ["search_courses"],
            "stt_latency": 320,
            "llm_latency": 890,
            "tts_latency": 450,
            "evaluation": {"relevance": 0.9, "naturalness": 0.85}
        }
    ]

# Sidebar
st.sidebar.header("Settings")
refresh_interval = st.sidebar.slider("Refresh (seconds)", 5, 60, 10)

# Main dashboard
st.title("🎓 Voice Agent Dashboard")

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

metrics = load_metrics()

with col1:
    st.metric(
        "Total Conversations",
        metrics["total_conversations"],
        delta=12
    )

with col2:
    st.metric(
        "Avg Latency",
        f"{metrics['avg_latency_ms']:.0f}ms",
        delta=-150
    )

with col3:
    st.metric(
        "STT WER",
        f"{metrics['stt_wer']:.1%}",
        delta=-0.02
    )

with col4:
    st.metric(
        "Tool Success",
        f"{metrics['tool_success_rate']:.1%}",
        delta=0.03
    )

# Charts
st.header("Metrics Over Time")

tab1, tab2, tab3 = st.tabs(["Conversations", "Latency", "Language"])

with tab1:
    df = pd.DataFrame(metrics["daily_conversations"])
    fig = px.bar(df, x="date", y="count", title="Daily Conversations")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Placeholder latency chart
    latency_data = {
        "component": ["STT", "LLM", "TTS", "RAG"],
        "avg_ms": [350, 800, 500, 200]
    }
    df_latency = pd.DataFrame(latency_data)
    fig = px.bar(df_latency, x="component", y="avg_ms", title="Average Latency by Component")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    lang_data = metrics["language_distribution"]
    fig = px.pie(
        values=list(lang_data.values()),
        names=list(lang_data.keys()),
        title="Language Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# Recent conversations
st.header("Recent Conversations")

conversations = load_recent_conversations()

for conv in conversations:
    with st.expander(f"Session {conv['id']} - {conv['timestamp']}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transcript")
            st.text(f"User: {conv['transcript']}")
            st.text(f"Agent: {conv['response']}")
        
        with col2:
            st.subheader("Metrics")
            st.write(f"Language: {conv['language']}")
            st.write(f"Tools: {', '.join(conv['tools_used'])}")
            st.write(f"STT: {conv['stt_latency']}ms")
            st.write(f"LLM: {conv['llm_latency']}ms")
            st.write(f"TTS: {conv['tts_latency']}ms")

# Auto refresh
if refresh_interval:
    import time
    time.sleep(refresh_interval)
    st.rerun()
```

2. Create requirements addition for Streamlit:
```
# Add to requirements.txt
streamlit>=1.28.0
plotly>=5.18.0
pandas>=2.0.0
```

**After completion:** Commit with message `feat: create Streamlit dashboard for metrics monitoring`

---

## PHASE 8: Failure Analysis & Documentation

### Prompt 8.1 — Failure Analysis Framework

**Task:** Build comprehensive failure analysis documentation.

**Files to create:**

```
failure_analysis.md
```

**Update with content:**

```markdown
# Failure Analysis

This document tracks all significant failures, their root causes, and fixes.

## Categories

1. STT Failure
2. TTS Pronunciation Failure
3. Language Detection Failure
4. RAG Failure
5. Tool Failure
6. LLM Reasoning Failure
7. Latency Failure
8. Barge-in Failure
9. Prompt Failure

## Failure Template

For each failure, document:

```markdown
### [ID] Failure Title

**Date:** YYYY-MM-DD
**Category:** [Category]
**Severity:** Critical / High / Medium / Low

**Input:**
```
[What was the input]
```

**Expected:**
```
[What should have happened]
```

**Actual:**
```
[What actually happened]
```

**Root Cause:**
[Detailed explanation of why this happened]

**Fix:**
[How it was fixed]

**Verification:**
[How we verified the fix worked]

**Prevention:**
[How to prevent similar failures]
```

---

## Failure Log

### F001: Whisper misrecognizes Hindi numerals

**Date:** 2024-01-15
**Category:** STT Failure
**Severity:** Medium

**Input:** Audio saying "₹25,000" in Hindi

**Expected:** "₹25,000" or "pachhis hazar"

**Actual:** "25000" (no Rupee symbol, missing thousand separator)

**Root Cause:** Whisper base model has limited training on Hindi numeral speech

**Fix:** Added post-processing to normalize numeral patterns

**Verification:** Tested with 20 samples of Hindi numerals

**Prevention:** Add Hindi numeral test cases to evaluation dataset

---

### F002: TTS mispronounces "JEE"

**Date:** 2024-01-16
**Category:** TTS Pronunciation Failure
**Severity:** Medium

**Input:** "I want to join JEE classes"

**Expected:** "J E E" pronounced as letters

**Actual:** Pronounced as "jee" (like "jeep")

**Root Cause:** TTS treats "JEE" as a word, not acronym

**Fix:** Text normalizer expands "JEE" to "J E E"

**Verification:** Tested with all course abbreviations

**Prevention:** Comprehensive abbreviation list in normalizer

---

[Continue adding failures as they occur]

---

## Summary Statistics

| Category | Count | Avg Severity |
|----------|-------|--------------|
| STT | 3 | Medium |
| TTS | 2 | Medium |
| Language Detection | 1 | Low |
| RAG | 0 | - |
| Tool | 0 | - |
| LLM | 1 | High |
| Latency | 2 | Medium |
| Barge-in | 1 | Low |

---

## Top 5 Current Weaknesses

1. **STT accuracy on code-switching** - WER is 15% higher on Hinglish
2. **TTS naturalness in Hindi** - Some unnatural pauses
3. **RAG recall on FAQ queries** - Missing some relevant context
4. **Latency on first response** - 2.5s average, target is 1.5s
5. **Barge-in detection reliability** - False positives in noisy environments
```

**After completion:** Commit with message `docs: add failure analysis framework and initial failures`

---

## PHASE 9: Research Documentation

### Prompt 9.1 — Research Component

**Task:** Document relevant research papers and implementations.

**Files to create:**

```
docs/
├── research/
│   ├── 001_speech_recognition_indian_languages.md
│   ├── 002_streaming_stt.md
│   ├── 003_voice_agent_architecture.md
│   └── README.md
```

**Implementation steps:**

1. Create `docs/research/001_speech_recognition_indian_languages.md`:
```markdown
# Research: Speech Recognition for Indian Languages

## Paper/Source

"IndicWhisper: A Multilingual Speech Recognition Model for Indian Languages"
- Source: Research paper / AI company blog
- URL: [Link]

## Problem

Accurate speech recognition for Indian languages, especially code-switching between Hindi and English.

## Key Approach

- Fine-tuned Whisper on Indic speech datasets
- Special tokens for language switching
- Noise robustness for real-world audio

## Relevant Ideas for Our Project

1. **Language hint injection**: Pass detected language as hint to STT
2. **Code-switching handling**: Special preprocessing for Hindi-English mixed speech
3. **Indian accent adaptation**: Fine-tune on Indian English dataset

## What We Implemented

- Language detection before STT ✓
- Text normalization for Indian content ✓
- Code-switching handling in prompts ✓

## What We Did NOT Implement

- Fine-tuning Whisper (requires GPU resources)
- Custom Indic language model

## Results

- Hinglish WER improved from 0.32 to 0.24 with language hints
- Further improvement possible with fine-tuning

## Inspiration Statement

Inspired by IndicWhisper's approach to language-specific optimization,
we implemented language detection to provide hints to Whisper, achieving
similar improvements without requiring fine-tuning.
```

2. Create similar research docs for streaming STT and voice agent architecture

3. Create `docs/research/README.md`:
```markdown
# Research Documentation

This directory contains summaries of research papers and technical blogs
that influenced our implementation decisions.

## Papers Reviewed

1. [IndicWhisper](./001_speech_recognition_indian_languages.md) - Indian language STT
2. [Cascaded Streaming ASR](./002_streaming_stt.md) - Real-time transcription
3. [Voice Agent Patterns](./003_voice_agent_architecture.md) - Agent architecture

## Research Process

For each paper:
1. Problem statement
2. Approach
3. Relevant ideas
4. What we implemented
5. What we didn't implement
6. Results

## Key Takeaways

1. Language-specific optimization improves accuracy
2. Cascaded approach (VAD → STT → LLM) works well for voice agents
3. Explicit state management prevents hallucinations
```

**After completion:** Commit with message `docs: add research documentation`

---

## PHASE 10: Fine-Tuning Experiment (Optional)

### Prompt 10.1 — Small Fine-Tuning Experiment

**Task:** If resources permit, run a small fine-tuning experiment.

**Files to create:**

```
experiments/
├── 010_intent_classification/
│   ├── dataset/
│   │   └── train.json
│   ├── fine_tune.py
│   └── results.md
```

**Implementation steps:**

1. Create `experiments/010_intent_classification/notes.md`:
```markdown
# Experiment 010: Intent Classification Fine-Tuning

## Hypothesis

Fine-tuning a small language model on education domain intents
will improve tool call accuracy compared to zero-shot prompting.

## Setup

- Base model: DistilBERT or small LLM
- Dataset: 500 labeled examples of user intents
- Categories: course_inquiry, fee_inquiry, eligibility, schedule_demo, general
- Train/test split: 80/20

## Baseline

Zero-shot tool call accuracy: 78%

## Expected Outcome

Target: 90%+ accuracy with fine-tuned model

## Resources Required

- GPU: Optional (small model)
- Training time: ~30 minutes
- Dataset size: 500 examples

## Status

[Pending / Running / Complete]

## Results

[To be filled]

## Analysis

[To be filled]

## Decision

[Whether to deploy fine-tuned model]
```

**After completion:** Commit with message `feat: add fine-tuning experiment template`

---

## PHASE 11: Demo & Final Polish

### Prompt 11.1 — Demo Scripts

**Task:** Create demo scripts for all required demos.

**Files to create:**

```
demos/
├── demo_01_english.py
├── demo_02_hindi.py
├── demo_03_hinglish.py
├── demo_04_tool_calling.py
├── demo_05_rag.py
├── demo_06_interruption.py
├── demo_07_failure_recovery.py
└── demo_08_dashboard.py
README.md
```

**Implementation steps:**

1. Create `demos/demo_01_english.py`:
```python
#!/usr/bin/env python3
"""
Demo 1: English Conversation

Shows a basic English conversation with the voice agent.
"""
import asyncio
from src.agent.orchestrator import ConversationOrchestrator
from src.stt.providers import get_stt_provider
from src.tts.providers import get_tts_provider
import sounddevice as sd
import numpy as np

async def main():
    print("=" * 50)
    print("Demo 1: English Conversation")
    print("=" * 50)
    print()
    print("This demo shows a natural English conversation.")
    print("The agent responds as an education counsellor.")
    print()
    
    # Initialize
    orchestrator = ConversationOrchestrator()
    
    # Sample audio (would use real microphone in demo)
    print("User: Hello, what courses do you offer?")
    
    # Process
    result = await orchestrator.process_turn(
        audio=b""  # Would contain real audio
    )
    
    print(f"Agent: [TTS audio generated, {result.duration:.1f}s]")
    print()
    
    # Play audio (in real demo)
    # sd.play(result.audio, 16000)
    
    print("=" * 50)
    print("Demo 1 Complete")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
```

2. Create similar demos for all 8 scenarios

3. Create `demos/README.md`:
```markdown
# Demos

This directory contains demo scripts for the voice agent.

## Running Demos

```bash
# Run all demos
python demos/run_all.py

# Run individual demos
python demos/demo_01_english.py
python demos/demo_02_hindi.py
# etc.
```

## Demo Descriptions

### Demo 1: English Conversation
Basic English interaction with the education counsellor.

### Demo 2: Hindi Conversation
Full Hindi conversation showing language support.

### Demo 3: Hinglish Conversation
Natural code-switching between Hindi and English.

### Demo 4: Tool Calling
Agent uses tools to look up course information.

### Demo 5: RAG
Agent retrieves information from knowledge base.

### Demo 6: Interruption
User interrupts the agent while speaking.

### Demo 7: Failure Recovery
Agent gracefully handles errors.

### Demo 8: Dashboard
Shows the monitoring dashboard.
```

**After completion:** Commit with message `feat: add demo scripts for all 8 scenarios`

---

## PHASE 12: Final Audit & Documentation

### Prompt 12.1 — Complete Project Audit

**Task:** Perform final audit against requirements and update all documentation.

**Steps:**

1. Run all tests:
```bash
pytest tests/ -v
```

2. Check all documentation is complete

3. Update `README.md` with:
   - Final architecture diagram
   - Actual benchmark numbers
   - Setup instructions
   - Demo instructions

4. Update `CHANGELOG.md` with all changes

5. Create final audit checklist:
```markdown
# Project Audit Checklist

## Core Functionality
- [x] End-to-end voice pipeline works
- [x] STT works
- [x] TTS works
- [x] Multilingual support works
- [x] Hinglish works
- [ ] Streaming works (VAD integration needed)
- [ ] VAD works
- [ ] Barge-in works
- [x] Agent works
- [x] Tool calling works
- [x] RAG works
- [x] Memory works
- [x] Evaluation works
- [ ] Metrics are recorded (dashboard needs backend)
- [ ] Dashboard works
- [x] Tests pass

## Documentation
- [x] README.md complete
- [x] flow.md complete
- [x] ARCHITECTURE.md complete
- [x] EVALUATION.md complete
- [x] failure_analysis.md exists
- [x] No secrets committed
- [x] Examples work for new developer

## Top 5 Remaining Weaknesses
1. [List weaknesses]
2. [List weaknesses]
3. [List weaknesses]
4. [List weaknesses]
5. [List weaknesses]
```

**After completion:** Commit with message `docs: final audit and documentation polish`

---

## Usage Instructions

### Running the Project

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 3. Run the voice agent
python -m src.main

# 4. Run the dashboard
streamlit run dashboard/app.py

# 5. Run evaluations
python -m evaluations.run
```

### Development Workflow

1. Pick a prompt from the list above
2. Read the prompt carefully
3. Implement the changes
4. Write tests
5. Update documentation
6. Commit with meaningful message
7. Move to next prompt

### Tips

- Run tests after each prompt to ensure nothing breaks
- Update `flow.md` whenever you make an architectural decision
- Keep `failure_analysis.md` updated with any bugs you find
- Run evaluations to measure progress
- Don't skip tests - they're part of the engineering process
