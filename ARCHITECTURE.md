# Architecture Documentation

## System Architecture

```mermaid
graph TD
    A[User] -->|Audio| B(Frontend)
    B -->|WebSocket| C{Voice Gateway}
    C --> D[VAD]
    D -->|Speech detected| E[STT]
    E -->|Text| F[Conversation Orchestrator]
    
    F --> G[Language Detection]
    F --> H[Intent Detection]
    F --> I[Memory]
    F --> J[RAG]
    F --> K[Tool Calling]
    F --> L[LLM]
    
    J --> M[Document Store]
    K --> N[Tool Registry]
    
    L -->|Response| O[Response Pipeline]
    O --> P[Text Normalization]
    P --> Q[TTS]
    Q -->|Audio| B
    
    F -->|Metrics| R[Observability]
```

## Voice Pipeline

```mermaid
graph LR
    A[Microphone] --> B[VAD]
    B -->|Speech| C[STT]
    C --> D[Text]
    D --> E[Orchestrator]
    E --> F[Language Detection]
    E --> G[Intent Detection]
    E --> H[RAG Retrieval]
    E --> I[Tool Calling]
    E --> J[LLM Generation]
    J --> K[Text Normalization]
    K --> L[TTS]
    L --> M[Speaker]
```

## Agent State Machine

```mermaid
stateDiagram-v2
    [*] --> IDLE
    IDLE --> LISTENING : start_conversation
    LISTENING --> TRANSCRIBING : speech_detected
    TRANSCRIBING --> THINKING : transcription_complete
    THINKING --> CALLING_TOOL : tool_needed
    THINKING --> GENERATING : no_tool_needed
    CALLING_TOOL --> GENERATING : tool_results_ready
    GENERATING --> SPEAKING : audio_ready
    SPEAKING --> LISTENING : speech_complete
    SPEAKING --> INTERRUPTED : barge_in
    INTERRUPTED --> LISTENING : interrupt_handled
    INTERRUPTED --> TRANSCRIBING : new_speech_detected
    LISTENING --> IDLE : timeout
    THINKING --> ERROR : error
    GENERATING --> ERROR : error
    CALLING_TOOL --> ERROR : error
    ERROR --> IDLE : reset
```

## RAG Pipeline

```mermaid
graph LR
    A[Documents] --> B[Cleaning]
    B --> C[Chunking]
    C --> D[Metadata]
    D --> E[Embedding]
    E --> F[Vector DB]
    G[Query] --> H[Query Rewriting]
    H --> I[Retrieval]
    I --> J[Re-ranking]
    J --> K[LLM Context]
    K --> L[LLM Generation]
```

## Data Flow

1. **User speaks** → Audio captured via microphone
2. **VAD** detects speech → Pass audio to STT
3. **STT** transcribes → Text with language
4. **Orchestrator** processes:
   - Detect language
   - Retrieve relevant context (RAG)
   - Execute tools if needed
   - Generate response (LLM)
5. **Text normalization** → TTS-ready text
6. **TTS** synthesizes → Audio response
7. **Audio streaming** → User hears response
8. **Telemetry** recorded for evaluation

## Key Components

### Voice Gateway
- WebSocket server for real-time audio
- VAD for speech detection
- Audio buffering and management

### STT Layer
- Provider-agnostic abstraction
- Whisper-based implementation
- Language detection

### Orchestrator
- State machine for conversation flow
- Coordinates all components
- Memory management

### RAG Pipeline
- Document chunking and embedding
- Vector search and retrieval
- Re-ranking for better results

### TTS Layer
- Provider-agnostic abstraction
- Text normalization
- Streaming support

## Evaluation Architecture

```mermaid
graph TD
    A[Test Dataset] --> B{Evaluator}
    B --> C[STT Metrics]
    B --> D[RAG Metrics]
    B --> E[Agent Metrics]
    C --> F[WER/CER]
    D --> G[Recall/Groundedness]
    E --> H[Task Completion/Tool Accuracy]
    F --> I[Report]
    G --> I
    H --> I
```