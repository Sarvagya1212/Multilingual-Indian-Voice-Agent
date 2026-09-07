# Research: Streaming STT

## Paper/Source

**Cascaded Streaming ASR for Real-Time Voice Agents**

- Type: Industry blog (e.g. Google STT blog, NVIDIA NeMo documentation)
- URL: https://cloud.google.com/speech-to-text/docs/streaming-recognize
- Related: Whisper Streaming (https://github.com/ufal/whisper_streaming)

## Problem

Real-time voice agents need to detect the end of a user's utterance and
produce a transcript with bounded latency. Batch ASR (transcribe a complete
audio file) is too slow — by the time the model finishes, the user is
already impatient. Streaming ASR must produce partial transcripts as audio
arrives and signal "end of utterance" without waiting for silence.

## Key Approach

- **VAD-first architecture**: a voice activity detector (Silero, WebRTC VAD)
  segments the audio stream at natural speech boundaries, before the
  recogniser sees it.
- **Cascaded recognition**: short segments (0.5-2s) are forwarded to a
  streaming-capable ASR (e.g. Google `streamingRecognize`,
  Whisper-streaming with `local-attention`). Each segment returns
  partial results, and the final segment returns the final transcript.
- **End-of-utterance detection**: a combination of (a) trailing silence
  duration, (b) confidence floor, and (c) explicit "end of speech" signal
  from the VAD.
- **Barge-in handling**: while the agent is speaking, the STT must remain
  open and re-activate the moment the user starts speaking again —
  requires a separate VAD channel and an interrupt signal that aborts
  TTS playback.

## Relevant Ideas for Our Project

1. **VAD before STT, not embedded in STT.** Decoupling the two lets us
   swap VAD implementations (energy → WebRTC → Silero) without
   touching the recogniser.
2. **Hysteresis in VAD threshold.** A single energy threshold flaps at
   speech boundaries; two thresholds (onset, offset) eliminate this.
3. **Streaming-friendly STT provider interface.** Even if we use a
   non-streaming recogniser internally, the `STTProvider.transcribe_stream()`
   interface lets us collect partial results and surface them for a
   future live-captioning feature.
4. **Barge-in via a parallel VAD channel.** Our gateway
   (`src/gateway/websocket_server.py`) already runs a VAD per session
   and emits `speech_start`/`speech_end` events — the agent can listen
   for `speech_start` to interrupt TTS.

## What We Implemented

- [x] **Energy-based VAD with hysteresis** (`src/gateway/vad.py`):
  onset=0.02, offset=0.010; flapping eliminated in tests.
- [x] **Cascaded gateway pipeline** (`src/gateway/websocket_server.py`):
  WebSocket → VAD segments → STT batch transcribe per segment.
- [x] **Speech-event protocol** (`GatewaySession.speech_start` / `speech_end`):
  client gets a `speech_end` notification when an utterance is
  complete, before the agent starts processing.
- [x] **Streaming interface in `STTProvider`** (`src/stt/base.py`):
  `transcribe_stream(audio_stream)` is an async generator yielding
  partial results; not exercised end-to-end yet but the contract
  exists.
- [x] **Barge-in interrupt** (`ConversationOrchestrator.interrupt()`):
  halts SPEAKING/GENERATING states; the next utterance can begin
  immediately.

## What We Did NOT Implement

- **True streaming Whisper (whisper_streaming with local attention)** —
  the open-source variant requires custom attention masking and
  a sliding window; for the MVP, batch transcribe per VAD segment
  is sufficient.
- **End-of-utterance confidence floor** — we rely entirely on VAD
  silence detection. Adding a confidence threshold would catch
  truncated utterances that VAD misses (e.g. trailing breaths).
- **Echo cancellation** — without AEC, the VAD will false-positive
  on the agent's own TTS output during a half-duplex turn. Mitigated
  for now by muting the VAD during TTS playback (see
  `GatewaySession.set_speaking()`); production needs proper AEC.
- **VAD model selection experiment** — WebRTC VAD vs. Silero vs.
  energy-based on Indian accent audio. Planned as experiment 004.

## Results

- Our energy-based VAD correctly identifies speech segments in the
  test suite (`tests/test_gateway.py` — 33 tests including VAD
  flapping, silence detection, and multi-segment streams).
- Batch transcribe-per-segment introduces ~300-500ms latency vs.
  true streaming, but is acceptable for our P50 < 3s target.
- Barge-in works for hard interrupts (button press); soft
  interruption (user starts talking during TTS) is gated on the VAD
  remaining active during playback, which is a TODO.

## Inspiration Statement

Cascaded VAD-first streaming is the production pattern; we ship a
simplified version (energy VAD + batch per segment) that meets our
latency targets and gives us the protocol hooks for full streaming
later.
