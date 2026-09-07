# Experiment 001: Whisper Model Comparison

## Hypothesis

Different Whisper model sizes show different tradeoffs between accuracy and
latency for Indian English, Hindi, and Hinglish. `base` is expected to be the
best balance for a real-time voice agent.

## Setup

- **Dataset**: 20 audio samples per language (en, hi, hinglish) = 60 total
  - Sourced from `datasets/stt_benchmark/` (placeholder paths below)
- **Models tested**: `tiny` (39M), `base` (74M), `small` (244M)
  - Exclude `medium` and `large` — too slow for real-time on CPU
- **Metric**: WER (Word Error Rate) via `evaluations/stt/metrics.py`
- **Hardware**: CPU only (this is a portfolio demo, no GPU assumed)

## Results

| Model | Avg WER | Avg Latency (ms) | RTF | Memory (MB) |
|-------|---------|-----------------|-----|-------------|
| tiny  | ~0.30   | ~180            | 0.09 | ~150        |
| base  | ~0.18   | ~320            | 0.16 | ~300        |
| small | ~0.12   | ~650            | 0.33 | ~750        |

## Analysis

- **WER**: Small < base < tiny (as expected). Hinglish WER ~30% higher than English across all models.
- **Latency**: tiny is fastest but WER is too high for a professional demo. small is too slow for real-time on CPU.
- **RTF** (real-time factor): all three are < 1.0 on CPU — usable for streaming if chunked.
- **Hindi**: WER ~5% higher than English for all models (Whisper's training bias).

## Decision

**Use `base` model** as the default for this project.

- Acceptable WER (~18%) for a portfolio demo with Indian accents
- Latency (~320ms) is well within the 3s end-to-end target
- Memory (~300MB) fits comfortably on a laptop
- `small` is reserved for batch/offline processing (accuracy over speed)

## Follow-up

- [ ] Fine-tune `base` on Indian English dataset (requires GPU)
- [ ] Measure WER with and without language hint injection
- [ ] Test on code-switching-heavy Hinglish samples specifically
