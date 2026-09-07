# Experiments Summary

## Experiment Index

| ID | Name | Date | Status | Hypothesis | Key Result | Decision |
|----|------|------|--------|------------|------------|----------|
| 001 | | | | | | |
| 002 | | | | | | |
| 003 | | | | | | |
| 004 | | | | | | |
| 005 | | | | | | |
| 006 | | | | | | |
| 007 | | | | | | |
| 008 | | | | | | |
| 009 | | | | | | |
| 010 | | | | | | |

## Template for New Experiments

### Experiment XXX: [Name]

**Date:** YYYY-MM-DD
**Status:** Planned / Running / Complete / Abandoned

**Hypothesis:**
[What we expect to learn or improve]

**Setup:**
- Dataset:
- Configuration:
- Baseline:

**Results:**
[Quantitative results with metrics]

**Analysis:**
[Interpretation of results]

**Decision:**
[What we'll do based on this experiment]

---

## Experiment Details

### Experiment 001: Whisper Model Comparison

**Date:** [TBD]
**Status:** [TBD]

**Hypothesis:**
Different Whisper model sizes will show different tradeoffs between accuracy and latency for Indian English/Hindi/Hinglish.

**Setup:**
- Test audio: 50 samples each of English, Hindi, Hinglish
- Models: tiny, base, small, medium
- Metrics: WER, CER, latency, RTF

**Results:**
| Model | Avg WER | Avg Latency | RTF |
|-------|---------|-------------|-----|
| tiny  | TBD     | TBD         | TBD |
| base  | TBD     | TBD         | TBD |
| small | TBD     | TBD         | TBD |
| medium| TBD     | TBD         | TBD |

**Analysis:**
[TBD]

**Decision:**
[TBD]

---

### Experiment 002: TTS Provider Comparison

**Date:** [TBD]
**Status:** [TBD]

**Hypothesis:**
Different TTS providers have different strengths for Indian languages.

**Setup:**
- Test texts: 20 sentences each in English, Hindi, Hinglish
- Providers: OpenAI TTS, others
- Metrics: MOS score, pronunciation accuracy, latency

**Results:**
[TBD]

**Analysis:**
[TBD]

**Decision:**
[TBD]

---

### Experiment 003: RAG Chunk Size Optimization

**Date:** [TBD]
**Status:** [TBD]

**Hypothesis:**
Optimal chunk size balances retrieval precision and context completeness.

**Setup:**
- Chunk sizes: 256, 512, 1024 tokens
- Overlap: 50, 100 tokens
- Metrics: Recall, groundedness, latency

**Results:**
[TBD]

**Analysis:**
[TBD]

**Decision:**
[TBD]

---

### Experiment 004: Prompt Versioning

**Date:** [TBD]
**Status:** [TBD]

**Hypothesis:**
Systematic prompt engineering improves agent performance.

**Setup:**
- Prompt versions: v1, v2, v3
- Test scenarios: 30 conversations
- Metrics: Task completion, naturalness, tool accuracy

**Results:**
[TBD]

**Analysis:**
[TBD]

**Decision:**
[TBD]

---

### Experiment 005: Barge-in Latency Optimization

**Date:** [TBD]
**Status:** [TBD]

**Hypothesis:**
VAD parameters can be tuned for faster interruption detection without false positives.

**Setup:**
- Energy threshold: 0.01, 0.02, 0.03
- Min speech: 150ms, 250ms, 350ms
- Metrics: Detection latency, false positive rate

**Results:**
[TBD]

**Analysis:**
[TBD]

**Decision:**
[TBD]

---

### Experiment 006: Language Detection Accuracy

**Date:** [TBD]
**Status:** [TBD]

**Hypothesis:**
Pre-processing audio with language detection improves STT accuracy.

**Setup:**
- With vs without language hint
- Test on code-switching samples
- Metrics: WER difference

**Results:**
[TBD]

**Analysis:**
[TBD]

**Decision:**
[TBD]

---

### Experiment 007: RAG Re-ranking

**Date:** [TBD]
**Status:** [TBD]

**Hypothesis:**
Cross-encoder re-ranking improves retrieval quality over vector search alone.

**Setup:**
- Vector search only vs + cross-encoder
- Metrics: Recall@5, Groundedness

**Results:**
[TBD]

**Analysis:**
[TBD]

**Decision:**
[TBD]

---

### Experiment 008: Intent Classification Fine-tuning

**Date:** [TBD]
**Status:** [TBD]

**Hypothesis:**
Fine-tuning a small model on domain intents improves tool call accuracy.

**Setup:**
- Base model: DistilBERT
- Dataset: 500 labeled examples
- Baseline: Zero-shot prompting

**Results:**
[TBD]

**Analysis:**
[TBD]

**Decision:**
[TBD]

---

### Experiment 009: Streaming STT

**Date:** [TBD]
**Status:** [TBD]

**Hypothesis:**
Streaming STT reduces time-to-first-token vs batch processing.

**Setup:**
- Batch vs streaming Whisper
- Metrics: Latency, partial result accuracy

**Results:**
[TBD]

**Analysis:**
[TBD]

**Decision:**
[TBD]

---

### Experiment 010: Additional Indian Language Support

**Date:** [TBD]
**Status:** [TBD]

**Hypothesis:**
Adding Tamil/Telugu support is feasible with current architecture.

**Setup:**
- Language detection for Dravidian languages
- TTS voice availability
- Evaluation on available data

**Results:**
[TBD]

**Analysis:**
[TBD]

**Decision:**
[TBD]

---

## Running Experiments

```bash
# Run single experiment
python -m experiments.001_whisper_comparison.run

# Run all experiments
python -m experiments.run_all

# Generate comparison report
python -m experiments.generate_report
```

## Experiment Structure

```
experiments/
├── 001_whisper_comparison/
│   ├── config.json
│   ├── results.json
│   ├── notes.md
│   └── run.py
├── 002_tts_comparison/
└── ...
```

Each experiment should have:
1. `config.json` - Reproducible configuration
2. `run.py` - Executable experiment code
3. `results.json` - Raw results
4. `notes.md` - Analysis and decisions