# Evaluation Framework Documentation

## Overview

This document defines the evaluation metrics, datasets, and methodology used to measure system performance.

## Metrics by Component

### Speech-to-Text (STT)

| Metric | Definition | Target | Measurement |
|--------|------------|--------|-------------|
| WER (Word Error Rate) | Standard word error metric | < 10% (English), < 15% (Hindi), < 20% (Hinglish) | Character-level Levenshtein distance |
| CER (Character Error Rate) | Character-level error rate | - | Used for language detection |
| Language Accuracy | Correct language detection | > 90% | Manual verification |
| Code-switching Accuracy | Handle Hindi+English mixed speech | > 80% | Specialized test sets |
| Latency | Time from audio end to text output | < 500ms | Wall-clock timing |
| Real-time Factor (RTF) | Processing time / audio duration | < 0.5 | Measure on device |

#### WER Formula

```
WER = (S + D + I) / N
Where:
S = Substitutions
D = Deletions
I = Insertions
N = Number of reference words
```

### Text-to-Speech (TTS)

| Metric | Definition | Target | Measurement |
|--------|------------|--------|-------------|
| Naturalness MOS | Mean Opinion Score (1-5 scale) | > 4.0 | Human evaluation |
| Latency (to-first-audio) | Time from text to first audio byte | < 500ms | Wall-clock timing |
| Pronunciation Accuracy | Correct pronunciation of Hindi/English/names | > 95% | Manual verification |
| Duration Accuracy | Output duration matches text duration | - | Automated check |

### Retrieval-Augmented Generation (RAG)

| Metric | Definition | Target | Measurement |
|--------|------------|--------|-------------|
| Retrieval Recall | Relevant docs retrieved | > 80% | Binary relevance |
| Context Relevance | Scores of retrieved chunks | > 0.7 | Semantic similarity |
| Groundedness | Answer supported by context | > 85% | Human evaluation |
| Hallucination Rate | Unsupported claims in answer | < 10% | Manual review |
| Latency | Query to response time | < 2s | Wall-clock timing |

### Conversational Agent

| Metric | Definition | Target | Measurement |
|--------|------------|--------|-------------|
| Task Completion | Goal achieved | > 90% | Annotation |
| Tool Call Accuracy | Correct tools invoked | > 95% | Schema validation |
| Response Relevance | On-topic and helpful | > 85% | Human evaluation |
| Turn Count | Conversations completed | ≤ 6 turns | Automatic |
| Barge-in Success | Interruption handled | > 95% | Test scenarios |

### Latency Budget

| Component | Target Latency |
|-----------|---------------|
| Total E2E | < 3000ms |
| STT | < 500ms |
| LLM (thinking) | < 1500ms |
| TTS | < 500ms to first audio |
| Tool call | < 200ms |
| RAG retrieval | < 500ms |
| Interruption detection | < 50ms |

## Test Datasets

### STT Test Dataset (`datasets/stt_test.json`)

```json
{
  "samples": [
    {
      "audio": "path/to/audio.wav",
      "reference": "Ground truth transcript",
      "language": "en|hi|eninglish",
      "accent": "indi",
      "noise_level": "clean|low|medium|high",
      "code_switching": true,
      "speaker_id": "speaker_001",
      "category": "greeting|inquiry|course|eligibility|fees"
    }
  ]
}
```

### Categories for STT Evaluation

- **Clean Speech**: Studio-quality recordings
- **Phone Microphone**: Smartphone recorder
- **Laptop Microphone**: Built-in mic
- **Background Noise**: Cafe, bus, home environment
- **Indian English**: Accent variations
- **Hindi**: Native Hindi speakers
- **Hinglish**: Mixed Hindi-English speech
- **Code-switching**: Numbers, names, technical terms mid-sentence
- **Fast/Slow Speech**: Different speaking rates

### RAG Evaluation Dataset (`datasets/rag_eval.json`)

```json
{
  "queries": [
    {
      "query": "JEE ke fees kya hote hain?",
      "golden_context": ["JEE course fee is ₹1,50,000 per year"],
      "expected_answer": "The JEE course fee is...",
      "relevant_docs": ["jee_course_001"],
      "category": "fees|eligibility|schedule"
    }
  ]
}
```

### Tool Call Evaluation Dataset

```json
{
  "test_cases": [
    {
      "intent": "course_inquiry",
      "utterance": "JEE kya hai?",
      "expected_tool": "search_courses",
      "expected_args": {"query": "JEE"},
      "expected_result_keys": ["id", "name", "category"]
    }
  ]
}
```

## Evaluation Commands

```bash
# Run all evaluations
python -m evaluations.run

# Run specific component evaluation
python -m evaluations.stt.evaluate
python -m evaluations.tts.evaluate
python -m evaluations.rag.evaluate
python -m evaluations.agent.evaluate

# Generate HTML report
python -m evaluations.report --html

# Compare with baseline
python -m evaluations.compare --baseline experiments/001_baselinedata.json
```

## Scoring Rubrics

### Response Naturalness (1-5 scale)

| Score | Criteria |
|-------|----------|
| 5 | Very natural, conversational, concise |
| 4 | Natural with minor issues |
| 3 | Understandable but rigid |
| 2 | Robotic, needs improvement |
| 1 | Very awkward, hard to understand |

### Groundedness (1-5 scale)

| Score | Criteria |
|-------|----------|
| 5 | Fully supported by retrieved context |
| 4 | Mostly supported, minor speculation |
| 3 | Partially supported |
| 2 | Significant speculation |
| 1 | Mostly hallucinated |

## Continuous Evaluation

- Run evaluation on every commit via CI
- Store results in `evaluations/reports/`
- Compare metrics against baseline
- Alert on significant degradation

## Baseline Metrics

Current baseline (expected from Prompt 1.5):

| Component | Baseline Metric |
|-----------|-----------------|
| STT WER | TBA |
| TTS Latency | TBA |
| Tool Accuracy | TBA |
| RAG Groundedness | TBA |

Update this table after completing Prompt 1.5.