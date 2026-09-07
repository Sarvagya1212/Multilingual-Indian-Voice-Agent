# Experiment 002: TTS Provider Comparison

## Hypothesis

Different TTS providers (OpenAI TTS, Edge TTS) have different strengths for
Indian languages, especially in pronouncing Indian names and educational
acronyms (JEE, NEET, IIT).

## Setup

- **Test texts**: 20 sentences each in English, Hindi, Hinglish
  - Includes course names (JEE Main, NEET UG, IIT Delhi)
  - Includes Indian phone numbers (9876543210)
  - Includes currency (Rs 1,50,000)
- **Providers tested**: `openai` (tts-1), `openai` (tts-1-hd)
- **Metric**: latency, RTF, pronunciation accuracy (manual)
- **Hardware**: API calls, network-bound

## Results

[To be filled after running experiment]

## Analysis

[To be filled after running experiment]

## Decision

[To be decided after experiment]

## Follow-up

- Test on larger set of Indian names
- Measure perceived naturalness via listening test
- Compare with Sarvam AI / Azure Indic voices
