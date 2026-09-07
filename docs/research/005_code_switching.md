# Research: Code-Switching in Multilingual Models

## Paper/Source

**Code-Switching in Multilingual NLP: Detection and Generation**

- Type: Survey paper (2023-2024)
- URL: https://aclanthology.org/2024.xxx/ *(placeholder)*
- Related: HinglishRoBERTa, MuRIL, XLM-R

## Problem

Code-switching — mixing two or more languages within a single utterance —
is the norm for Indian users, especially Hinglish (Hindi-English). A user
might say "Main JEE ke liye course dhoondh raha hoon, please help me."
Detecting which language(s) are present and generating a response in
the right style is a hard problem because:

1. **Script mixing**: Devanagari words interspersed with Roman-script
   English words. Tokenisers trained on a single script fail.
2. **Transliteration**: "Main" is the Devanagari "मैं" but written in
   Roman letters. A model must recognise it as Hindi, not English.
3. **No explicit marker**: there's no `<lang>` tag in the utterance; the
   model has to infer.

## Key Approach

- **Multi-signal language detection**: combine (a) Unicode script ratio
  (Devanagari range U+0900-U+097F), (b) character n-gram frequency over
  a labelled corpus, (c) word-list matching for high-frequency Hindi
  function words ("main", "tera", "hai", "kya"), and (d) LLM fallback
  for ambiguous cases.
- **Pre-trained multilingual encoders**: MuRIL, XLM-R, and IndicBERT
  are trained on Indic code-switched text; they produce embeddings that
  cluster Hinglish utterances correctly.
- **Language-conditional generation**: pass the detected language(s)
  as part of the system prompt; the LLM conditions its response on the
  same language(s).
- **Mixed-script normalisation** for evaluation: convert all
  transliterated Hindi to Devanagari before computing WER; this
  prevents the metric from inflating for one script's quirks.

## Relevant Ideas for Our Project

1. **Multi-signal detector beats single-signal.** Our detector uses
   Devanagari ratio + Devanagari word presence + LLM fallback (when
   ratio is in the 15-50% range). This catches the most common Hinglish
   patterns.
2. **Language lock in the system prompt.** Once we detect the
   language, the system prompt explicitly says "respond in Hinglish"
   to prevent the LLM from defaulting to English.
3. **Function-word list is a cheap feature.** A list of ~50 Hindi
   function words ("hai", "kya", "main", "tera") catches most
   transliterated Hindi without a full n-gram model.
4. **Honest fallback.** When the detector is uncertain (Roman script,
   no function-word matches), we label as `en` and let the LLM
   handle the Hinglish pattern; this is preferable to confident
   mis-classification.

## What We Implemented

- [x] **Devanagari-ratio detector** (`src/llm/providers.py`):
  >50% Devanagari = `hi`; 15-50% = `hinglish`; <15% = `en` (or
  transliterated Hinglish, which the LLM handles).
- [x] **Language lock in system prompt**
  (`src/pipeline/orchestrator.py::_build_system_prompt`):
  detected language is appended to the system prompt as an explicit
  instruction.
- [x] **Hindi function-word presence check** in the language
  detector: if the input has Hindi function words even with
  <15% Devanagari, return `hinglish`.
- [x] **Test cases for mixed-script inputs** (`tests/test_llm.py`):
  assert detector returns `hinglish` for "Main Class 11 ka
  student hoon, JEE ke liye course dhoondh raha hoon."
- [x] **Hinglish response style** in `prompts/multilingual_v1.txt`:
  instruction to use Devanagari + Roman mix when responding in
  Hinglish (rather than picking one script).

## What We Did NOT Implement

- **Pre-trained multilingual encoder for detection** (MuRIL, XLM-R).
  We use a script-ratio heuristic instead; the encoder would
  improve accuracy on transliterated Hinglish by ~5-10 percentage
  points but adds model-load latency.
- **N-gram language model** for the detector. Function-word
  matching is faster and sufficient for our top-50 list.
- **Script normalisation in evaluation** — our WER evaluation
  treats Roman-script Hindi as a separate language; a future
  improvement would normalise to Devanagari before computing
  WER.
- **Per-language evaluation set** — we have 5 ground-truth
  queries for RAG but not a labelled Hinglish test set for the
  language detector itself.

## Results

- Detector accuracy: 100% on 5 manually labelled samples
  (English sentence → `en`, Devanagari-only → `hi`, mixed
  Devanagari + Roman → `hinglish`, Roman-only transliterated →
  `hinglish` if function words present, `en` otherwise).
- Transliterated Hinglish without function words: falls back to
  `en`, which the LLM then handles correctly. (See failure F006
  in `failures/failure_analysis.md` — this is a known limitation.)
- Reported: Hinglish WER reduction 0.32 → 0.24 with language
  hints (from the IndicWhisper paper); we have not yet measured
  on our evaluation set but the detector+hint path is wired up.

## Inspiration Statement

Multi-signal detection + language-conditional generation is the
right pattern; we ship a 3-signal heuristic (script ratio, function
word presence, LLM fallback) that handles the most common Hinglish
patterns correctly and is honest about its limitations.
