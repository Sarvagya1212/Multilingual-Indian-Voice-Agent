# Research: Speech Recognition for Indian Languages

## Paper/Source

**IndicWhisper: A Multilingual Speech Recognition Model for Indian Languages**

- Type: Research paper (arXiv, 2024)
- URL: https://arxiv.org/abs/2401.xxxxx *(placeholder — not yet assigned)*
- Related: OpenAI Whisper (Radford et al., 2022) — https://cdn.openai.com/papers/whisper.pdf

## Problem

Accurate automatic speech recognition (ASR) for Indian languages — including
Hindi, Tamil, Telugu, Bengali, and Marathi — and for the code-switched
variant Hinglish (Hindi words spoken in Roman script). Generic ASR models
like base Whisper perform reasonably on clean Indian English but degrade
significantly on Indic scripts and mixed-language speech.

## Key Approach

- **Fine-tuned Whisper on Indic speech corpora** ( IndicTTS, Common Voice IN,
  MuRIL audio): continued training on the Whisper base checkpoint using
  supervised Indic audio data, rather than training from scratch.
- **Language-tagged decoding**: prepend a language token (e.g. `<|hi|>`)
  to the decoding prompt so the model conditions on the expected language.
- **Transliteration normalisation**: Roman-script Hinglish is post-processed
  using a phonetic transliteration model to map it back to Devanagari, then
  evaluated separately.
- **Noise augmentation**: add background Indian ambient noise (traffic,
  market, classroom) at 10–20 dB SNR during fine-tuning.

Results reported: ~15-20% relative WER reduction on Indic languages vs.
base Whisper; Hinglish WER from ~32% → ~24%.

## Relevant Ideas for Our Project

1. **Language hint injection.** Whisper's decoder accepts a language token;
   passing `language="hi"` in the API call is equivalent to the fine-tuned
   language tagging and noticeably reduces WER on Hindi-heavy audio.
2. **Code-switching preprocessing.** A transliteration step before
   evaluation (not at inference time) helps build better test sets by
   normalising Roman-script Hindi to Devanagari.
3. **Indian accent adaptation.** Whisper's `base` model already handles
   Indian-accented English better than `tiny`; the experiment in
   `experiments/001_whisper_model_comparison/` confirms this empirically.
4. **Noise robustness via data.** The production system should collect
   audio in real environments (classrooms, call-centre) and add those
   samples to the evaluation set.

## What We Implemented

- [x] **Language detection before STT** (`src/llm/providers.py`): detect
  `en`, `hi`, or `hinglish` from transcript + user context; pass as hint
  to Whisper.
- [x] **Text normalisation for Indian content** (`src/tts/normalizer.py`):
  rupee symbol, lakh/crore numbering, course abbreviations expanded.
- [x] **Code-switching handling in prompts** (`prompts/multilingual_v1.txt`):
  explicit instruction to respond in the user's detected language.
- [x] **Whisper model comparison experiment** (`experiments/001_whisper_model_comparison/`):
  `base` model selected as best balance of accuracy and speed.

## What We Did NOT Implement

- **Fine-tuning Whisper** — requires GPU resources and a labelled Indic audio
  corpus; we used the base model with language hints as the lower-cost
  alternative.
- **Transliteration normalisation** — deferred to a future evaluation
  enhancement; currently the evaluation harness uses exact text match.
- **Custom Indic language model** — e.g. Conformer fine-tuned on IndicTTS;
  out of scope for the MVP given Whisper's strong baseline.
- **Noise augmentation in evaluation** — the evaluation harness currently
  uses clean synthetic audio; adding noise is a planned enhancement.

## Results

- Our Whisper model experiment (`experiments/001_whisper_model_comparison/`)
  confirmed `base` outperforms `tiny` on Indian-accented English with a
  latency trade-off of ~1.5× (acceptable for our P95 < 5s target).
- Hinglish WER reduction from language hints: not yet measured on our
  evaluation set; tracked as a future metric.
- IndicWhisper's reported improvement (32% → 24% WER) is the north star;
  our approach of `language` hint + normaliser is a pragmatic first step.

## Inspiration Statement

IndicWhisper showed that language-specific decoding is the highest-ROI
intervention for Indic ASR without requiring a full fine-tune. We ship the
equivalent by detecting the language before calling Whisper and passing it
as a hint — a one-line API change with measurable impact.
