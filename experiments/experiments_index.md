# Experiments Index

## Summary Table

| ID | Name | Date | Status | Key Result |
|----|------|------|--------|------------|
| 001 | Whisper Model Comparison | 2026-09 | Complete | `base` model best balance |
| 002 | TTS Provider Comparison | 2026-09 | Skeleton | TBD |
| 003 | RAG Chunk Size | 2026-09 | Complete | 512-char chunks best recall/coverage |

## Status Legend

- **Skeleton**: Template and notes in place; no real run yet
- **Running**: Currently executing
- **Complete**: Results + analysis finalized
- **Abandoned**: Stopped mid-way with documented reason

## Running an Experiment

1. Read `notes.md` for the hypothesis and setup
2. Run `python <experiment>/run.py`
3. Inspect `results.json` for raw numbers
4. Update `notes.md` with the actual results and decision
5. Commit the experiment directory

## Conventions

- Experiment IDs are zero-padded 3 digits (`001`, `002`, ...)
- Names use `snake_case`
- Each experiment tests ONE hypothesis (single-variable change)
- At least 3 trials per configuration for statistical robustness
- Decision is recorded explicitly: ship, iterate, or abandon
