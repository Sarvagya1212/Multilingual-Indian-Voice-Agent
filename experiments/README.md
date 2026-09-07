# Experiment Tracking

This directory contains all experiments conducted on the project.
Each experiment follows a structured format for reproducibility and comparison.

## Experiment Structure

Each experiment directory follows this structure:

```
experiment_id_name/
├── config.json      # Hyperparameters, environment, dataset
├── run.py           # Main experiment script
├── results.json     # Raw results (generated)
├── notes.md         # Analysis, conclusions, decisions
└── artifacts/      # Plots, models, etc. (generated)
```

## Quick Links

- [001: Whisper Model Comparison](./001_whisper_model_comparison/)
- [002: TTS Provider Comparison](./002_tts_provider_comparison/)
- [003: RAG Chunk Size](./003_rag_chunk_size/)

## Running Experiments

```bash
# Run all experiments
python -m experiments.run_all

# Run specific experiment
python experiments/001_whisper_model_comparison/run.py

# Generate comparison report
python -m experiments.generate_report
```

## Experiment Index

See `experiments_index.md` for the full table of all experiments.
