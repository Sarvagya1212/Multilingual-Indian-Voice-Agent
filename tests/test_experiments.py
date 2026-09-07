"""Tests for the experiment tracking system."""
import json
from pathlib import Path

import pytest

from experiments.run_all import EXPERIMENTS


class TestExperimentStructure:
    """Verify each experiment has required files."""

    @pytest.mark.parametrize("exp_id", EXPERIMENTS)
    def test_experiment_dir_exists(self, exp_id):
        exp_dir = Path(__file__).parent.parent / "experiments" / exp_id
        assert exp_dir.exists(), f"Experiment directory {exp_id} does not exist"

    @pytest.mark.parametrize("exp_id", EXPERIMENTS)
    def test_has_notes_md(self, exp_id):
        exp_dir = Path(__file__).parent.parent / "experiments" / exp_id
        assert (exp_dir / "notes.md").exists(), f"{exp_id}/notes.md missing"

    @pytest.mark.parametrize("exp_id", EXPERIMENTS)
    def test_has_config_json(self, exp_id):
        exp_dir = Path(__file__).parent.parent / "experiments" / exp_id
        config = exp_dir / "config.json"
        assert config.exists(), f"{exp_id}/config.json missing"
        # Valid JSON
        data = json.loads(config.read_text())
        assert "experiment_id" in data
        assert "status" in data

    @pytest.mark.parametrize("exp_id", EXPERIMENTS)
    def test_has_run_py(self, exp_id):
        exp_dir = Path(__file__).parent.parent / "experiments" / exp_id
        assert (exp_dir / "run.py").exists(), f"{exp_id}/run.py missing"


class TestRunAll:
    """Test the run_all module."""

    def test_experiments_list_not_empty(self):
        assert len(EXPERIMENTS) >= 3

    def test_experiments_have_descriptive_ids(self):
        for exp_id in EXPERIMENTS:
            assert exp_id.startswith("00"), f"Experiment ID should start with 00: {exp_id}"
            parts = exp_id.split("_", 1)
            assert len(parts) == 2, f"Experiment ID should be NN_name: {exp_id}"


class TestExperimentConfig:
    """Test experiment config files have required fields."""

    @pytest.mark.parametrize("exp_id", EXPERIMENTS)
    def test_config_has_required_fields(self, exp_id):
        exp_dir = Path(__file__).parent.parent / "experiments" / exp_id
        config = json.loads((exp_dir / "config.json").read_text())
        assert "experiment_id" in config
        assert "hypothesis" in config
        assert "date" in config
        assert "status" in config

    def test_all_statuses_are_valid(self):
        for exp_id in EXPERIMENTS:
            exp_dir = Path(__file__).parent.parent / "experiments" / exp_id
            config = json.loads((exp_dir / "config.json").read_text())
            valid = {"skeleton", "running", "complete", "abandoned"}
            assert config["status"] in valid, f"{exp_id} has invalid status"


class TestExperiment003:
    """Integration test for experiment 003 (RAG chunk size) — self-contained."""

    @pytest.mark.asyncio
    async def test_chunk_size_experiment_runs(self):
        from experiments.run_all import _run_experiment

        exp_id = "003_rag_chunk_size"
        result = _run_experiment(exp_id)
        assert result["id"] == exp_id
        assert "status" in result

    def test_chunk_size_experiment_is_complete(self):
        exp_dir = Path(__file__).parent.parent / "experiments" / "003_rag_chunk_size"
        config = json.loads((exp_dir / "config.json").read_text())
        # Updated to "complete" after running the experiment
        assert config["status"] == "complete"
