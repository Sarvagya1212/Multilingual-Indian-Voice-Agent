"""Tests for the demo scripts.

Exercises each demo's main() function in-process and verifies:
1. It doesn't raise an exception.
2. It produces expected output lines.
3. The run_all.py orchestrator runs all demos successfully.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DEMOS_DIR = ROOT / "demos"

DEMOS = [
    "demo_01_english",
    "demo_02_hindi",
    "demo_03_hinglish",
    "demo_04_tool_calling",
    "demo_05_rag",
    "demo_06_interruption",
    "demo_07_failure_recovery",
    "demo_08_dashboard",
]


# ----------------------------------------------------------------------
# Module-level checks
# ----------------------------------------------------------------------


class TestDemosExist:
    def test_all_demo_scripts_exist(self):
        for name in DEMOS:
            path = DEMOS_DIR / f"{name}.py"
            assert path.exists(), f"{path} is missing"

    def test_demos_have_readme(self):
        assert (DEMOS_DIR / "README.md").exists()

    def test_demos_have_run_all(self):
        assert (DEMOS_DIR / "run_all.py").exists()

    def test_demos_have_common_helper(self):
        assert (DEMOS_DIR / "_common.py").exists()

    def test_common_has_api_key_checker(self):
        text = (DEMOS_DIR / "_common.py").read_text(encoding="utf-8")
        assert "def has_api_key" in text

    def test_common_has_synthetic_wav(self):
        text = (DEMOS_DIR / "_common.py").read_text(encoding="utf-8")
        assert "def create_synthetic_wav" in text


# ----------------------------------------------------------------------
# Each demo's main() runs without error
# ----------------------------------------------------------------------


class TestDemoSmoke:
    """Run each demo's main() and verify it completes without exception."""

    @pytest.mark.parametrize("demo", DEMOS, ids=DEMOS)
    def test_demo_main_runs(self, demo: str):
        # Suppress structured-log INFO lines from src modules so they don't
        # pollute test output. Capture stderr too (logger writes there).
        env = {**subprocess.os.environ.copy()}
        env["LOG_LEVEL"] = "WARNING"
        # Force UTF-8 stdout/stderr so Devanagari / Hindi text in the demo
        # output doesn't crash the parent's cp1252 reader thread.
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "-c", f"""
import sys
sys.path.insert(0, r'{ROOT}')
import asyncio
from demos.{demo} import main
asyncio.run(main())
"""],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=env,
        )
        assert result.returncode == 0, (
            f"{demo} failed with:\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    @pytest.mark.parametrize("demo", DEMOS, ids=DEMOS)
    def test_demo_prints_pass_marker(self, demo: str):
        """Each demo should print '[OK]' on completion."""
        env = {**subprocess.os.environ.copy()}
        env["LOG_LEVEL"] = "WARNING"
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "-c", f"""
import sys
sys.path.insert(0, r'{ROOT}')
import asyncio
from demos.{demo} import main
asyncio.run(main())
"""],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=env,
        )
        assert "[OK]" in result.stdout, (
            f"{demo} did not print '[OK]' marker.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )


# ----------------------------------------------------------------------
# run_all.py
# ----------------------------------------------------------------------


class TestRunAll:
    def test_run_all_script_exists_and_is_importable(self):
        sys.path.insert(0, str(ROOT))
        import demos.run_all as ra

        assert hasattr(ra, "DEMO_MODULES")
        assert len(ra.DEMO_MODULES) == 8

    def test_run_all_cli_runs_all_demos(self):
        env = {**subprocess.os.environ.copy()}
        env["LOG_LEVEL"] = "WARNING"
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, str(DEMOS_DIR / "run_all.py")],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=env,
        )
        assert result.returncode == 0, (
            f"run_all.py failed:\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
        # Summary line should report all 8 passing
        assert "8/8 passed" in result.stdout, (
            f"Expected '8/8 passed' in output.\nstdout:\n{result.stdout}"
        )

    def test_run_all_accepts_indices(self):
        env = {**subprocess.os.environ.copy()}
        env["LOG_LEVEL"] = "WARNING"
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, str(DEMOS_DIR / "run_all.py"), "1", "5"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            env=env,
        )
        assert result.returncode == 0
        assert "2 of 8 demos" in result.stdout

    def test_run_all_stop_on_failure(self):
        env = {**subprocess.os.environ.copy()}
        env["LOG_LEVEL"] = "WARNING"
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, str(DEMOS_DIR / "run_all.py"), "--stop"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=env,
        )
        assert result.returncode == 0
        assert "passed" in result.stdout


# ----------------------------------------------------------------------
# Common helpers
# ----------------------------------------------------------------------


class TestCommonHelpers:
    def test_create_synthetic_wav_produces_bytes(self):
        from demos._common import create_synthetic_wav

        audio = create_synthetic_wav(duration=1.0)
        assert isinstance(audio, bytes)
        assert len(audio) > 1000

    def test_has_api_key_returns_bool(self):
        from demos._common import has_api_key

        result = has_api_key("DOES_NOT_EXIST_xyz123")
        assert isinstance(result, bool)

    def test_scripted_response_returns_string(self):
        from demos._common import scripted_response

        result = scripted_response("What are the fees?", "en")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_scripted_response_covers_fee_query(self):
        from demos._common import scripted_response

        # fee query should return a response mentioning fees
        result = scripted_response("What is the fee for JEE?", "en")
        assert "fee" in result.lower() or "INR" in result

    def test_scripted_response_covers_eligibility_query(self):
        from demos._common import scripted_response

        result = scripted_response("Am I eligible for JEE?", "en")
        assert "eligible" in result.lower() or "Class" in result

    def test_scripted_response_covers_demo_query(self):
        from demos._common import scripted_response

        result = scripted_response("I want to schedule a demo", "en")
        assert "demo" in result.lower() or "schedule" in result.lower()
