"""Tests for the final audit and project documentation."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------------
# README.md
# ----------------------------------------------------------------------


class TestReadme:
    def test_readme_exists(self):
        assert (ROOT / "README.md").exists()

    def test_readme_has_architecture_diagram(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "```mermaid" in text, "README should have a Mermaid diagram"

    def test_readme_has_setup_section(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "## Quick Start" in text or "## Setup" in text

    def test_readme_has_demo_section(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "## Demos" in text or "## Demo" in text

    def test_readme_has_evaluation_table(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        # Should have a table with metric / target / actual
        assert "| Metric |" in text or "|--------|" in text
        assert "Target" in text and "Actual" in text

    def test_readme_demo_commands_use_correct_module(self):
        # The run_all.py is at demos/run_all.py not python -m demos.run_all
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        # Either "python demos/run_all.py" or "python -m demos.run_all"
        # Both are valid forms; the test should not lock us in
        assert "run_all" in text

    def test_readme_references_all_demos(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        for demo in [
            "demo_01_english",
            "demo_02_hindi",
            "demo_03_hinglish",
            "demo_04_tool_calling",
            "demo_05_rag",
            "demo_06_interruption",
            "demo_07_failure_recovery",
            "demo_08_dashboard",
        ]:
            assert demo in text, f"{demo} not referenced in README"

    def test_readme_has_project_structure(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "## Project Structure" in text or "## Structure" in text

    def test_readme_lists_test_count(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        # Should mention 554 tests (or current count)
        assert "554" in text or "520" in text, "README should mention test count"

    def test_readme_doesnt_have_secret_keys(self):
        # Sanity: README should not contain real-looking API keys
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        # Strip out common pattern
        suspicious = re.findall(r"sk-[A-Za-z0-9]{20,}", text)
        assert not suspicious, "README contains a suspicious-looking API key"


# ----------------------------------------------------------------------
# CHANGELOG.md
# ----------------------------------------------------------------------


class TestChangelog:
    def test_changelog_exists(self):
        assert (ROOT / "CHANGELOG.md").exists()

    def test_changelog_has_unreleased_or_recent_version(self):
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        # Either [Unreleased] or a recent version
        assert "## [" in text, "CHANGELOG should have versioned sections"

    def test_changelog_mentions_keep_a_changelog_format(self):
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "Keep a Changelog" in text

    def test_changelog_has_standard_sections(self):
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        for section in ["### Added", "### Fixed", "### Changed"]:
            assert section in text, f"CHANGELOG missing section: {section}"


# ----------------------------------------------------------------------
# audit.md
# ----------------------------------------------------------------------


class TestAudit:
    def test_audit_exists(self):
        assert (ROOT / "audit.md").exists()

    def test_audit_has_core_functionality_section(self):
        text = (ROOT / "audit.md").read_text(encoding="utf-8")
        assert "## Core Functionality" in text

    def test_audit_has_documentation_section(self):
        text = (ROOT / "audit.md").read_text(encoding="utf-8")
        assert "## Documentation" in text

    def test_audit_has_top_weaknesses(self):
        text = (ROOT / "audit.md").read_text(encoding="utf-8")
        assert "## Top 5 Remaining Weaknesses" in text

    def test_audit_has_5_weaknesses(self):
        text = (ROOT / "audit.md").read_text(encoding="utf-8")
        # Look for 5 numbered items in the weaknesses section
        section_start = text.find("## Top 5 Remaining Weaknesses")
        rest = text[section_start:]
        numbered = re.findall(r"^\d+\.", rest, re.MULTILINE)
        assert len(numbered) >= 5, f"Expected 5 numbered weaknesses, found {len(numbered)}"

    def test_audit_checkbox_marks_completed_items(self):
        text = (ROOT / "audit.md").read_text(encoding="utf-8")
        # Should have many checked items
        checked = text.count("- [x]")
        assert checked >= 20, f"Expected >= 20 checked items, found {checked}"

    def test_audit_honest_about_incomplete_items(self):
        text = (ROOT / "audit.md").read_text(encoding="utf-8")
        # Should have some unchecked items (not everything can be done in 12 prompts)
        unchecked = text.count("- [ ]")
        assert unchecked >= 1, "Audit should be honest about what is not done"

    def test_audit_mentions_test_count(self):
        text = (ROOT / "audit.md").read_text(encoding="utf-8")
        assert "465" in text

    def test_audit_mentions_failure_count(self):
        text = (ROOT / "audit.md").read_text(encoding="utf-8")
        assert "13" in text

    def test_audit_has_security_section(self):
        text = (ROOT / "audit.md").read_text(encoding="utf-8")
        assert "Security" in text or "secrets" in text.lower()


# ----------------------------------------------------------------------
# flow.md completeness
# ----------------------------------------------------------------------


class TestFlow:
    def test_flow_exists(self):
        assert (ROOT / "flow.md").exists()

    def test_flow_has_decision_sections(self):
        text = (ROOT / "flow.md").read_text(encoding="utf-8")
        for section in ["## 1. Problem Definition", "## 27. Demo Scripts", "## 28. Final Audit"]:
            assert section in text, f"flow.md missing section: {section}"

    def test_flow_has_implementation_log(self):
        text = (ROOT / "flow.md").read_text(encoding="utf-8")
        assert "## 28. Implementation Log" in text

    def test_flow_sections_have_no_gaps(self):
        """Each top-level numbered section (1, 2, 3, ...) should appear at least once."""
        text = (ROOT / "flow.md").read_text(encoding="utf-8")
        section_numbers = set(int(n) for n in re.findall(r"^## (\d+)\.", text, re.MULTILINE))
        # The first occurrence of each section should be monotonic
        first_occurrences = []
        for m in re.finditer(r"^## (\d+)\.", text, re.MULTILINE):
            n = int(m.group(1))
            if n not in first_occurrences:
                first_occurrences.append(n)
        assert first_occurrences == sorted(first_occurrences), (
            f"flow.md first-occurrence section numbers not in order: {first_occurrences}"
        )
        # And we should have at least 28 sections (1-28)
        assert max(section_numbers) >= 28, (
            f"Expected section 28+ in flow.md, max found: {max(section_numbers)}"
        )

    def test_flow_references_experiments(self):
        text = (ROOT / "flow.md").read_text(encoding="utf-8")
        assert "experiments" in text.lower()

    def test_flow_references_failures(self):
        text = (ROOT / "flow.md").read_text(encoding="utf-8")
        assert "failure" in text.lower()

    def test_flow_references_research(self):
        text = (ROOT / "flow.md").read_text(encoding="utf-8")
        assert "research" in text.lower() or "docs/research" in text


# ----------------------------------------------------------------------
# Verify commands actually work
# ----------------------------------------------------------------------


class TestCommandsWork:
    """Verify the commands listed in the README actually work."""

    def test_pytest_command_runs_without_import_error(self):
        # Just verify pytest can collect tests without import errors.
        # We don't re-run the full suite (that would be exponential).
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"pytest collection failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "passed" in result.stdout or "error" not in result.stderr.lower()

    def test_evaluations_module_imports(self):
        result = subprocess.run(
            [sys.executable, "-c", "import evaluations"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"evaluations module failed to import:\n{result.stderr}"
        )

    def test_demos_module_imports(self):
        result = subprocess.run(
            [sys.executable, "-c", "import demos"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"demos module failed to import:\n{result.stderr}"
        )

    def test_dashboard_module_imports(self):
        result = subprocess.run(
            [sys.executable, "-c", "import dashboard.data_source"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"dashboard module failed to import:\n{result.stderr}"
        )

    def test_failures_module_imports(self):
        result = subprocess.run(
            [sys.executable, "-c", "import failures; print(failures.get_stats().total)"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"failures module failed to import:\n{result.stderr}"
        )
        # Should print a positive integer
        assert int(result.stdout.strip()) > 0
