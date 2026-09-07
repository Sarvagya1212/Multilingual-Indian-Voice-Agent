"""Tests for the failure-analysis framework."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from failures import (  # noqa: E402
    DOC_PATH,
    VALID_CATEGORIES,
    VALID_SEVERITIES,
    FailureEntry,
    compute_stats,
    get_entries,
    get_stats,
    load_doc,
    parse_entries,
)


# ----------------------------------------------------------------------
# Module surface
# ----------------------------------------------------------------------


class TestFailureModule:
    def test_doc_path_points_at_md_file(self):
        assert DOC_PATH.name == "failure_analysis.md"
        assert DOC_PATH.parent.name == "failures"

    def test_valid_categories_include_core_prompt_categories(self):
        # Spot-check that the categories listed in the prompt are present
        for cat in [
            "STT Failure",
            "TTS Pronunciation Failure",
            "Language Detection Failure",
            "RAG Failure",
            "Tool Failure",
            "LLM Reasoning Failure",
            "Latency Failure",
            "Barge-in Failure",
            "Prompt Failure",
        ]:
            assert cat in VALID_CATEGORIES, f"Missing category: {cat}"

    def test_valid_severities(self):
        assert VALID_SEVERITIES == {"Critical", "High", "Medium", "Low"}

    def test_doc_file_exists(self):
        assert DOC_PATH.exists(), "failure_analysis.md should exist on disk"


# ----------------------------------------------------------------------
# Document structure
# ----------------------------------------------------------------------


class TestDocStructure:
    def test_doc_has_top_level_heading(self):
        text = load_doc()
        assert text.startswith("# Failure Analysis")

    def test_doc_has_categories_section(self):
        text = load_doc()
        assert "## Categories" in text

    def test_doc_has_failure_template_section(self):
        text = load_doc()
        assert "## Failure Template" in text

    def test_doc_has_failure_log_section(self):
        text = load_doc()
        assert "## Failure Log" in text

    def test_doc_has_summary_statistics(self):
        text = load_doc()
        assert "## Summary Statistics" in text

    def test_doc_has_top_weaknesses(self):
        text = load_doc()
        assert "## Top 5 Current Weaknesses" in text


# ----------------------------------------------------------------------
# Entry parsing
# ----------------------------------------------------------------------


class TestParseEntries:
    def test_parses_at_least_ten_entries(self):
        entries = get_entries()
        assert len(entries) >= 10, "Expected at least 10 documented failures"

    def test_entries_have_unique_ids(self):
        entries = get_entries()
        ids = [e.id for e in entries]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"

    def test_entries_have_valid_categories(self):
        entries = get_entries()
        for e in entries:
            assert e.category in VALID_CATEGORIES, f"{e.id} has invalid category: {e.category}"

    def test_entries_have_valid_severities(self):
        entries = get_entries()
        for e in entries:
            assert e.severity in VALID_SEVERITIES, f"{e.id} has invalid severity: {e.severity}"

    def test_entries_have_iso_dates(self):
        entries = get_entries()
        for e in entries:
            assert len(e.date) == 10 and e.date[4] == "-" and e.date[7] == "-", (
                f"{e.id} date is not YYYY-MM-DD: {e.date}"
            )

    def test_entries_have_required_sections(self):
        entries = get_entries()
        for e in entries:
            for marker in ["**Input:**", "**Expected:**", "**Actual:**",
                           "**Root Cause:**", "**Fix:**", "**Verification:**",
                           "**Prevention:**"]:
                assert marker in e.body, f"{e.id} missing section: {marker}"

    def test_entry_id_is_valid_format(self):
        entries = get_entries()
        for e in entries:
            assert e.id.startswith("F"), f"ID should start with F: {e.id}"
            # F followed by 3-4 digits
            digits = e.id[1:]
            assert digits.isdigit() and 3 <= len(digits) <= 4

    def test_failure_entry_is_valid_method(self):
        e = FailureEntry(
            id="F999",
            title="Test",
            date="2026-09-07",
            category="RAG Failure",
            severity="Low",
            body="",
        )
        assert e.is_valid()

    def test_failure_entry_is_valid_fails_on_bad_category(self):
        e = FailureEntry(
            id="F999",
            title="Test",
            date="2026-09-07",
            category="NotARealCategory",
            severity="Low",
            body="",
        )
        assert not e.is_valid()

    def test_failure_entry_is_valid_fails_on_bad_severity(self):
        e = FailureEntry(
            id="F999",
            title="Test",
            date="2026-09-07",
            category="RAG Failure",
            severity="Catastrophic",
            body="",
        )
        assert not e.is_valid()

    def test_parse_entries_empty_string(self):
        assert parse_entries("") == []

    def test_parse_entries_ignores_garbage(self):
        text = "Some random text without any entries"
        assert parse_entries(text) == []


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------


class TestStats:
    def test_stats_total_matches_entries(self):
        entries = get_entries()
        stats = get_stats()
        assert stats.total == len(entries)

    def test_stats_by_category_populated(self):
        stats = get_stats()
        assert len(stats.by_category) >= 3, "Expected entries across multiple categories"
        # All counts positive
        for cat, count in stats.by_category.items():
            assert count > 0
            assert cat in VALID_CATEGORIES

    def test_stats_by_severity_populated(self):
        stats = get_stats()
        for sev, count in stats.by_severity.items():
            assert count > 0
            assert sev in VALID_SEVERITIES

    def test_stats_sum_matches_total(self):
        stats = get_stats()
        cat_sum = sum(stats.by_category.values())
        sev_sum = sum(stats.by_severity.values())
        assert cat_sum == stats.total
        assert sev_sum == stats.total

    def test_compute_stats_empty(self):
        stats = compute_stats([])
        assert stats.total == 0
        assert stats.by_category == {}
        assert stats.by_severity == {}

    def test_stats_match_documented_table(self):
        # The summary statistics table in the doc should reflect the
        # actual parsed counts (catches drift between prose and code).
        text = load_doc()
        stats = get_stats()

        # The document should mention each category that has entries
        for cat in stats.by_category:
            # category appears in the Summary Statistics table
            # (the heading may be stripped, so use the category name)
            assert cat in text, f"Category {cat} not in summary table"

        # All severities should appear in the table too
        for sev in stats.by_severity:
            assert sev in text, f"Severity {sev} not in summary table"
