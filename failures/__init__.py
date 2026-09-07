"""Failure-analysis package.

Programmatic helpers for the failure-analysis framework. The primary
artifact is `failure_analysis.md` (human-readable); this module lets
tests verify the document is well-formed and aggregate statistics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

DOC_PATH = Path(__file__).parent / "failure_analysis.md"

VALID_CATEGORIES = {
    "STT Failure",
    "TTS Pronunciation Failure",
    "Language Detection Failure",
    "RAG Failure",
    "Tool Failure",
    "LLM Reasoning Failure",
    "Latency Failure",
    "Barge-in Failure",
    "Prompt Failure",
    "Test Infrastructure Failure",
    "Build/Infrastructure Failure",
    "Config Failure",
}

VALID_SEVERITIES = {"Critical", "High", "Medium", "Low"}

# A failure entry starts with "### F0NN:" and ends at the next "---"
# Accepts blank lines between the title line and the metadata fields.
ENTRY_PATTERN = re.compile(
    r"###\s+(F\d{3,4}):\s+(?P<title>.+?)\n"
    r"(?:\n)?"  # optional blank line after title
    r"\*\*Date:\*\*\s+(?P<date>\d{4}-\d{2}-\d{2})\s*\n"
    r"\*\*Category:\*\*\s+(?P<category>.+?)\s*\n"
    r"\*\*Severity:\*\*\s+(?P<severity>.+?)\s*\n",
    re.MULTILINE,
)


@dataclass
class FailureEntry:
    """A single parsed failure record."""

    id: str
    title: str
    date: str
    category: str
    severity: str
    body: str

    def is_valid(self) -> bool:
        return (
            self.id.startswith("F")
            and self.category in VALID_CATEGORIES
            and self.severity in VALID_SEVERITIES
        )


@dataclass
class FailureStats:
    """Aggregated statistics over all parsed failure entries."""

    by_category: Dict[str, int] = field(default_factory=dict)
    by_severity: Dict[str, int] = field(default_factory=dict)
    total: int = 0


def parse_entries(doc_text: str) -> List[FailureEntry]:
    """Parse failure entries from a failure_analysis.md document body."""
    entries: List[FailureEntry] = []
    matches = list(ENTRY_PATTERN.finditer(doc_text))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc_text)
        body = doc_text[start:end]
        entries.append(
            FailureEntry(
                id=m.group(1),
                title=m.group("title").strip(),
                date=m.group("date"),
                category=m.group("category").strip(),
                severity=m.group("severity").strip(),
                body=body,
            )
        )
    return entries


def compute_stats(entries: List[FailureEntry]) -> FailureStats:
    stats = FailureStats()
    stats.total = len(entries)
    for e in entries:
        stats.by_category[e.category] = stats.by_category.get(e.category, 0) + 1
        stats.by_severity[e.severity] = stats.by_severity.get(e.severity, 0) + 1
    return stats


def load_doc(path: Optional[Path] = None) -> str:
    """Load the failure-analysis document text."""
    p = path or DOC_PATH
    return p.read_text(encoding="utf-8")


def get_entries(path: Optional[Path] = None) -> List[FailureEntry]:
    """Load and parse all failure entries from the default document."""
    return parse_entries(load_doc(path))


def get_stats(path: Optional[Path] = None) -> FailureStats:
    return compute_stats(get_entries(path))
