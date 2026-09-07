"""Tests for the research documentation."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOCS_DIR = ROOT / "docs" / "research"
# Match any numbered research doc: 001_..., 002_..., etc.
DOC_FILES = sorted(DOCS_DIR.glob("[0-9][0-9][0-9]*.md"))


# ----------------------------------------------------------------------
# Required sections per research doc
# ----------------------------------------------------------------------
REQUIRED_SECTIONS = [
    "Paper/Source",
    "Problem",
    "Key Approach",
    "Relevant Ideas for Our Project",
    "What We Implemented",
    "What We Did NOT Implement",
    "Results",
    "Inspiration Statement",
]


# ----------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------


class TestResearchDocsExist:
    def test_readme_exists(self):
        assert (DOCS_DIR / "README.md").exists()

    def test_readme_lists_all_research_docs(self):
        readme = (DOCS_DIR / "README.md").read_text(encoding="utf-8")
        for md in DOC_FILES:
            assert md.name in readme, f"{md.name} not listed in README"


class TestResearchDocCount:
    def test_at_least_five_research_docs(self):
        assert len(DOC_FILES) >= 5, f"Expected >=5 research docs, found {len(DOC_FILES)}"


class TestResearchDocStructure:
    @pytest.mark.parametrize("doc", DOC_FILES, ids=[d.name for d in DOC_FILES])
    def test_has_required_sections(self, doc: Path):
        text = doc.read_text(encoding="utf-8")
        for section in REQUIRED_SECTIONS:
            assert f"## {section}" in text, f"{doc.name}: missing section '{section}'"

    @pytest.mark.parametrize("doc", DOC_FILES, ids=[d.name for d in DOC_FILES])
    def test_has_paper_url(self, doc: Path):
        text = doc.read_text(encoding="utf-8")
        assert "URL:" in text, f"{doc.name}: missing URL field"

    @pytest.mark.parametrize("doc", DOC_FILES, ids=[d.name for d in DOC_FILES])
    def test_implemented_section_has_checkboxes(self, doc: Path):
        text = doc.read_text(encoding="utf-8")
        assert "[x]" in text or "[ ]" in text, (
            f"{doc.name}: 'What We Implemented' section should have [x] / [ ] checkboxes"
        )

    @pytest.mark.parametrize("doc", DOC_FILES, ids=[d.name for d in DOC_FILES])
    def test_not_implemented_section_not_empty(self, doc: Path):
        text = doc.read_text(encoding="utf-8")
        idx = text.find("## What We Did NOT Implement")
        assert idx >= 0, f"{doc.name}: missing 'What We Did NOT Implement'"
        # Find the next ## heading after this one
        next_heading = text.find("##", idx + 1)
        body = text[idx:next_heading if next_heading >= 0 else len(text)]
        assert len(body.strip()) > 30, (
            f"{doc.name}: 'What We Did NOT Implement' should have content"
        )

    @pytest.mark.parametrize("doc", DOC_FILES, ids=[d.name for d in DOC_FILES])
    def test_results_section_not_empty(self, doc: Path):
        text = doc.read_text(encoding="utf-8")
        idx = text.find("## Results")
        assert idx >= 0, f"{doc.name}: missing 'Results'"
        next_heading = text.find("##", idx + 1)
        body = text[idx:next_heading if next_heading >= 0 else len(text)]
        assert len(body.strip()) > 30, f"{doc.name}: 'Results' should have content"

    @pytest.mark.parametrize("doc", DOC_FILES, ids=[d.name for d in DOC_FILES])
    def test_inspiration_statement_not_empty(self, doc: Path):
        text = doc.read_text(encoding="utf-8")
        idx = text.find("## Inspiration Statement")
        assert idx >= 0, f"{doc.name}: missing 'Inspiration Statement'"
        next_heading = text.find("##", idx + 1)
        body = text[idx:next_heading if next_heading >= 0 else len(text)]
        assert len(body.strip()) > 20, (
            f"{doc.name}: 'Inspiration Statement' should have content"
        )

    @pytest.mark.parametrize("doc", DOC_FILES, ids=[d.name for d in DOC_FILES])
    def test_relevant_ideas_has_bullets(self, doc: Path):
        text = doc.read_text(encoding="utf-8")
        idx = text.find("## Relevant Ideas for Our Project")
        assert idx >= 0, f"{doc.name}: missing 'Relevant Ideas'"
        next_heading = text.find("##", idx + 1)
        body = text[idx:next_heading if next_heading >= 0 else len(text)]
        # Should have at least 2 numbered/bulleted items
        items = re.findall(r"^\d+\.", body, re.MULTILINE)
        assert len(items) >= 2, (
            f"{doc.name}: 'Relevant Ideas' should have at least 2 numbered items"
        )


class TestResearchDocFormat:
    @pytest.mark.parametrize("doc", DOC_FILES, ids=[d.name for d in DOC_FILES])
    def test_doc_starts_with_h1_heading(self, doc: Path):
        text = doc.read_text(encoding="utf-8")
        assert text.startswith("# "), f"{doc.name}: should start with a level-1 heading"

    @pytest.mark.parametrize("doc", DOC_FILES, ids=[d.name for d in DOC_FILES])
    def test_no_broken_links(self, doc: Path):
        # Check for placeholder URLs (unresolved arXiv links)
        text = doc.read_text(encoding="utf-8")
        # arxiv.org/abs/xxxxxxx without an assigned number is a placeholder
        placeholder = re.findall(r"arxiv\.org/abs/\d{7,}\*", text)
        assert not placeholder, (
            f"{doc.name}: contains unassigned arXiv placeholder links: {placeholder}"
        )

    @pytest.mark.parametrize("doc", DOC_FILES, ids=[d.name for d in DOC_FILES])
    def test_references_project_files(self, doc: Path):
        # At least one doc should reference our actual project files
        # (src/, experiments/, evaluations/, prompts/, failures/)
        text = doc.read_text(encoding="utf-8")
        refs = re.findall(r"`(?:src|experiments|evaluations|prompts|failures|dashboard)/[^`]*`", text)
        assert len(refs) >= 1, (
            f"{doc.name}: should reference actual project files (src/, experiments/, etc.)"
        )

    @pytest.mark.parametrize("doc", DOC_FILES, ids=[d.name for d in DOC_FILES])
    def test_has_paper_source_type(self, doc: Path):
        text = doc.read_text(encoding="utf-8")
        # Should have one of: Research paper, Industry blog, Survey, etc.
        valid_types = [
            "Research paper",
            "Industry blog",
            "Survey",
            "Technical report",
            "Documentation",
        ]
        has_type = any(t in text for t in valid_types)
        assert has_type, f"{doc.name}: should specify a source type (e.g. 'Research paper')"

    @pytest.mark.parametrize("doc", DOC_FILES, ids=[d.name for d in DOC_FILES])
    def test_all_implemented_checkboxes_are_ticked(self, doc: Path):
        """All [ ] items in What We Implemented should be [x] (nothing deferred there)."""
        text = doc.read_text(encoding="utf-8")
        idx = text.find("## What We Implemented")
        next_heading = text.find("##", idx + 1)
        body = text[idx:next_heading if next_heading >= 0 else len(text)]
        unchecked = re.findall(r"\[[ x]\]", body)
        # We allow [x] and nothing else
        assert all(item.strip() == "[x]" for item in unchecked), (
            f"{doc.name}: all items in 'What We Implemented' should be [x], "
            f"not [ ]. Found: {unchecked}"
        )


class TestReadme:
    def test_readme_has_table(self):
        readme = (DOCS_DIR / "README.md").read_text(encoding="utf-8")
        assert "| # |" in readme or "|---|" in readme, (
            "README should have a table of contents"
        )

    def test_readme_lists_key_takeaways(self):
        readme = (DOCS_DIR / "README.md").read_text(encoding="utf-8")
        assert "## Key Takeaways" in readme

    def test_readme_describes_research_process(self):
        readme = (DOCS_DIR / "README.md").read_text(encoding="utf-8")
        assert "## Research Process" in readme

    def test_readme_distinguishes_inspiration_from_citation(self):
        readme = (DOCS_DIR / "README.md").read_text(encoding="utf-8")
        assert "inspiration" in readme.lower() and "citation" in readme.lower()
