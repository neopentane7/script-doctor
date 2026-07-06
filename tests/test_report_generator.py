"""Tests for the HTML report generator."""

import pytest
from report.generator import generate_report, _score_color, DIMENSION_LABELS, DIMENSION_KEYS


# ── Helper fixtures ───────────────────────────────────────────────────────────

SAMPLE_ITERATION = {
    "scores": {
        "dialogue": 7,
        "pacing": 6,
        "character_consistency": 8,
        "theme": 7,
        "tension": 6,
        "overall": 7,
    },
    "critique_text": "Good dialogue but pacing needs work.",
    "improvements": ["Tighten act two.", "Add subtext to Sarah's lines."],
    "defense_notes": "Kept the long monologue for emotional weight.",
}


def _make_history(n: int = 2) -> list:
    """Generate n iterations of sample data."""
    return [SAMPLE_ITERATION for _ in range(n)]


# ── _score_color() ────────────────────────────────────────────────────────────

class TestScoreColor:
    """Verify the score-to-color mapping helper."""

    def test_high_score_green(self):
        assert _score_color(8) == "#22c55e"
        assert _score_color(9) == "#22c55e"
        assert _score_color(10) == "#22c55e"

    def test_mid_score_yellow(self):
        assert _score_color(6) == "#eab308"
        assert _score_color(7) == "#eab308"

    def test_low_score_red(self):
        assert _score_color(0) == "#ef4444"
        assert _score_color(5) == "#ef4444"
        assert _score_color(1) == "#ef4444"


# ── generate_report() — HTML output ──────────────────────────────────────────

class TestGenerateReport:
    """Verify the report generator produces valid, complete HTML."""

    def test_returns_html_string_when_save_false(self):
        """When save=False, should return raw HTML string."""
        html = generate_report(
            prompt="Test prompt",
            final_script="INT. ROOM - DAY\nHello world.",
            iteration_history=_make_history(2),
            final_score=7,
            timestamp="2026-01-01 12:00:00",
            save=False,
        )
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")

    def test_html_contains_prompt(self):
        """The original prompt should appear in the report."""
        prompt = "A vampire and a hunter in a cathedral"
        html = generate_report(
            prompt=prompt,
            final_script="INT. CATHEDRAL - NIGHT",
            iteration_history=_make_history(1),
            final_score=8,
            timestamp="2026-01-01 12:00:00",
            save=False,
        )
        assert prompt in html

    def test_html_contains_final_score(self):
        """The final score should appear in the report."""
        html = generate_report(
            prompt="Test",
            final_script="Script content",
            iteration_history=_make_history(1),
            final_score=9,
            timestamp="2026-01-01 12:00:00",
            save=False,
        )
        assert "9/10" in html

    def test_html_contains_iteration_count(self):
        """Number of iterations should appear in the report."""
        html = generate_report(
            prompt="Test",
            final_script="Script content",
            iteration_history=_make_history(3),
            final_score=7,
            timestamp="2026-01-01 12:00:00",
            save=False,
        )
        assert "3" in html  # iteration count

    def test_html_contains_chart_js(self):
        """Report should include Chart.js for visualizations."""
        html = generate_report(
            prompt="Test",
            final_script="Script",
            iteration_history=_make_history(1),
            final_score=7,
            timestamp="2026-01-01 12:00:00",
            save=False,
        )
        assert "chart.js" in html.lower() or "Chart" in html

    def test_html_contains_radar_chart(self):
        """Report should contain the radar chart canvas."""
        html = generate_report(
            prompt="Test",
            final_script="Script",
            iteration_history=_make_history(1),
            final_score=7,
            timestamp="2026-01-01 12:00:00",
            save=False,
        )
        assert "radarChart" in html

    def test_html_contains_line_chart(self):
        """Report should contain the line chart canvas."""
        html = generate_report(
            prompt="Test",
            final_script="Script",
            iteration_history=_make_history(1),
            final_score=7,
            timestamp="2026-01-01 12:00:00",
            save=False,
        )
        assert "lineChart" in html

    def test_script_content_is_escaped(self):
        """HTML special chars in the script should be escaped."""
        html = generate_report(
            prompt="Test",
            final_script="<script>alert('xss')</script>",
            iteration_history=_make_history(1),
            final_score=7,
            timestamp="2026-01-01 12:00:00",
            save=False,
        )
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_empty_iteration_history(self):
        """Should handle empty iteration history gracefully."""
        html = generate_report(
            prompt="Test",
            final_script="Script",
            iteration_history=[],
            final_score=0,
            timestamp="2026-01-01 12:00:00",
            save=False,
        )
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_html_contains_defense_notes(self):
        """Defense notes should appear when present."""
        html = generate_report(
            prompt="Test",
            final_script="Script",
            iteration_history=_make_history(1),
            final_score=7,
            timestamp="2026-01-01 12:00:00",
            save=False,
        )
        assert "Defense Notes" in html or "defense" in html.lower()

    def test_html_contains_dimension_labels(self):
        """All 5 dimension labels should appear somewhere in the report."""
        html = generate_report(
            prompt="Test",
            final_script="Script",
            iteration_history=_make_history(1),
            final_score=7,
            timestamp="2026-01-01 12:00:00",
            save=False,
        )
        for label in DIMENSION_LABELS:
            assert label in html, f"Dimension label '{label}' missing from report"

    def test_saves_file_when_save_true(self, tmp_path):
        """When save=True, should write an HTML file and return its path."""
        result = generate_report(
            prompt="Test",
            final_script="Script",
            iteration_history=_make_history(1),
            final_score=7,
            timestamp="2026-01-01 12:00:00",
            output_dir=str(tmp_path),
            save=True,
        )
        assert result.endswith(".html")
        from pathlib import Path
        assert Path(result).exists()
        content = Path(result).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content


# ── Module-level constants ────────────────────────────────────────────────────

class TestReportConstants:
    """Verify report module constants are consistent."""

    def test_dimension_labels_and_keys_same_length(self):
        assert len(DIMENSION_LABELS) == len(DIMENSION_KEYS)

    def test_dimension_keys_match_score_dict(self):
        """Keys should match what the critic produces."""
        expected_keys = {"dialogue", "pacing", "character_consistency", "theme", "tension"}
        assert set(DIMENSION_KEYS) == expected_keys
