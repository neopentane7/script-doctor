"""Tests for the Critic's Pydantic evaluation schema."""

import pytest
from pydantic import ValidationError
from agents.critic import ScriptEvaluation


class TestScriptEvaluationValid:
    """Valid ScriptEvaluation instances should be accepted."""

    def test_minimal_valid_evaluation(self):
        """All scores at minimum valid values."""
        ev = ScriptEvaluation(
            dialogue_score=0,
            pacing_score=0,
            character_consistency_score=0,
            theme_score=0,
            tension_score=0,
            overall_score=0,
            critique="Needs significant work.",
            improvements=["Fix dialogue.", "Improve pacing.", "Add tension."],
        )
        assert ev.overall_score == 0
        assert len(ev.improvements) == 3

    def test_maximum_valid_evaluation(self):
        """All scores at maximum valid values."""
        ev = ScriptEvaluation(
            dialogue_score=10,
            pacing_score=10,
            character_consistency_score=10,
            theme_score=10,
            tension_score=10,
            overall_score=10,
            critique="Exceptional work across all dimensions.",
            improvements=["Minor polish on page 2.", "Consider tightening act break.", "Optional: add visual motif."],
        )
        assert ev.overall_score == 10

    def test_mid_range_scores(self):
        """Typical mid-range evaluation."""
        ev = ScriptEvaluation(
            dialogue_score=6,
            pacing_score=7,
            character_consistency_score=5,
            theme_score=8,
            tension_score=6,
            overall_score=6,
            critique="Competent but needs more subtext in dialogue.",
            improvements=["Rework Sarah's monologue.", "Add pause before climax.", "Sharpen antagonist motivation."],
        )
        assert ev.dialogue_score == 6
        assert ev.theme_score == 8


class TestScriptEvaluationInvalid:
    """Invalid inputs should raise ValidationError."""

    def test_score_above_10_rejected(self):
        """Scores > 10 should fail Pydantic validation."""
        with pytest.raises(ValidationError):
            ScriptEvaluation(
                dialogue_score=11,
                pacing_score=5,
                character_consistency_score=5,
                theme_score=5,
                tension_score=5,
                overall_score=5,
                critique="Test.",
                improvements=["Fix it."],
            )

    def test_negative_score_rejected(self):
        """Negative scores should fail Pydantic validation."""
        with pytest.raises(ValidationError):
            ScriptEvaluation(
                dialogue_score=-1,
                pacing_score=5,
                character_consistency_score=5,
                theme_score=5,
                tension_score=5,
                overall_score=5,
                critique="Test.",
                improvements=["Fix it."],
            )

    def test_missing_required_field_rejected(self):
        """Omitting a required field should fail."""
        with pytest.raises(ValidationError):
            ScriptEvaluation(
                dialogue_score=5,
                # pacing_score missing
                character_consistency_score=5,
                theme_score=5,
                tension_score=5,
                overall_score=5,
                critique="Test.",
                improvements=["Fix it."],
            )

    def test_non_integer_score_rejected(self):
        """String scores should fail."""
        with pytest.raises(ValidationError):
            ScriptEvaluation(
                dialogue_score="high",
                pacing_score=5,
                character_consistency_score=5,
                theme_score=5,
                tension_score=5,
                overall_score=5,
                critique="Test.",
                improvements=["Fix it."],
            )

    def test_missing_critique_rejected(self):
        """Critique field is required."""
        with pytest.raises(ValidationError):
            ScriptEvaluation(
                dialogue_score=5,
                pacing_score=5,
                character_consistency_score=5,
                theme_score=5,
                tension_score=5,
                overall_score=5,
                # critique missing
                improvements=["Fix it."],
            )

    def test_missing_improvements_rejected(self):
        """Improvements field is required."""
        with pytest.raises(ValidationError):
            ScriptEvaluation(
                dialogue_score=5,
                pacing_score=5,
                character_consistency_score=5,
                theme_score=5,
                tension_score=5,
                overall_score=5,
                critique="Test.",
                # improvements missing
            )


class TestScriptEvaluationEdgeCases:
    """Boundary and edge case tests."""

    def test_boundary_score_zero(self):
        """Score of exactly 0 is valid."""
        ev = ScriptEvaluation(
            dialogue_score=0, pacing_score=0,
            character_consistency_score=0, theme_score=0,
            tension_score=0, overall_score=0,
            critique="Very poor.", improvements=["Rewrite everything."],
        )
        assert ev.overall_score == 0

    def test_boundary_score_ten(self):
        """Score of exactly 10 is valid."""
        ev = ScriptEvaluation(
            dialogue_score=10, pacing_score=10,
            character_consistency_score=10, theme_score=10,
            tension_score=10, overall_score=10,
            critique="Flawless.", improvements=["None needed."],
        )
        assert ev.overall_score == 10

    def test_empty_string_critique_accepted(self):
        """Empty string critique is technically valid (Pydantic str field)."""
        ev = ScriptEvaluation(
            dialogue_score=5, pacing_score=5,
            character_consistency_score=5, theme_score=5,
            tension_score=5, overall_score=5,
            critique="",
            improvements=["Fix it."],
        )
        assert ev.critique == ""

    def test_empty_improvements_list_accepted(self):
        """Empty improvements list is technically valid (List[str])."""
        ev = ScriptEvaluation(
            dialogue_score=5, pacing_score=5,
            character_consistency_score=5, theme_score=5,
            tension_score=5, overall_score=5,
            critique="Needs work.",
            improvements=[],
        )
        assert len(ev.improvements) == 0

    def test_large_improvements_list_accepted(self):
        """Many improvements should be accepted."""
        ev = ScriptEvaluation(
            dialogue_score=5, pacing_score=5,
            character_consistency_score=5, theme_score=5,
            tension_score=5, overall_score=5,
            critique="Multiple issues.",
            improvements=[f"Fix issue {i}" for i in range(10)],
        )
        assert len(ev.improvements) == 10


class TestOverallScoreClamping:
    """Verify the model_validator clamps overall_score to the dimension average."""

    def test_inflated_overall_score_is_clamped(self):
        """overall_score much higher than average should be clamped to avg."""
        # avg = (4+4+4+4+4)/5 = 4, overall=10 → more than ±1 off → should clamp to 4
        ev = ScriptEvaluation(
            dialogue_score=4,
            pacing_score=4,
            character_consistency_score=4,
            theme_score=4,
            tension_score=4,
            overall_score=10,  # wildly inflated
            critique="Test.",
            improvements=["Fix it."],
        )
        assert ev.overall_score == 4

    def test_deflated_overall_score_is_clamped(self):
        """overall_score much lower than average should be clamped to avg."""
        # avg = (9+9+9+9+9)/5 = 9, overall=2 → more than ±1 off → should clamp to 9
        ev = ScriptEvaluation(
            dialogue_score=9,
            pacing_score=9,
            character_consistency_score=9,
            theme_score=9,
            tension_score=9,
            overall_score=2,  # wildly deflated
            critique="Test.",
            improvements=["Tweak it."],
        )
        assert ev.overall_score == 9

    def test_within_tolerance_is_preserved(self):
        """overall_score within ±1 of average should not be clamped."""
        # avg = (7+7+7+7+7)/5 = 7.0, overall=8 → within ±1 → preserved
        ev = ScriptEvaluation(
            dialogue_score=7,
            pacing_score=7,
            character_consistency_score=7,
            theme_score=7,
            tension_score=7,
            overall_score=8,  # one above avg — holistic leniency allowed
            critique="Good work.",
            improvements=["Minor polish."],
        )
        assert ev.overall_score == 8

    def test_exact_average_is_unchanged(self):
        """overall_score matching the exact average should be preserved."""
        # avg = (6+7+5+8+4)/5 = 6.0, overall=6
        ev = ScriptEvaluation(
            dialogue_score=6,
            pacing_score=7,
            character_consistency_score=5,
            theme_score=8,
            tension_score=4,
            overall_score=6,
            critique="Solid draft.",
            improvements=["Tighten act 2.", "Rework ending.", "Add subtext."],
        )
        assert ev.overall_score == 6
