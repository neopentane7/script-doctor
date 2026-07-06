"""Tests for the regression harness's pure logic (no API calls)."""

import textwrap
import pytest

from eval.harness import evaluate_case, load_cases, DEFAULT_CASES


# ── evaluate_case() ───────────────────────────────────────────────────────────

class TestEvaluateCase:
    """The scoring/bounds logic must correctly flag drift."""

    def test_passes_when_within_bounds(self):
        expect = {"min_final_score": 7, "min_iterations": 1, "max_iterations": 5}
        result = {"final_score": 8, "iterations": 3}
        assert evaluate_case(expect, result, threshold=9) == []

    def test_fails_below_min_final_score(self):
        expect = {"min_final_score": 8}
        result = {"final_score": 6, "iterations": 2}
        failures = evaluate_case(expect, result, threshold=9)
        assert len(failures) == 1 and "final_score 6" in failures[0]

    def test_fails_above_max_final_score(self):
        expect = {"max_final_score": 8}
        result = {"final_score": 10, "iterations": 2}
        assert any("> max 8" in f for f in evaluate_case(expect, result))

    def test_fails_above_max_iterations(self):
        expect = {"max_iterations": 3}
        result = {"final_score": 9, "iterations": 5}
        assert any("iterations 5 > max 3" in f for f in evaluate_case(expect, result))

    def test_fails_below_min_iterations(self):
        expect = {"min_iterations": 2}
        result = {"final_score": 9, "iterations": 1}
        assert any("iterations 1 < min 2" in f for f in evaluate_case(expect, result))

    def test_require_converged_fails_when_below_threshold(self):
        expect = {"require_converged": True}
        result = {"final_score": 7, "iterations": 5}
        assert any("did not converge" in f for f in evaluate_case(expect, result, threshold=9))

    def test_require_converged_passes_at_threshold(self):
        expect = {"require_converged": True}
        result = {"final_score": 9, "iterations": 4}
        assert evaluate_case(expect, result, threshold=9) == []

    def test_missing_final_score_is_a_failure(self):
        assert evaluate_case({"min_final_score": 7}, {"iterations": 2}) == ["result has no final_score"]

    def test_run_error_short_circuits(self):
        failures = evaluate_case({"min_final_score": 7}, {"error": "boom"})
        assert failures == ["run error: boom"]

    def test_empty_expectations_always_passes(self):
        assert evaluate_case({}, {"final_score": 3, "iterations": 5}) == []

    def test_multiple_failures_reported_together(self):
        expect = {"min_final_score": 9, "max_iterations": 2}
        result = {"final_score": 5, "iterations": 5}
        assert len(evaluate_case(expect, result)) == 2


# ── load_cases() ──────────────────────────────────────────────────────────────

class TestLoadCases:
    """Case-file parsing and validation."""

    def test_loads_bundled_cases(self):
        """The shipped cases.yaml must parse and have the required shape."""
        cases = load_cases(DEFAULT_CASES)
        assert len(cases) >= 1
        for c in cases:
            assert "id" in c and "prompt" in c

    def test_loads_valid_temp_file(self, tmp_path):
        p = tmp_path / "cases.yaml"
        p.write_text(textwrap.dedent("""
            - id: t1
              prompt: A scene.
              expect:
                min_final_score: 7
        """), encoding="utf-8")
        cases = load_cases(p)
        assert cases[0]["id"] == "t1"
        assert cases[0]["expect"]["min_final_score"] == 7

    def test_rejects_non_list(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("id: not-a-list\nprompt: x\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a YAML list"):
            load_cases(p)

    def test_rejects_case_missing_required_keys(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- prompt: missing an id\n", encoding="utf-8")
        with pytest.raises(ValueError, match="'id' and 'prompt'"):
            load_cases(p)
