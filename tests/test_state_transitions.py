"""Tests for LangGraph state transitions and graph structure."""

import pytest
from main import should_continue, build_graph, SCORE_THRESHOLD, MAX_ITERATIONS


# ── should_continue() ────────────────────────────────────────────────────────

class TestShouldContinue:
    """Validate the conditional branching logic of the Writer ↔ Critic loop."""

    def test_low_score_first_iteration_continues(self):
        """Score below threshold on first iteration → keep revising."""
        state = {"score": 3, "iteration_count": 1}
        assert should_continue(state) == "revise"

    def test_mid_score_continues(self):
        """Score just below threshold → keep revising."""
        state = {"score": SCORE_THRESHOLD - 1, "iteration_count": 2}
        assert should_continue(state) == "revise"

    def test_threshold_score_ends(self):
        """Score exactly at threshold → accept and end."""
        state = {"score": SCORE_THRESHOLD, "iteration_count": 2}
        assert should_continue(state) == "end"

    def test_high_score_ends(self):
        """Score above threshold → accept and end."""
        state = {"score": 10, "iteration_count": 1}
        assert should_continue(state) == "end"

    def test_max_iterations_reached_ends(self):
        """Hit iteration cap even with low score → end."""
        state = {"score": 3, "iteration_count": MAX_ITERATIONS}
        assert should_continue(state) == "end"

    def test_max_iterations_exceeded_ends(self):
        """Past iteration cap → end."""
        state = {"score": 2, "iteration_count": MAX_ITERATIONS + 1}
        assert should_continue(state) == "end"

    def test_both_conditions_met_ends(self):
        """Both score threshold and iteration cap met → end."""
        state = {"score": SCORE_THRESHOLD, "iteration_count": MAX_ITERATIONS}
        assert should_continue(state) == "end"

    def test_missing_score_defaults_to_revise(self):
        """Missing score key → defaults to 0, should continue."""
        state = {"iteration_count": 1}
        assert should_continue(state) == "revise"

    def test_missing_iteration_count_defaults_to_revise(self):
        """Missing iteration_count key → defaults to 0, should continue."""
        state = {"score": 5}
        assert should_continue(state) == "revise"

    def test_empty_state_continues(self):
        """Completely empty state → defaults to 0/0, should continue."""
        assert should_continue({}) == "revise"

    def test_zero_score_zero_iterations_continues(self):
        """Initial state values → should continue."""
        state = {"score": 0, "iteration_count": 0}
        assert should_continue(state) == "revise"

    def test_scores_progression_stagnation_exits_early(self):
        """Scores plateau or decrease without setting a new high score for 2 steps → end."""
        # Baseline = 6, latest two are [6, 6] (max is 6 <= 6) -> end
        state1 = {"score": 6, "iteration_count": 3, "scores_progression": [6, 6, 6]}
        assert should_continue(state1) == "end"

        # Baseline = 7, latest two are [6, 7] (max is 7 <= 7) -> end
        state2 = {"score": 7, "iteration_count": 3, "scores_progression": [7, 6, 7]}
        assert should_continue(state2) == "end"

        # Baseline = 7, latest two are [6, 6] (max is 6 <= 7) -> end
        state3 = {"score": 6, "iteration_count": 3, "scores_progression": [7, 6, 6]}
        assert should_continue(state3) == "end"

    def test_scores_progression_improving_continues(self):
        """Scores are consistently improving or hitting a new high → revise."""
        # Baseline = 5, latest two are [6, 7] (max is 7 > 5) -> revise
        state1 = {"score": 7, "iteration_count": 3, "scores_progression": [5, 6, 7]}
        assert should_continue(state1) == "revise"

        # Baseline = 6, latest two are [7, 7] (max is 7 > 6) -> revise
        state2 = {"score": 7, "iteration_count": 3, "scores_progression": [5, 6, 7, 7]}
        assert should_continue(state2) == "revise"

    def test_scores_progression_too_short_continues(self):
        """Not enough history (< 3 items) → revise."""
        state = {"score": 6, "iteration_count": 2, "scores_progression": [6, 6]}
        assert should_continue(state) == "revise"


# ── Constants ─────────────────────────────────────────────────────────────────

class TestConstants:
    """Verify that the pipeline constants are sensible."""

    def test_score_threshold_is_reasonable(self):
        assert 1 <= SCORE_THRESHOLD <= 10

    def test_max_iterations_is_reasonable(self):
        assert 1 <= MAX_ITERATIONS <= 20

    def test_threshold_consistent_with_rubric(self):
        """Threshold should be 9 — the rubric reserves 9 for 'Exceptional' work,
        which (with the strict critic) forces the refinement loop to engage."""
        assert SCORE_THRESHOLD == 9


# ── Graph structure ───────────────────────────────────────────────────────────

class TestGraphBuild:
    """Verify the LangGraph compiles and has the expected topology."""

    def test_graph_compiles_without_error(self):
        """build_graph() should return a compiled graph object."""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        """Graph should contain outliner, writer, and critic nodes."""
        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        # LangGraph adds __start__ and __end__ internal nodes
        assert "outliner" in node_names
        assert "writer" in node_names
        assert "critic" in node_names
