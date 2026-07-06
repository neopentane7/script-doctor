"""Offline tests for the agent node functions.

The LLM and RAG calls are mocked, so these exercise each node's state handling,
prompt-path selection, and output shaping without touching any API. End-to-end
behavior with real models is covered separately by the (API-costing) regression
harness in eval/.
"""

from unittest.mock import patch, MagicMock

from agents.outliner import outliner_node
from agents.writer import writer_node, WriterOutput
from agents.critic import critic_node, ScriptEvaluation


def _make_eval(overall=7, structural=False):
    return ScriptEvaluation(
        dialogue_score=7, pacing_score=7, character_consistency_score=7,
        theme_score=7, tension_score=7, overall_score=overall,
        critique="A specific critique.",
        improvements=["Fix A.", "Fix B.", "Fix C."],
        structural_revision_needed=structural,
    )


# ── Outliner ──────────────────────────────────────────────────────────────────

class TestOutlinerNode:

    @patch("agents.outliner.invoke_with_retry")
    @patch("agents.outliner.get_llm")
    @patch("agents.outliner.get_similar_dialogues", return_value=["ref"])
    def test_first_pass_cleans_markdown(self, mock_rag, mock_llm, mock_invoke):
        mock_invoke.return_value = MagicMock(content="## Beat Sheet\n**Bold** opening beat.")
        out = outliner_node({"prompt": "A tense scene."})
        assert "Beat Sheet" in out["outline"]
        assert "**" not in out["outline"] and "##" not in out["outline"]  # stripped
        # First pass must NOT touch the restructure counters
        assert "reoutline_count" not in out
        assert "needs_restructure" not in out

    @patch("agents.outliner.invoke_with_retry")
    @patch("agents.outliner.get_llm")
    @patch("agents.outliner.get_similar_dialogues", return_value=["ref"])
    def test_restructure_increments_and_clears_flag(self, mock_rag, mock_llm, mock_invoke):
        mock_invoke.return_value = MagicMock(content="Rebuilt beats.")
        state = {
            "prompt": "A tense scene.",
            "outline": "old, broken outline",
            "critique": "The escalation is structurally flat.",
            "reoutline_count": 0,
        }
        out = outliner_node(state)
        assert out["outline"] == "Rebuilt beats."
        assert out["reoutline_count"] == 1
        assert out["needs_restructure"] is False


# ── Writer ────────────────────────────────────────────────────────────────────

class TestWriterNode:

    @patch("agents.writer.invoke_with_retry")
    @patch("agents.writer.get_llm")
    @patch("agents.writer.get_similar_dialogues", return_value=["ref"])
    def test_first_draft_path(self, mock_rag, mock_llm, mock_invoke):
        mock_invoke.return_value = WriterOutput(defense_notes="notes", draft="INT. ROOM - DAY")
        out = writer_node({"prompt": "p", "outline": "o"})  # no draft => first draft
        assert out["draft"] == "INT. ROOM - DAY"
        assert out["defense_notes"] == "notes"

    @patch("agents.writer.invoke_with_retry")
    @patch("agents.writer.get_llm")
    @patch("agents.writer.get_similar_dialogues", return_value=["ref"])
    def test_revision_path_uses_history(self, mock_rag, mock_llm, mock_invoke):
        mock_invoke.return_value = WriterOutput(defense_notes="d", draft="revised draft")
        state = {
            "prompt": "p", "outline": "o", "draft": "old draft", "critique": "fix it",
            "history": [{"critique": "prev critique", "draft": "old"}],
            "last_critique_text": "prose is flat", "last_improvements": ["sharpen dialogue"],
        }
        out = writer_node(state)
        assert out["draft"] == "revised draft"
        assert out["defense_notes"] == "d"
        # RAG query for a revision is grounded on the scene (prompt+outline), not the critique
        assert mock_rag.call_args.args[0].startswith("p")


# ── Critic ────────────────────────────────────────────────────────────────────

class TestCriticNode:

    @patch("agents.critic.invoke_with_retry")
    @patch("agents.critic._get_critic_llm")
    def test_returns_scores_and_increments_iteration(self, mock_get_critic, mock_invoke):
        mock_get_critic.return_value = (MagicMock(), "groq:test-model")
        mock_invoke.return_value = _make_eval(overall=7, structural=False)
        out = critic_node({"prompt": "p", "draft": "d", "iteration_count": 0, "history": []})
        assert out["score"] == 7
        assert out["iteration_count"] == 1
        assert out["needs_restructure"] is False
        assert out["last_scores"]["overall"] == 7
        assert out["scores_progression"] == [7]
        assert len(out["history"]) == 1  # this iteration's critique appended

    @patch("agents.critic.invoke_with_retry")
    @patch("agents.critic._get_critic_llm")
    def test_propagates_structural_flag(self, mock_get_critic, mock_invoke):
        mock_get_critic.return_value = (MagicMock(), "groq:test-model")
        mock_invoke.return_value = _make_eval(overall=5, structural=True)
        out = critic_node({"prompt": "p", "draft": "d", "iteration_count": 2,
                           "history": [{"critique": "c1", "draft": "d1"}],
                           "scores_progression": [4]})
        assert out["needs_restructure"] is True
        assert out["iteration_count"] == 3
        assert out["scores_progression"] == [4, 5]
        assert len(out["history"]) == 2  # prior + this one
