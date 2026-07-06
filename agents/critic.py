import os
import logging
from pydantic import BaseModel, Field, model_validator
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from utils.llm import get_llm, get_groq_llm, DEFAULT_GROQ_MODEL
from utils.retry import invoke_with_retry

logger = logging.getLogger(__name__)

# Bound how much prior critique the critic re-reads each cycle. Mirrors the
# writer's memory buffer so the critic prompt does not grow unboundedly across
# iterations (which previously inflated latency and cost late in the loop).
CRITIC_HISTORY_WINDOW = 2

# The critic defaults to a DIFFERENT model family than the Gemini writer (Groq's
# free-tier Llama 3.3 70B). A heterogeneous judge reduces the self-preference
# bias you get when one model grades its own output, and — being stricter than
# a lenient Flash critic — it actually drives the revision loop.
#
# All of this is resolved from the environment at call time (after .env loads):
#   CRITIC_PROVIDER = "groq" (default) | "google"
#   CRITIC_MODEL    = provider-specific override (optional)
_DEFAULT_CRITIC_PROVIDER = "groq"
_DEFAULT_GOOGLE_CRITIC_MODEL = "gemini-2.5-flash"


def _get_critic_llm(temperature: float = 0.15):
    """Build the critic LLM from env config, returning (llm, label)."""
    provider = os.getenv("CRITIC_PROVIDER", _DEFAULT_CRITIC_PROVIDER).lower()
    if provider == "groq":
        model = os.getenv("CRITIC_MODEL", DEFAULT_GROQ_MODEL)
        return get_groq_llm(temperature=temperature, model=model), f"groq:{model}"
    # Fallback: keep the whole pipeline on Google if explicitly requested.
    model = os.getenv("CRITIC_MODEL", _DEFAULT_GOOGLE_CRITIC_MODEL)
    return get_llm(temperature=temperature, model=model), f"google:{model}"


class ScriptEvaluation(BaseModel):
    """Structured rubric for evaluating a screenplay scene."""

    dialogue_score: int = Field(
        ge=0, le=10,
        description=(
            "Score 0-10 for dialogue quality: naturalness, subtext, character voice "
            "distinctness, and avoidance of on-the-nose exposition."
        ),
    )
    pacing_score: int = Field(
        ge=0, le=10,
        description=(
            "Score 0-10 for narrative rhythm: scene beats land at the right moment, "
            "action lines are lean, no unnecessary padding."
        ),
    )
    character_consistency_score: int = Field(
        ge=0, le=10,
        description=(
            "Score 0-10 for how faithfully characters behave relative to the archetypes "
            "and relationships established in the original prompt. Deduct points for "
            "behaviour that contradicts what the prompt implies about each character."
        ),
    )
    theme_score: int = Field(
        ge=0, le=10,
        description=(
            "Score 0-10 for thematic resonance: does the scene faithfully serve the core "
            "themes, emotional register, and dramatic stakes implied by the original prompt, "
            "regardless of genre?"
        ),
    )
    tension_score: int = Field(
        ge=0, le=10,
        description=(
            "Score 0-10 for dramatic tension: are the stakes clear and visceral? Does "
            "every beat ratchet up the pressure? Does the scene end on a meaningful hook?"
        ),
    )
    overall_score: int = Field(
        ge=0, le=10,
        description=(
            "Holistic score 0-10. Must reflect the AVERAGE of the five dimension scores "
            "rounded to the nearest integer. Only award 8+ to genuinely exceptional work."
        ),
    )
    critique: str = Field(
        description=(
            "Detailed, specific critique (3-5 sentences) explaining what works and what "
            "needs improvement. Reference specific lines or moments from the draft."
        ),
    )
    improvements: List[str] = Field(
        description=(
            "Exactly 3-5 concrete, actionable revision instructions. Be precise -- "
            "name the character, line, or beat that needs changing."
        ),
    )
    structural_revision_needed: bool = Field(
        default=False,
        description=(
            "Set True ONLY when the scene's core problems are structural — the beat "
            "sheet itself is flawed (wrong escalation, missing climax, mis-ordered "
            "beats, no dramatic hook) so that no amount of line-level rewriting can "
            "fix it. Set False when the structure is sound and only the prose, "
            "dialogue, or pacing within the existing beats needs polishing."
        ),
    )

    @model_validator(mode="after")
    def clamp_overall_to_average(self) -> "ScriptEvaluation":
        """Sanity-clamp overall_score against the dimension average.

        The overall score is deliberately allowed to diverge from the raw mean so
        the critic can exercise genuine holistic judgment (a scene can be more —
        or less — than the sum of its rubric parts). We only clamp gross drift of
        more than ±2 points, which almost always signals a hallucinated or
        mis-typed score rather than a considered verdict.
        """
        computed_avg = (
            self.dialogue_score
            + self.pacing_score
            + self.character_consistency_score
            + self.theme_score
            + self.tension_score
        ) / 5.0
        rounded_avg = round(computed_avg)
        # Allow ±2 holistic leniency; clamp anything more extreme.
        if abs(self.overall_score - rounded_avg) > 2:
            self.overall_score = max(0, min(10, rounded_avg))
        return self


def critic_node(state: dict) -> dict:
    prompt_text = state.get("prompt", "")
    draft = state.get("draft", "")
    defense_notes = state.get("defense_notes", "None.")
    iteration_count = state.get("iteration_count", 0)
    history = state.get("history", [])

    logger.info("Evaluating draft (iteration %d)...", iteration_count + 1)
    critic_llm, critic_label = _get_critic_llm(temperature=0.15)
    logger.info("Critic model: %s", critic_label)
    structured_llm = critic_llm.with_structured_output(ScriptEvaluation)

    system_msg = (
        "You are a senior script doctor and industry critic with 20 years of experience "
        "across all genres. Evaluate the script draft rigorously across five dimensions: "
        "dialogue, pacing, character consistency (relative to the original prompt's archetypes), "
        "thematic resonance (relative to whatever themes and emotional register the prompt implies), "
        "and dramatic tension.\n\n"
        "Scoring principles (calibrate STRICTLY — most drafts belong in the 5-7 band):\n"
        "- 1-3: Broken — craft or structural failures a reader notices immediately\n"
        "- 4-5: Amateur — grammatical prose but flat characters, on-the-nose dialogue, or slack pacing\n"
        "- 6: Functional first-draft work — the scene works but has clear, nameable weaknesses\n"
        "- 7: Good — professional polish; only minor issues remain\n"
        "- 8: Excellent — you would be proud to submit this; nothing distracts\n"
        "- 9: Exceptional — distinctive voice, not a wasted line, lingers after reading\n"
        "- 10: Flawless — the best possible version of this scene\n\n"
        "Calibration rules (enforce these — do NOT be agreeable):\n"
        "- Default to skepticism. A FIRST draft almost never exceeds 6-7; treat 8+ as something\n"
        "  the writer must EARN through revision, not a reward for competence.\n"
        "- Per dimension: if you can name a concrete, specific weakness in that dimension, it has\n"
        "  NOT yet reached 8 — score it 7 or below until that weakness is actually fixed.\n"
        "- You must list 3-5 required improvements. A draft with real improvements still pending\n"
        "  is by definition not yet exceptional — reserve an overall 9-10 for a draft you would\n"
        "  change essentially nothing about.\n"
        "- Never inflate a score to be encouraging. Withholding approval is how you force the\n"
        "  revision that makes the scene great. As dimensions genuinely improve, raise their scores.\n\n"
        "Crucially, READ THE WRITER'S DEFENSE NOTES. If the writer makes a compelling creative "
        "argument for ignoring a past note or taking a specific direction, consider it valid and grade accordingly.\n\n"
        "Finally, diagnose the LEVEL of the problem. If the weaknesses are structural — the "
        "underlying beat sheet escalates poorly, lacks a real climax, orders its beats wrong, or "
        "ends without a hook — set structural_revision_needed=True so the outline itself is rebuilt. "
        "If the bones are sound and only the dialogue/prose/pacing within the beats needs work, "
        "set it False. Do not flag structural revision for merely cosmetic issues."
    )
    user_msg = (
        "ORIGINAL PROMPT:\n{prompt}\n\n"
        "WRITER'S DEFENSE NOTES:\n{defense_notes}\n\n"
        "REVISION HISTORY (past critiques — detect any regressions):\n{history_summary}\n\n"
        "SCRIPT DRAFT TO EVALUATE:\n{draft}\n\n"
        "Provide your structured evaluation:"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", user_msg),
    ])

    chain = prompt_template | structured_llm

    # Build a compact history summary so the critic can detect regression.
    # Only the most recent CRITIC_HISTORY_WINDOW critiques are included so the
    # prompt stays bounded as the loop runs (older critiques are already baked
    # into the current draft).
    history_summary = "None — this is the first evaluation."
    if history:
        recent = history[-CRITIC_HISTORY_WINDOW:]
        start_index = len(history) - len(recent)
        lines = []
        for offset, h in enumerate(recent):
            iteration_number = start_index + offset + 1
            lines.append(f"--- Iteration {iteration_number} critique ---\n{h['critique']}")
        history_summary = "\n\n".join(lines)

    result: ScriptEvaluation = invoke_with_retry(
        chain,
        {"prompt": prompt_text, "defense_notes": defense_notes, "draft": draft,
         "history_summary": history_summary},
        caller="Critic",
    )

    # Guard: clamp scores in case the model drifts despite Pydantic validators
    scores = {
        "Dialogue":              result.dialogue_score,
        "Pacing":                result.pacing_score,
        "Character Consistency": result.character_consistency_score,
        "Thematic Resonance":    result.theme_score,
        "Dramatic Tension":      result.tension_score,
    }

    formatted_critique = (
        f"=== ITERATION {iteration_count + 1} EVALUATION ===\n\n"
        f"DIMENSION SCORES:\n"
        + "".join(f"  • {k}: {v}/10\n" for k, v in scores.items())
        + f"  • Overall:             {result.overall_score}/10\n\n"
        f"CRITIQUE:\n{result.critique}\n\n"
        f"REQUIRED IMPROVEMENTS:\n"
        + "".join(f"  {i+1}. {imp}\n" for i, imp in enumerate(result.improvements))
    )

    logger.info("\n%s", formatted_critique)
    
    # Update history for the next iteration (so the writer remembers this critique)
    updated_history = list(history)
    updated_history.append({
        "draft": draft,
        "critique": formatted_critique
    })

    # Track overall scores progression to detect stagnation and exit early
    scores_progression = list(state.get("scores_progression", []))
    scores_progression.append(result.overall_score)

    return {
        "critique": formatted_critique,
        "score": result.overall_score,
        "iteration_count": iteration_count + 1,
        "history": updated_history,
        "scores_progression": scores_progression,
        # Routing signal: True → structural rebuild via the outliner, False → line-level revision
        "needs_restructure": result.structural_revision_needed,
        # Structured data consumed by the report generator
        "last_scores": {
            "dialogue":             result.dialogue_score,
            "pacing":               result.pacing_score,
            "character_consistency":result.character_consistency_score,
            "theme":                result.theme_score,
            "tension":              result.tension_score,
            "overall":              result.overall_score,
        },
        "last_critique_text":  result.critique,
        "last_improvements":   result.improvements,
    }
