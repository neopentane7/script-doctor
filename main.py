import os
import sys
import logging
import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, List

# Ensure UTF-8 output on Windows terminals that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langgraph.graph import StateGraph, START, END

from rag.retriever import create_vectorstore
from agents.outliner import outliner_node
from agents.writer import writer_node
from agents.critic import critic_node
from report.generator import generate_report
from utils.tracing import configure_tracing

# Load environment variables
load_dotenv(override=True)

# ── Logging configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Quiet noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ── Output location ──────────────────────────────────────────────────────────
# All generated run artifacts (.txt transcripts + .html reports) are written
# here instead of the project root, keeping the source tree clean. Gitignored.
RUNS_DIR = str(Path(__file__).parent / "runs")

# ── Quality threshold ────────────────────────────────────────────────────────
# 9/10 is deliberately demanding: paired with the strict critic rubric, a first
# draft won't clear it, so the Writer ↔ Critic loop actually engages and refines.
SCORE_THRESHOLD = 9      # Script is accepted when overall score >= this value
MAX_ITERATIONS  = 5      # Hard cap on Writer ↔ Critic revision cycles
MAX_REOUTLINES  = 2      # Hard cap on structural (beat-sheet) rebuilds per run


# ── State definition ─────────────────────────────────────────────────────────
class ScriptState(TypedDict):
    prompt:             str
    outline:            str
    defense_notes:      str
    draft:              str
    critique:           str
    score:              int
    iteration_count:    int
    history:            List[dict]
    # Per-iteration structured data (updated by critic each cycle)
    last_scores:        dict
    last_critique_text: str
    last_improvements:  List[str]
    scores_progression: List[int]
    # Structural-revision routing (critic → outliner rebuild loop)
    needs_restructure:  bool
    reoutline_count:    int


def should_continue(state: ScriptState) -> str:
    if state.get("score", 0) >= SCORE_THRESHOLD or state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "end"

    # Early exit on score stagnation (no improvement over the last two iterations)
    scores = state.get("scores_progression", [])
    if len(scores) >= 3:
        max_before = max(scores[:-2])
        max_latest = max(scores[-2:])
        if max_latest <= max_before:
            logger.info(
                "Overall score has stagnated over last 2 iterations (progression: %s). "
                "Exiting loop early.",
                scores,
            )
            return "end"

    # When the critic diagnoses a structural (beat-sheet) flaw that line-level
    # rewriting can't fix, route back to the outliner to rebuild the structure —
    # but only up to MAX_REOUTLINES times to keep the run bounded and cheap.
    if state.get("needs_restructure") and state.get("reoutline_count", 0) < MAX_REOUTLINES:
        logger.info(
            "Critic flagged a structural flaw (re-outline %d/%d). Rebuilding the beat sheet.",
            state.get("reoutline_count", 0) + 1, MAX_REOUTLINES,
        )
        return "reoutline"

    return "revise"


# ── Graph construction ────────────────────────────────────────────────────────
def build_graph():
    workflow = StateGraph(ScriptState)
    workflow.add_node("outliner", outliner_node)
    workflow.add_node("writer",   writer_node)
    workflow.add_node("critic",   critic_node)
    
    workflow.add_edge(START,       "outliner")
    workflow.add_edge("outliner",  "writer")
    workflow.add_edge("writer",    "critic")
    workflow.add_conditional_edges("critic", should_continue,
                                   {"revise": "writer",
                                    "reoutline": "outliner",
                                    "end": END})
    return workflow.compile()


# ── Output helpers ────────────────────────────────────────────────────────────
def _build_output_text(final_state: dict, timestamp: str) -> str:
    sep = "=" * 60
    return (
        f"{sep}\n  SCRIPT DOCTOR -- FINAL OUTPUT\n"
        f"  Generated: {timestamp}\n"
        f"  Final Score: {final_state.get('score','?')}/10  |  "
        f"Iterations: {final_state.get('iteration_count','?')}\n{sep}\n\n"
        f"ORIGINAL PROMPT:\n{final_state.get('prompt','')}\n\n{sep}\n\n"
        f"APPROVED OUTLINE:\n\n{final_state.get('outline','')}\n\n{sep}\n\n"
        f"FINAL APPROVED SCRIPT:\n\n{final_state.get('draft','')}\n\n{sep}\n\n"
        f"LAST CRITIQUE:\n\n{final_state.get('critique','')}\n{sep}\n"
    )


def _save_txt(text: str, timestamp: str, output_dir: str = ".") -> str:
    ts = timestamp.replace(":", "-").replace(" ", "_")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"output_{ts}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return filename


# ── Core pipeline runner (used by both CLI and web server) ───────────────────
def run_pipeline(
    prompt: str,
    on_node: callable = None,
    save_files: bool = True,
    output_dir: str = RUNS_DIR,
) -> tuple[dict, list, str]:
    """
    Run the full Script Doctor pipeline for a given prompt.

    Args:
        prompt:      The scene premise to write.
        on_node:     Optional callback(node_name: str) called after each node
                     completes, for streaming progress updates.
        save_files:  If True, saves .txt and .html reports to output_dir.
        output_dir:  Directory to save output files in.

    Returns:
        (final_state, iteration_history, timestamp)
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")

    # The critic defaults to Groq (a free, stricter, heterogeneous judge). If that
    # provider is active, its key must be present too — fail early with guidance.
    if os.getenv("CRITIC_PROVIDER", "groq").lower() == "groq" and not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set, but the critic uses Groq by default. "
            "Get a free key at https://console.groq.com/keys and add it to .env, "
            "or set CRITIC_PROVIDER=google to run the critic on Gemini instead."
        )

    # Opt-in observability: if LangSmith env vars are set, every agent call is
    # auto-traced (tokens, latency, cost) under the project name.
    configure_tracing()

    if on_node:
        on_node("init", "Initializing Vector DB...")
    create_vectorstore()

    graph = build_graph()
    if on_node:
        on_node("ready", "Pipeline ready. Starting agents...")

    initial_state: ScriptState = {
        "prompt":             prompt,
        "outline":            "",
        "defense_notes":      "",
        "draft":              "",
        "critique":           "",
        "score":              0,
        "iteration_count":    0,
        "history":            [],
        "last_scores":        {},
        "last_critique_text": "",
        "last_improvements":  [],
        "scores_progression": [],
        "needs_restructure":  False,
        "reoutline_count":    0,
    }

    final_state = dict(initial_state)
    iteration_history: list[dict] = []

    for event in graph.stream(initial_state):
        for node_name, state_update in event.items():
            logger.info("-" * 60)
            logger.info("Finished node: %s", node_name.upper())
            final_state.update(state_update)

            if on_node:
                on_node(node_name, f"✅ {node_name.capitalize()} completed.")

            if node_name == "critic" and state_update.get("last_scores"):
                # Capture defense_notes from the state *at this moment* — the writer
                # sets defense_notes before the critic runs, so it reflects the notes
                # for the draft that was just evaluated.
                iteration_history.append({
                    "scores":        state_update["last_scores"],
                    "critique_text": state_update.get("last_critique_text", ""),
                    "improvements":  state_update.get("last_improvements", []),
                    "defense_notes": final_state.get("defense_notes", ""),
                })

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if save_files:
        txt_file = _save_txt(_build_output_text(final_state, timestamp), timestamp, output_dir)
        logger.info("Text output saved: %s", txt_file)

    return final_state, iteration_history, timestamp


# ── CLI entry point ───────────────────────────────────────────────────────────
DEFAULT_PROMPT = (
    "A centuries-old vampire and a grizzled monster hunter are forced into a cathedral's "
    "basement to hide from an approaching mob. The holy ground is slowly draining the "
    "vampire's strength, and the hunter must decide whether to protect his ancient enemy "
    "or let the mob do his job for him."
)


def run_agents():
    logger.info("Starting Multi-Agent Script Doctor...")
    logger.info("-" * 60)
    final_state, iteration_history, timestamp = run_pipeline(
        prompt=DEFAULT_PROMPT,
        save_files=True,
    )

    report_file = generate_report(
        prompt=final_state["prompt"],
        final_script=final_state["draft"],
        iteration_history=iteration_history,
        final_score=final_state["score"],
        timestamp=timestamp,
        output_dir=RUNS_DIR,
        score_threshold=SCORE_THRESHOLD,
    )
    logger.info("HTML report saved: %s", report_file)
    logger.info("=" * 60)
    logger.info(
        "FINAL SCORE: %d/10  |  ITERATIONS: %d",
        final_state["score"], final_state["iteration_count"],
    )
    logger.info("=" * 60)
    print(final_state["draft"])


if __name__ == "__main__":
    run_agents()