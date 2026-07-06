import re
import logging
from langchain_core.prompts import ChatPromptTemplate
from rag.retriever import get_similar_dialogues
from utils.llm import get_llm
from utils.retry import invoke_with_retry

logger = logging.getLogger(__name__)


def _strip_markdown_noise(text: str) -> str:
    """Remove common LLM preamble/markdown formatting noise from raw output.

    Strips fenced code blocks, heading markers (#), bold markers (**),
    and common preamble phrases so only the clean beat-sheet content
    reaches downstream agents.
    """
    # Remove fenced code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove heading markers (## Heading → Heading)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    # Remove common LLM preamble lines
    preamble = re.compile(
        r"^(sure[,!]?|here('s| is)|certainly|of course|absolutely)[^\n]*\n",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    text = preamble.sub("", text)
    return text.strip()


def outliner_node(state: dict) -> dict:
    prompt_text = state.get("prompt", "")
    prev_outline = state.get("outline", "")
    critique = state.get("critique", "")
    reoutline_count = state.get("reoutline_count", 0)

    # Fetch RAG context for craft guidance on structuring
    docs = get_similar_dialogues(prompt_text, k=3)
    rag_context = "\n\n---\n\n".join(docs)

    llm = get_llm(temperature=0.6, model="gemini-2.5-pro")

    is_restructure = bool(prev_outline and critique)

    if not is_restructure:
        # -- First-pass outline ------------------------------------------------
        logger.info("Structuring the beat sheet...")
        system_msg = (
            "You are a master script outliner. Given a scene prompt, write a detailed "
            "beat sheet. Break the scene down into a clear beginning, middle (escalation), "
            "and ending (climax/hook). Outline the emotional shifts and power dynamics "
            "between the characters. Do not write the actual dialogue, only the structural beats."
        )
        user_msg = (
            "CRAFT REFERENCE EXAMPLES (for tone/pacing reference):\n{rag_context}\n\n"
            "SCENE PROMPT:\n{prompt}\n\n"
            "Provide the detailed beat sheet:"
        )
        chain_input = {"rag_context": rag_context, "prompt": prompt_text}
    else:
        # -- Structural rebuild triggered by the critic ------------------------
        logger.info("Rebuilding the beat sheet to fix structural problems (re-outline #%d)...",
                    reoutline_count + 1)
        system_msg = (
            "You are a master script outliner performing a STRUCTURAL rebuild. The critic "
            "judged the previous beat sheet structurally flawed — the escalation, climax, "
            "beat order, or ending hook does not work, so line-level rewriting cannot save "
            "the scene. Produce a NEW beat sheet that fixes these structural problems while "
            "still honoring the original prompt. Re-think the escalation and the climax; do "
            "not merely restate the old structure. Outline emotional shifts and power "
            "dynamics. Do not write dialogue — only the structural beats."
        )
        user_msg = (
            "CRAFT REFERENCE EXAMPLES (for tone/pacing reference):\n{rag_context}\n\n"
            "SCENE PROMPT:\n{prompt}\n\n"
            "PREVIOUS BEAT SHEET (structurally flawed — rebuild it):\n{prev_outline}\n\n"
            "CRITIC FEEDBACK THAT TRIGGERED THE REBUILD:\n{critique}\n\n"
            "Provide the improved beat sheet:"
        )
        chain_input = {
            "rag_context": rag_context,
            "prompt": prompt_text,
            "prev_outline": prev_outline,
            "critique": critique,
        }

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", user_msg),
    ])

    chain = prompt_template | llm

    result = invoke_with_retry(chain, chain_input, caller="Outliner")

    clean_outline = _strip_markdown_noise(result.content)

    updates = {"outline": clean_outline}
    if is_restructure:
        # Consume the routing flag and count this rebuild so the loop stays bounded.
        updates["reoutline_count"] = reoutline_count + 1
        updates["needs_restructure"] = False
    return updates
