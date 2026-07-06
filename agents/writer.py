import logging
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from rag.retriever import get_similar_dialogues
from utils.llm import get_llm
from utils.retry import invoke_with_retry

logger = logging.getLogger(__name__)


class WriterOutput(BaseModel):
    defense_notes: str = Field(description="Your explanation or defense of creative choices made in this draft. If you ignored a critic's note, explain why.")
    draft: str = Field(description="The fully formatted screenplay script scene.")


def writer_node(state: dict) -> dict:
    prompt_text = state.get("prompt", "")
    outline = state.get("outline", "")
    draft = state.get("draft", "")
    critique = state.get("critique", "")
    history = state.get("history", [])

    llm = get_llm(temperature=0.85).with_structured_output(WriterOutput)

    # RAG grounding: retrieve craft examples that match the SCENE ITSELF — its
    # premise and structure — rather than the critique. The corpus holds craft
    # exemplars indexed by tone/genre/technique, so a query built from the
    # critique ("dialogue is on-the-nose…") retrieves examples that are *about*
    # the complaint, not ones that *demonstrate the fix*. Grounding on the
    # prompt + outline keeps retrieval tonally relevant and stable across
    # iterations, which is what actually guides the Writer's craft.
    query = f"{prompt_text}\n{outline}".strip() if outline else prompt_text
    docs = get_similar_dialogues(query, k=3)
    rag_context = "\n\n---\n\n".join(docs)
    
    # Format history - trim to the latest 2 iterations to save context tokens
    history_str = ""
    if history:
        trimmed_history = history[-2:]
        start_index = len(history) - len(trimmed_history)
        for idx, h in enumerate(trimmed_history):
            iteration_number = start_index + idx + 1
            history_str += f"\n--- ITERATION {iteration_number} ---\nCRITIQUE RECEIVED:\n{h['critique']}\n"

    if not draft:
        # -- First draft -------------------------------------------------------
        logger.info("Drafting original script from outline...")
        system_msg = (
            "You are an award-winning scriptwriter with mastery across all genres. "
            "Write a compelling, cinematic script scene based on the user's prompt "
            "AND the provided structural outline.\n\n"
            "Guidelines:\n"
            "- Open with a slug line appropriate to the scene (INT./EXT. LOCATION -- TIME)\n"
            "- Use proper screenplay format: action lines, character cues, parentheticals\n"
            "- Dialogue must feel earned and layered -- avoid on-the-nose exposition\n"
            "- Match tone and genre to whatever the prompt implies\n"
            "- End on a beat of genuine dramatic tension\n"
            "- Study the provided CRAFT REFERENCE EXAMPLES for technique (subtext, "
            "pacing, character voice) -- do NOT copy their content or setting"
        )
        user_msg = (
            "CRAFT REFERENCE EXAMPLES (technique only -- not content to copy):\n{rag_context}\n\n"
            "SCENE PROMPT:\n{prompt}\n\n"
            "APPROVED OUTLINE/BEAT SHEET:\n{outline}\n\n"
            "Write the script scene and any defense/creative notes you want the critic to see:"
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("user", user_msg),
        ])
        chain = prompt_template | llm

        result: WriterOutput = invoke_with_retry(
            chain,
            {"rag_context": rag_context, "prompt": prompt_text, "outline": outline},
            caller="Writer",
        )

    else:
        # -- Revision pass -----------------------------------------------------
        logger.info("Revising script based on critique...")
        system_msg = (
            "You are an award-winning scriptwriter revising a draft based on critic feedback.\n\n"
            "Revision rules:\n"
            "- Review the current critique and the history of past critiques to avoid regression.\n"
            "- Address the critic's points. However, if a critic's suggestion ruins the emotional truth "
            "or pacing of the scene, you may ignore it, BUT you must vigorously defend your choice in your defense_notes.\n"
            "- Preserve what already works well.\n"
            "- Maintain proper screenplay format throughout.\n"
            "- Ensure dialogue is subtext-rich -- characters rarely say exactly what they mean\n"
            "- Use the CRAFT REFERENCE EXAMPLES for technique guidance on whatever "
            "the prompt's genre demands"
        )
        user_msg = (
            "CRAFT REFERENCE EXAMPLES (technique only -- not content to copy):\n{rag_context}\n\n"
            "ORIGINAL PROMPT:\n{prompt}\n\n"
            "APPROVED OUTLINE:\n{outline}\n\n"
            "PAST CRITIQUES HISTORY:\n{history_str}\n\n"
            "CURRENT DRAFT:\n{draft}\n\n"
            "LATEST CRITIC FEEDBACK:\n{critique}\n\n"
            "Provide your defense notes and the fully revised script scene:"
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("user", user_msg),
        ])
        chain = prompt_template | llm

        result: WriterOutput = invoke_with_retry(
            chain,
            {
                "rag_context": rag_context,
                "prompt": prompt_text,
                "outline": outline,
                "history_str": history_str if history_str else "None.",
                "draft": draft,
                "critique": critique,
            },
            caller="Writer",
        )

    logger.info("Writer's Defense Notes:\n%s", result.defense_notes)

    return {
        "draft": result.draft,
        "defense_notes": result.defense_notes
    }
