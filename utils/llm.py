"""Shared LLM factory — single source of truth for model configuration.

Usage:
    from utils.llm import get_llm

    llm = get_llm(temperature=0.85)               # creative writer
    llm = get_llm(temperature=0.15)               # analytical critic
    llm = get_llm(model="gemini-2.5-pro")          # upgrade model
"""

import logging
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# Default model used across all agents
DEFAULT_MODEL = "gemini-2.5-flash"


@lru_cache(maxsize=None)
def get_llm(
    temperature: float = 0.7,
    model: str = DEFAULT_MODEL,
) -> ChatGoogleGenerativeAI:
    """Return a cached Google Gemini LLM for the given (temperature, model) pair.

    Instances are cached by their arguments, so calling
    ``get_llm(temperature=0.85)`` twice returns the same object.
    """
    logger.info("Creating LLM instance: model=%s, temperature=%.2f", model, temperature)
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


# Default free-tier Groq model — a different family from the Gemini writer, which
# makes for a stronger (bias-reduced) LLM judge than a model grading itself.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


@lru_cache(maxsize=None)
def get_groq_llm(
    temperature: float = 0.15,
    model: str = DEFAULT_GROQ_MODEL,
):
    """Return a cached Groq chat model (free tier, OpenAI-tool compatible).

    Imported lazily so the Groq dependency is only required when an agent is
    actually configured to use it. Requires the ``GROQ_API_KEY`` env var.
    """
    from langchain_groq import ChatGroq  # lazy import — only needed for Groq agents
    logger.info("Creating Groq LLM instance: model=%s, temperature=%.2f", model, temperature)
    return ChatGroq(model=model, temperature=temperature)
