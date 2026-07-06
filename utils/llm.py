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
    """Return a cached LLM instance for the given (temperature, model) pair.

    Instances are cached by their arguments, so calling
    ``get_llm(temperature=0.85)`` twice returns the same object.
    """
    logger.info("Creating LLM instance: model=%s, temperature=%.2f", model, temperature)
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)
