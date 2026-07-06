"""LangSmith tracing configuration — opt-in observability for all LLM calls.

LangChain/LangGraph auto-instrument every runnable when LangSmith tracing is
enabled via environment variables, so no per-call wiring is needed. This module
just normalizes the env, logs whether tracing is active, and lets callers tag a
run with a project + metadata.

Enable by setting these in `.env` (all optional — absent = tracing off):

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=ls__...          # https://smith.langchain.com/settings
    LANGSMITH_PROJECT=script-doctor    # optional; defaults below

Once enabled, per-agent token counts, latency, inputs/outputs, and cost show up
in the LangSmith UI grouped under the project name.
"""

import os
import logging

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "script-doctor"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def tracing_enabled() -> bool:
    """True if LangSmith tracing is switched on and a key is present."""
    switch = _truthy(os.getenv("LANGSMITH_TRACING")) or _truthy(os.getenv("LANGCHAIN_TRACING_V2"))
    has_key = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
    return switch and has_key


def configure_tracing(project: str | None = None) -> bool:
    """Normalize LangSmith env vars and log the tracing status.

    Bridges the legacy ``LANGCHAIN_*`` names to the current ``LANGSMITH_*`` ones
    so either style in `.env` works. Safe to call multiple times. Returns whether
    tracing is active.
    """
    # Bridge legacy <-> current env var names in both directions.
    if _truthy(os.getenv("LANGCHAIN_TRACING_V2")) and not os.getenv("LANGSMITH_TRACING"):
        os.environ["LANGSMITH_TRACING"] = "true"
    if _truthy(os.getenv("LANGSMITH_TRACING")) and not os.getenv("LANGCHAIN_TRACING_V2"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if os.getenv("LANGCHAIN_API_KEY") and not os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = os.environ["LANGCHAIN_API_KEY"]

    if not tracing_enabled():
        logger.info("LangSmith tracing: OFF (set LANGSMITH_TRACING=true + LANGSMITH_API_KEY to enable).")
        return False

    resolved = project or os.getenv("LANGSMITH_PROJECT") or DEFAULT_PROJECT
    os.environ["LANGSMITH_PROJECT"] = resolved
    os.environ["LANGCHAIN_PROJECT"] = resolved  # legacy consumers
    logger.info("LangSmith tracing: ON  (project=%s)", resolved)
    return True
