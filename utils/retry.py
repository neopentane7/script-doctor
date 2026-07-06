"""Shared retry utility for LLM chain invocations.

Usage:
    from utils.retry import invoke_with_retry

    result = invoke_with_retry(chain, {"prompt": "...", "draft": "..."})
"""

import time
import random
import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 2  # seconds

# Error class names that indicate a PERMANENT failure — retrying them only wastes
# the backoff budget (a bad API key or malformed request won't fix itself).
_PERMANENT_EXC_NAMES = frozenset({
    "InvalidArgument", "BadRequest", "ValidationError", "PermissionDenied",
    "Unauthenticated", "Unauthorized", "Forbidden", "NotFound",
})
# HTTP status codes that are permanent client errors (never worth retrying).
_PERMANENT_STATUS_CODES = frozenset({400, 401, 403, 404, 422})


def _is_permanent(exc: Exception) -> bool:
    """Best-effort classification: True if retrying ``exc`` cannot succeed.

    Unknown/ambiguous errors default to retryable (return False) so transient
    network and rate-limit failures stay resilient.
    """
    if type(exc).__name__ in _PERMANENT_EXC_NAMES:
        return True
    # google-genai / httpx surface HTTP status via `code` or `status_code`
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and value in _PERMANENT_STATUS_CODES:
            return True
    return False


def invoke_with_retry(
    chain,
    kwargs: dict[str, Any],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    caller: str = "Agent",
) -> Any:
    """Invoke a LangChain chain with exponential backoff retry and jitter.

    Args:
        chain:       A LangChain runnable (prompt | llm).
        kwargs:      Dictionary of variables to pass to ``chain.invoke()``.
        max_retries: Maximum number of attempts before raising.
        base_delay:  Base delay in seconds (doubled each retry, with jitter).
        caller:      Label for log messages (e.g. "Writer", "Critic").

    Returns:
        The result of ``chain.invoke(kwargs)``.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            return chain.invoke(kwargs)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "[%s] Error on attempt %d/%d: %s",
                caller, attempt + 1, max_retries, exc,
            )
            # Don't burn the retry budget on failures that can't recover.
            if _is_permanent(exc):
                logger.error(
                    "[%s] Permanent error (%s) — not retrying.",
                    caller, type(exc).__name__,
                )
                raise
            if attempt < max_retries - 1:
                # Exponential backoff with ±50% jitter to avoid thundering herd
                base = base_delay * (2 ** attempt)
                jitter = random.uniform(0, base * 0.5)
                delay = base + jitter
                logger.info("[%s] Retrying in %.1fs...", caller, delay)
                time.sleep(delay)

    # All retries exhausted — re-raise the last exception
    raise last_exc  # type: ignore[misc]
