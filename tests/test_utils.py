"""Tests for the shared utility modules."""

import pytest
from unittest.mock import MagicMock, patch, call


# ── utils.retry ───────────────────────────────────────────────────────────────

class TestInvokeWithRetry:
    """Validate retry behavior with exponential backoff."""

    def test_succeeds_on_first_attempt(self):
        """No retries needed when chain succeeds immediately."""
        from utils.retry import invoke_with_retry

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "success"

        result = invoke_with_retry(mock_chain, {"key": "val"}, caller="Test")
        assert result == "success"
        assert mock_chain.invoke.call_count == 1

    @patch("utils.retry.time.sleep")
    def test_retries_on_failure_then_succeeds(self, mock_sleep):
        """Should retry and eventually succeed."""
        from utils.retry import invoke_with_retry

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            RuntimeError("API error"),
            "success",
        ]

        result = invoke_with_retry(
            mock_chain, {"key": "val"}, max_retries=3, base_delay=1, caller="Test"
        )
        assert result == "success"
        assert mock_chain.invoke.call_count == 2
        mock_sleep.assert_called_once()

    @patch("utils.retry.time.sleep")
    def test_raises_after_all_retries_exhausted(self, mock_sleep):
        """Should raise the last exception after max retries."""
        from utils.retry import invoke_with_retry

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("Persistent error")

        with pytest.raises(RuntimeError, match="Persistent error"):
            invoke_with_retry(
                mock_chain, {"key": "val"}, max_retries=3, base_delay=0.01, caller="Test"
            )

        assert mock_chain.invoke.call_count == 3

    @patch("utils.retry.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep):
        """Delays should follow exponential backoff: base * 2^attempt with jitter."""
        from utils.retry import invoke_with_retry

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            RuntimeError("Error 1"),
            RuntimeError("Error 2"),
            "success",
        ]

        invoke_with_retry(
            mock_chain, {}, max_retries=3, base_delay=2, caller="Test"
        )

        # First retry: base is 2 * 2^0 = 2s, plus up to 50% jitter (2s to 3s)
        # Second retry: base is 2 * 2^1 = 4s, plus up to 50% jitter (4s to 6s)
        assert mock_sleep.call_count == 2
        calls = [arg[0][0] for arg in mock_sleep.call_args_list]
        assert 2.0 <= calls[0] <= 3.0
        assert 4.0 <= calls[1] <= 6.0

    def test_passes_kwargs_correctly(self):
        """Chain should receive exactly the kwargs we pass."""
        from utils.retry import invoke_with_retry

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "ok"

        kwargs = {"prompt": "test", "draft": "content"}
        invoke_with_retry(mock_chain, kwargs, caller="Test")
        mock_chain.invoke.assert_called_once_with(kwargs)


# ── utils.llm ─────────────────────────────────────────────────────────────────

class TestGetLlm:
    """Validate the cached LLM factory."""

    @patch("utils.llm.ChatGoogleGenerativeAI")
    def test_returns_llm_instance(self, mock_cls):
        """Should create and return a ChatGoogleGenerativeAI instance."""
        # Clear the lru_cache between tests
        from utils.llm import get_llm
        get_llm.cache_clear()

        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        result = get_llm(temperature=0.5, model="gemini-2.5-flash")
        assert result == mock_instance
        mock_cls.assert_called_once_with(model="gemini-2.5-flash", temperature=0.5)

    @patch("utils.llm.ChatGoogleGenerativeAI")
    def test_caches_same_args(self, mock_cls):
        """Same (temperature, model) should return the cached instance."""
        from utils.llm import get_llm
        get_llm.cache_clear()

        mock_cls.return_value = MagicMock()

        result1 = get_llm(temperature=0.7)
        result2 = get_llm(temperature=0.7)
        assert result1 is result2
        # Only one actual construction call
        assert mock_cls.call_count == 1

    @patch("utils.llm.ChatGoogleGenerativeAI")
    def test_different_temperatures_create_separate_instances(self, mock_cls):
        """Different temperature values should create separate LLM instances."""
        from utils.llm import get_llm
        get_llm.cache_clear()

        mock_cls.side_effect = [MagicMock(name="creative"), MagicMock(name="analytical")]

        result1 = get_llm(temperature=0.85)
        result2 = get_llm(temperature=0.15)
        assert result1 is not result2
        assert mock_cls.call_count == 2
