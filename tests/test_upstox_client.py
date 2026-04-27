"""Unit tests for data.upstox_client — mocks the SDK so no live API calls."""

from __future__ import annotations

import functools
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_caches() -> None:
    """Clear lru_cache singletons between tests."""
    from data import upstox_client as mod

    for fn in (mod._get_api_client, mod.get_history_api, mod.get_market_quote_api):
        if isinstance(fn, functools._lru_cache_wrapper):
            fn.cache_clear()


@pytest.fixture(autouse=True)
def _reset_singletons() -> Any:
    """Ensure each test starts with fresh singletons."""
    _clear_caches()
    yield
    _clear_caches()


# ---------------------------------------------------------------------------
# Token loading
# ---------------------------------------------------------------------------


class TestLoadToken:
    def test_prefers_analytics_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("UPSTOX_ANALYTICS_TOKEN", "analytics_tok")
        monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "oauth_tok")

        from data.upstox_client import _load_token

        assert _load_token() == "analytics_tok"

    def test_falls_back_to_access_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("UPSTOX_ANALYTICS_TOKEN", "")
        monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "oauth_tok")

        from data.upstox_client import _load_token

        assert _load_token() == "oauth_tok"

    def test_raises_when_no_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("UPSTOX_ANALYTICS_TOKEN", "")
        monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "")

        from data.upstox_client import _load_token

        with pytest.raises(RuntimeError, match="No Upstox token found"):
            _load_token()

    def test_strips_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("UPSTOX_ANALYTICS_TOKEN", "  tok_123  ")
        monkeypatch.delenv("UPSTOX_ACCESS_TOKEN", raising=False)

        from data.upstox_client import _load_token

        assert _load_token() == "tok_123"


# ---------------------------------------------------------------------------
# Singleton getters
# ---------------------------------------------------------------------------


class TestSingletons:
    @patch("data.upstox_client._load_token", return_value="fake_token")
    def test_get_history_api_returns_same_instance(self, _mock_tok: Any) -> None:
        from data.upstox_client import get_history_api

        a = get_history_api()
        b = get_history_api()
        assert a is b

    @patch("data.upstox_client._load_token", return_value="fake_token")
    def test_get_market_quote_api_returns_same_instance(self, _mock_tok: Any) -> None:
        from data.upstox_client import get_market_quote_api

        a = get_market_quote_api()
        b = get_market_quote_api()
        assert a is b


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestRetry:
    def test_retries_on_429(self) -> None:
        from upstox_client.rest import ApiException

        from data.upstox_client import upstox_retry

        mock_fn = MagicMock(
            side_effect=[ApiException(status=429, reason="Too Many Requests"), "ok"]
        )

        @upstox_retry
        def call() -> str:
            result: str = mock_fn()
            return result

        assert call() == "ok"
        assert mock_fn.call_count == 2

    def test_retries_on_500(self) -> None:
        from upstox_client.rest import ApiException

        from data.upstox_client import upstox_retry

        mock_fn = MagicMock(side_effect=[ApiException(status=500, reason="Server Error"), "ok"])

        @upstox_retry
        def call() -> str:
            result: str = mock_fn()
            return result

        assert call() == "ok"
        assert mock_fn.call_count == 2

    def test_does_not_retry_on_400(self) -> None:
        from upstox_client.rest import ApiException

        from data.upstox_client import upstox_retry

        mock_fn = MagicMock(side_effect=ApiException(status=400, reason="Bad Request"))

        @upstox_retry
        def call() -> None:
            mock_fn()

        with pytest.raises(ApiException):
            call()
        assert mock_fn.call_count == 1
