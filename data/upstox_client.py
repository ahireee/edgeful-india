"""Thin wrapper around the Upstox Python SDK.

Loads credentials from .env, configures the SDK, and exposes cached singleton
API instances with automatic retry on transient errors (5xx, 429).
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from upstox_client import ApiClient, Configuration
from upstox_client.api.history_v3_api import HistoryV3Api
from upstox_client.api.market_quote_v3_api import MarketQuoteV3Api
from upstox_client.rest import ApiException

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient Upstox API errors we should retry."""
    return isinstance(exc, ApiException) and exc.status in _RETRYABLE_STATUS_CODES


def upstox_retry(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """Decorator: retry on transient Upstox errors (429, 5xx) with exp backoff."""
    wrapped: Callable[_P, _R] = retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )(fn)
    return wrapped


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _load_token() -> str:
    """Return the best available Upstox token from the environment.

    Prefers UPSTOX_ANALYTICS_TOKEN (long-lived, read-only).
    Falls back to UPSTOX_ACCESS_TOKEN (daily OAuth).
    Raises RuntimeError if neither is set.
    """
    load_dotenv()

    token = os.environ.get("UPSTOX_ANALYTICS_TOKEN", "").strip()
    if token:
        logger.debug("Using UPSTOX_ANALYTICS_TOKEN")
        return token

    token = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip()
    if token:
        logger.debug("Falling back to UPSTOX_ACCESS_TOKEN")
        return token

    raise RuntimeError(
        "No Upstox token found. Set UPSTOX_ANALYTICS_TOKEN (preferred) "
        "or UPSTOX_ACCESS_TOKEN in your .env file."
    )


@functools.lru_cache(maxsize=1)
def _get_api_client() -> ApiClient:
    """Create and cache a configured ApiClient singleton."""
    config = Configuration()
    config.access_token = _load_token()
    return ApiClient(config)


# ---------------------------------------------------------------------------
# Public API getters (cached singletons)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def get_history_api() -> HistoryV3Api:
    """Return a cached HistoryV3Api instance."""
    return HistoryV3Api(_get_api_client())


@functools.lru_cache(maxsize=1)
def get_market_quote_api() -> MarketQuoteV3Api:
    """Return a cached MarketQuoteV3Api instance."""
    return MarketQuoteV3Api(_get_api_client())
