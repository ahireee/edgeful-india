"""Shared types and utilities for all reports."""

from __future__ import annotations

import math
from typing import TypedDict

import polars as pl
from pydantic import BaseModel


class ReportResult(TypedDict):
    """Standard output for every report."""

    buckets: pl.DataFrame  # probability table
    summary: dict[str, object]  # headline stats
    methodology: str  # human-readable description


class ReportParams(BaseModel):
    """Common inputs for every report."""

    symbol: str
    lookback_days: int = 180
    recency_window_days: int = 30
    or_minutes: int = 15  # opening range window (ORB / IB)
    timeframe: str = "15m"  # candle timeframe (Engulfing): "5m" | "15m" | "1h"
    lookahead_candles: int = 3  # forward window (Engulfing) used to confirm a reversal


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

# Minimum absolute gap (%) to count as a real gap. Gaps smaller than this
# are noise / rounding artifacts and are excluded.
MIN_GAP_PCT: float = 0.005  # ~0.5 basis points


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Return (lower, upper) of the Wilson score 95% confidence interval.

    Returns (0.0, 0.0) when trials == 0.
    """
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = p + z * z / (2 * trials)
    spread = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    low = max(0.0, (centre - spread) / denom)
    high = min(1.0, (centre + spread) / denom)
    return (low, high)
