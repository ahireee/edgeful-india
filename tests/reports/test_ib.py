"""Hand-checked tests for the Initial Balance Breakout report.

The IB window is the first 60 bars (09:15-10:14).  These tests focus on
behaviour that is IB-specific — the ORB compute logic itself is already
covered by tests/reports/test_orb.py, so we only verify:

  1. The IB window is exactly 60 bars (bars 0-59 are inside, bar 60 is the
     first post-IB bar).
  2. A breakout that happens before bar 60 is *not* counted (it's still
     inside the IB).
  3. or_minutes is forced to 60 even when the caller passes something else.
  4. The methodology text is the IB methodology, not ORB's.
  5. Standard ORB metrics still flow through correctly with the 60-min window.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from reports.base import ReportParams
from reports.ib import IB_MINUTES, compute


def _make_bars(
    date_str: str,
    ib_high: float,
    ib_low: float,
    *,
    break_bar: int | None = None,
    break_side: str | None = None,
    break_both: bool = False,
    eod_close: float = 99.0,
    num_bars: int = 375,
) -> list[dict[str, object]]:
    """Generate 1-min bars for one trading day.

    Bars 0-59 form the IB with the given high/low envelope.  If break_bar
    is set, that bar carries a high/low that pierces the IB.
    """
    base = datetime.fromisoformat(f"{date_str}T09:15:00")
    mid = (ib_high + ib_low) / 2
    bars: list[dict[str, object]] = []

    for i in range(num_bars):
        ts = base + timedelta(minutes=i)
        h = mid
        lo = mid
        c = eod_close if i == num_bars - 1 else mid

        # Apply IB envelope inside the first 60 bars
        if i < IB_MINUTES:
            if i == 0:
                h = ib_high
            if i == 1:
                lo = ib_low

        if break_bar is not None and i == break_bar:
            if break_both:
                h = ib_high + 1
                lo = ib_low - 1
            elif break_side == "up":
                h = ib_high + 1
            elif break_side == "down":
                lo = ib_low - 1

        bars.append(
            {
                "symbol": "TEST",
                "ts_ist": ts,
                "open": mid,
                "high": h,
                "low": lo,
                "close": c,
                "volume": 100,
            }
        )
    return bars


def _build_fixture() -> pl.DataFrame:
    """Six trading days designed for the IB report.

    Day 1: Upside IB breakout, continuation. Break at bar 65, EOD 102.
    Day 2: Downside IB breakout, continuation. Break at bar 75, EOD 96.
    Day 3: Upside IB breakout, false break. Break at bar 80, EOD 99.
    Day 4: No breakout — all post-IB bars stay inside the range.
    Day 5: Both sides break same minute (bar 70, both) — excluded.
    Day 6: "Breakout" placed at bar 30 (inside IB window) -> NOT a breakout
            for IB purposes; the bar's high/low simply widens the IB itself.
    """
    bars: list[dict[str, object]] = []
    bars.extend(
        _make_bars("2025-04-07", 100.0, 98.0, break_bar=65, break_side="up", eod_close=102.0)
    )
    bars.extend(
        _make_bars("2025-04-08", 100.0, 98.0, break_bar=75, break_side="down", eod_close=96.0)
    )
    bars.extend(
        _make_bars("2025-04-09", 100.0, 98.0, break_bar=80, break_side="up", eod_close=99.0)
    )
    bars.extend(_make_bars("2025-04-10", 100.0, 98.0, eod_close=99.0))
    bars.extend(
        _make_bars("2025-04-11", 100.0, 98.0, break_bar=70, break_both=True, eod_close=99.0)
    )
    bars.extend(
        _make_bars("2025-04-14", 100.0, 98.0, break_bar=30, break_side="up", eod_close=99.0)
    )
    return pl.DataFrame(bars)


@pytest.fixture()
def fixture_bars() -> pl.DataFrame:
    return _build_fixture()


@pytest.fixture()
def result(fixture_bars: pl.DataFrame) -> dict[str, object]:
    params = ReportParams(symbol="TEST", lookback_days=500, recency_window_days=30)
    return compute(fixture_bars, params)  # type: ignore[return-value]


class TestIBWindow:
    def test_or_minutes_in_summary_is_60(self, result: dict[str, object]) -> None:
        assert result["summary"]["or_minutes"] == 60  # type: ignore[index]

    def test_total_days_excludes_both_break(self, result: dict[str, object]) -> None:
        # Day 5 is excluded (both-side break), so 5 days remain.
        assert result["summary"]["total_days"] == 5  # type: ignore[index]

    def test_breakout_happening_inside_ib_is_not_counted(self, result: dict[str, object]) -> None:
        """Day 6 has a 'breakout' bar at minute 30, which is inside the
        60-minute IB window.  That bar's high simply becomes part of the
        IB itself, so post-IB never breaks.  Day 6 should be a no-breakout
        day, not a breakout day."""
        # Days that breakout: 1 (up), 2 (down), 3 (up). Day 4, 6 = no break.
        assert result["summary"]["breakout_days"] == 3  # type: ignore[index]


class TestIBStats:
    def test_upside_instances(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        up = buckets.filter(pl.col("breakout_direction") == "upside")
        # Day 1 (cont) + Day 3 (false) = 2
        assert up.get_column("instances")[0] == 2

    def test_downside_instances(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        down = buckets.filter(pl.col("breakout_direction") == "downside")
        # Day 2 only
        assert down.get_column("instances")[0] == 1

    def test_upside_continuation_rate(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        up = buckets.filter(pl.col("breakout_direction") == "upside")
        # 1 cont (Day 1) out of 2 -> 0.5
        assert up.get_column("continuation_rate")[0] == 0.5

    def test_downside_continuation_rate_full(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        down = buckets.filter(pl.col("breakout_direction") == "downside")
        # Day 2 alone, continuation -> 1.0
        assert down.get_column("continuation_rate")[0] == 1.0


class TestIBSemantics:
    def test_caller_or_minutes_is_overridden(self, fixture_bars: pl.DataFrame) -> None:
        """Even if the caller hands us or_minutes=15, the IB report must
        force the 60-minute window."""
        params = ReportParams(
            symbol="TEST", lookback_days=500, recency_window_days=30, or_minutes=15
        )
        r = compute(fixture_bars, params)
        assert r["summary"]["or_minutes"] == 60

    def test_methodology_is_ib_not_orb(self, result: dict[str, object]) -> None:
        m = result["methodology"]
        assert isinstance(m, str)
        assert "Initial Balance" in m
        assert "Opening Range Breakout" not in m

    def test_breakout_at_bar_60_is_first_valid_post_ib_bar(self) -> None:
        """A breakout at exactly bar 60 (first post-IB bar) should be
        counted.  Bar 59 is the last IB bar; bar 60 is post-IB."""
        bars = _make_bars("2025-05-05", 100.0, 98.0, break_bar=60, break_side="up", eod_close=102.0)
        # Need at least 2 trading days for the report (matches ORB's expectations).
        prev = _make_bars("2025-05-02", 100.0, 98.0, eod_close=99.0)
        df = pl.DataFrame(prev + bars)
        params = ReportParams(symbol="TEST", lookback_days=500)
        r = compute(df, params)
        assert r["summary"]["breakout_days"] == 1
        buckets = r["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        up = buckets.filter(pl.col("breakout_direction") == "upside")
        assert up.get_column("instances")[0] == 1
