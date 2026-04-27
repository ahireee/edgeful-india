"""Hand-checked tests for the Opening Range Breakout report.

Fixture: 8 trading days, 375 bars each, 15-min opening range (09:15-09:29).

Day 1: Clear upside breakout that continues.
        OR: high=100, low=98.  Bar at 09:35 high=101 (breaks up).
        EOD close=102 (above 100) -> continuation.
Day 2: Clear downside breakout that continues.
        OR: high=100, low=98.  Bar at 09:40 low=97 (breaks down).
        EOD close=96 (below 98) -> continuation.
Day 3: Upside breakout that reverses (false break).
        OR: high=100, low=98.  Bar at 09:32 high=101 (breaks up).
        EOD close=99 (below 100) -> false break.
Day 4: No breakout at all.
        OR: high=100, low=98.  All post-OR bars stay within 98-100.
Day 5: Both sides break in the same minute (excluded).
        OR: high=100, low=98.  Bar at 09:31: high=101, low=97 -> exclude.
Day 6: Downside breakout that reverses (false break).
        OR: high=100, low=98.  Bar at 09:45 low=97 (breaks down).
        EOD close=99 (above 98) -> false break.
Day 7: Short session (60 bars) -> excluded from report entirely.
Day 8: Upside breakout, continuation, large move.
        OR: high=100, low=98.  Bar at 09:31 high=101.
        EOD close=105 -> continuation, cont_size = 5%.

After exclusions: Day 5 excluded (both-side break), Day 7 excluded (short).
Remaining: 6 days (Day 1-4, 6, 8).

Expected:
  total_days = 6
  breakout_days = 5 (Day 1, 2, 3, 6, 8 -- Day 4 has no breakout)
  breakout_rate = 5/6

  upside: instances=3 (Day 1, 3, 8)
    continuation=2 (Day 1, 8), false=1 (Day 3)
    continuation_rate = 2/3

  downside: instances=2 (Day 2, 6)
    continuation=1 (Day 2), false=1 (Day 6)
    continuation_rate = 1/2
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from reports.base import ReportParams
from reports.orb import compute


def _make_bars(
    date_str: str,
    or_high: float,
    or_low: float,
    *,
    break_bar: int | None = None,
    break_side: str | None = None,
    break_both: bool = False,
    eod_close: float = 99.0,
    num_bars: int = 375,
) -> list[dict[str, object]]:
    """Generate synthetic 1-min bars for one trading day.

    The first 15 bars (09:15-09:29) form the opening range with the given
    high/low.  If break_bar is set, that post-OR bar triggers the breakout.
    """
    base = datetime.fromisoformat(f"{date_str}T09:15:00")
    mid = (or_high + or_low) / 2
    bars: list[dict[str, object]] = []

    for i in range(num_bars):
        ts = base + timedelta(minutes=i)
        o = mid
        c = eod_close if i == num_bars - 1 else mid
        h = mid
        lo = mid

        # Opening range bars: set high/low envelope
        if i < 15:
            if i == 0:
                h = or_high
            if i == 1:
                lo = or_low

        # Breakout bar
        if break_bar is not None and i == break_bar:
            if break_both:
                h = or_high + 1
                lo = or_low - 1
            elif break_side == "up":
                h = or_high + 1
            elif break_side == "down":
                lo = or_low - 1

        bars.append(
            {
                "symbol": "TEST",
                "ts_ist": ts,
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "volume": 100,
            }
        )
    return bars


def _build_fixture() -> pl.DataFrame:
    all_bars: list[dict[str, object]] = []

    # Day 1: upside breakout, continuation (close=102 > OR high=100)
    all_bars.extend(
        _make_bars(
            "2025-02-03",
            100.0,
            98.0,
            break_bar=20,
            break_side="up",
            eod_close=102.0,
        )
    )
    # Day 2: downside breakout, continuation (close=96 < OR low=98)
    all_bars.extend(
        _make_bars(
            "2025-02-04",
            100.0,
            98.0,
            break_bar=25,
            break_side="down",
            eod_close=96.0,
        )
    )
    # Day 3: upside breakout, false break (close=99 < OR high=100)
    all_bars.extend(
        _make_bars(
            "2025-02-05",
            100.0,
            98.0,
            break_bar=17,
            break_side="up",
            eod_close=99.0,
        )
    )
    # Day 4: no breakout
    all_bars.extend(
        _make_bars(
            "2025-02-06",
            100.0,
            98.0,
            eod_close=99.0,
        )
    )
    # Day 5: both sides break same minute (excluded)
    all_bars.extend(
        _make_bars(
            "2025-02-07",
            100.0,
            98.0,
            break_bar=16,
            break_both=True,
            eod_close=99.0,
        )
    )
    # Day 6: downside breakout, false break (close=99 > OR low=98)
    all_bars.extend(
        _make_bars(
            "2025-02-10",
            100.0,
            98.0,
            break_bar=30,
            break_side="down",
            eod_close=99.0,
        )
    )
    # Day 7: short session (60 bars) -- excluded
    all_bars.extend(
        _make_bars(
            "2025-02-11",
            100.0,
            98.0,
            num_bars=60,
            eod_close=99.0,
        )
    )
    # Day 8: upside breakout, big continuation (close=105, cont = 5%)
    all_bars.extend(
        _make_bars(
            "2025-02-12",
            100.0,
            98.0,
            break_bar=16,
            break_side="up",
            eod_close=105.0,
        )
    )

    return pl.DataFrame(all_bars)


@pytest.fixture()
def fixture_bars() -> pl.DataFrame:
    return _build_fixture()


@pytest.fixture()
def result(fixture_bars: pl.DataFrame) -> dict[str, object]:
    params = ReportParams(symbol="TEST", lookback_days=500, recency_window_days=30)
    return compute(fixture_bars, params)  # type: ignore[return-value]


class TestORBCounts:
    def test_total_days(self, result: dict[str, object]) -> None:
        assert result["summary"]["total_days"] == 6  # type: ignore[index]

    def test_breakout_days(self, result: dict[str, object]) -> None:
        assert result["summary"]["breakout_days"] == 5  # type: ignore[index]

    def test_breakout_rate(self, result: dict[str, object]) -> None:
        assert result["summary"]["breakout_rate"] == round(5 / 6, 4)  # type: ignore[index]

    def test_upside_instances(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        up = buckets.filter(pl.col("breakout_direction") == "upside")
        assert up.get_column("instances")[0] == 3

    def test_downside_instances(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        down = buckets.filter(pl.col("breakout_direction") == "downside")
        assert down.get_column("instances")[0] == 2

    def test_upside_continuation_rate(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        up = buckets.filter(pl.col("breakout_direction") == "upside")
        assert up.get_column("continuation_rate")[0] == round(2 / 3, 4)

    def test_downside_continuation_rate(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        down = buckets.filter(pl.col("breakout_direction") == "downside")
        assert down.get_column("continuation_rate")[0] == 0.5

    def test_upside_false_break_rate(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        up = buckets.filter(pl.col("breakout_direction") == "upside")
        assert up.get_column("false_break_rate")[0] == round(1 / 3, 4)

    def test_downside_false_break_rate(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        down = buckets.filter(pl.col("breakout_direction") == "downside")
        assert down.get_column("false_break_rate")[0] == 0.5

    def test_overall_continuation_rate(self, result: dict[str, object]) -> None:
        # 3 continuations out of 5 breakouts
        assert result["summary"]["overall_continuation_rate"] == 0.6  # type: ignore[index]


class TestORBEdgeCases:
    def test_empty_bars(self) -> None:
        df = pl.DataFrame(
            {
                "symbol": [],
                "ts_ist": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        ).cast(
            {
                "ts_ist": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            }
        )
        params = ReportParams(symbol="X", lookback_days=500)
        r = compute(df, params)
        assert r["summary"]["total_days"] == 0

    def test_short_session_excluded(self) -> None:
        """A single day with 60 bars should produce 0 days."""
        bars = _make_bars("2025-03-03", 100.0, 98.0, num_bars=60, eod_close=99.0)
        df = pl.DataFrame(bars)
        params = ReportParams(symbol="TEST", lookback_days=500)
        r = compute(df, params)
        assert r["summary"]["total_days"] == 0

    def test_configurable_or_window(self) -> None:
        """With or_minutes=5, only the first 5 bars form the range."""
        # Build a day where bars 0-4 have high=100, low=98,
        # and bar 5 (minute 5 = 09:20) breaks up.
        base = datetime(2025, 3, 3, 9, 15)
        bars: list[dict[str, object]] = []
        # Also need a "previous day" so we have at least 1 day in the report
        prev_base = datetime(2025, 2, 28, 9, 15)
        for i in range(375):
            bars.append(
                {
                    "symbol": "T",
                    "ts_ist": prev_base + timedelta(minutes=i),
                    "open": 99.0,
                    "high": 100.0,
                    "low": 98.0,
                    "close": 99.0,
                    "volume": 1,
                }
            )
        for i in range(375):
            ts = base + timedelta(minutes=i)
            h = 100.0 if i < 5 else 99.0
            lo = 98.0 if i < 5 else 99.0
            if i == 5:
                h = 101.0  # breakout at minute 5
            bars.append(
                {
                    "symbol": "T",
                    "ts_ist": ts,
                    "open": 99.0,
                    "high": h,
                    "low": lo,
                    "close": 102.0 if i == 374 else 99.0,
                    "volume": 1,
                }
            )

        df = pl.DataFrame(bars)
        params = ReportParams(symbol="T", lookback_days=500, or_minutes=5)
        r = compute(df, params)
        buckets = r["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        # Should detect the upside breakout on the second day
        up = buckets.filter(pl.col("breakout_direction") == "upside")
        assert up.height == 1
        assert up.get_column("instances")[0] >= 1
