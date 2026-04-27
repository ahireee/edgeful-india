"""Hand-checked tests for the PDH/PDL breakout report.

Fixture: 8 consecutive trading days, 375 bars each.  Day 1 is the seed
(only used as PDH/PDL source for Day 2).  Each subsequent day is built so
its prior day's high/low form a known (PDH, PDL) pair.

Convention used in `_make_day`:
  - The day's intraday range is [day_low, day_high].  Bar 0 sets the high,
    bar 1 sets the low; the rest sit at the midpoint.
  - If `break_bar` / `break_side` is set, that bar's high or low is set to
    `break_price` so it crosses the prior day's PDH or PDL.
  - The last bar's close is `eod_close`.

Days:
  Day 1: high=100, low=95, close=98.   (seed only)
  Day 2: PDH=100, PDL=95.  Breaks PDH at bar 30 (high=101), EOD=102.
         -> upside breakout, continuation, cont_size = (102-100)/100 = 2%
  Day 3: PDH=101 (Day 2's high), PDL=95 (Day 2's low=95 since mid=98).
         Wait: Day 2's day_low. Day 2 bar 1 sets low to 95 (we choose).
         Breaks PDL at bar 60 (low=94), EOD=98.
         -> downside breakout, false break (EOD 98 > PDL 95).
  Day 4: PDH=Day 3 high, PDL=Day 3 low.  No breakout — stays inside.
  Day 5: Both sides break in the same minute -> excluded.
  Day 6: Breaks PDH, continuation.
  Day 7: Breaks PDL, continuation (cont_size = (PDL - EOD)/PDL).
  Day 8: Short session (60 bars) — excluded entirely.

After exclusions:
  - Day 1 is dropped (no prior day).
  - Day 5 is excluded (both-side same-minute break).
  - Day 8 is excluded (short session) — and it also can't be a "prior day"
    for anything since it's the last day.
  Net: total_days = 5 (days 2, 3, 4, 6, 7).
       breakout_days = 4 (days 2, 3, 6, 7 — day 4 had no break).
       upside instances: 2 (days 2, 6), both continuations -> cont_rate 1.0
       downside instances: 2 (days 3 false, 7 cont) -> cont_rate 0.5
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from reports.base import ReportParams
from reports.pdh_pdl import compute


def _make_day(
    date_str: str,
    day_high: float,
    day_low: float,
    *,
    eod_close: float,
    break_bar: int | None = None,
    break_side: str | None = None,  # "up" | "down" | "both"
    break_price: float | None = None,
    num_bars: int = 375,
) -> list[dict[str, object]]:
    """Generate 1-min bars for one trading day with a controlled high/low
    envelope and an optional breakout bar."""
    base = datetime.fromisoformat(f"{date_str}T09:15:00")
    mid = (day_high + day_low) / 2
    bars: list[dict[str, object]] = []

    for i in range(num_bars):
        ts = base + timedelta(minutes=i)
        h = mid
        lo = mid
        c = eod_close if i == num_bars - 1 else mid

        if i == 0:
            h = day_high
        elif i == 1:
            lo = day_low

        if break_bar is not None and i == break_bar:
            if break_side == "up" and break_price is not None:
                h = break_price
            elif break_side == "down" and break_price is not None:
                lo = break_price
            elif break_side == "both" and break_price is not None:
                # break_price is interpreted as upside; downside symmetric
                h = break_price
                lo = day_low - 1

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
    bars: list[dict[str, object]] = []

    # Day 1: seed.  high=100, low=95, eod 98.
    bars.extend(_make_day("2025-06-02", 100.0, 95.0, eod_close=98.0))

    # Day 2: PDH=100, PDL=95.  Up break at bar 30 to 101, EOD 102.
    bars.extend(
        _make_day(
            "2025-06-03",
            day_high=100.5,  # within prior range until break_bar
            day_low=96.0,
            eod_close=102.0,
            break_bar=30,
            break_side="up",
            break_price=101.0,
        )
    )

    # Day 3: PDH=101 (Day 2 break), PDL=95 (Day 2 low envelope).
    # Down break at bar 60 to 94, EOD 98 (false break — back inside).
    bars.extend(
        _make_day(
            "2025-06-04",
            day_high=100.5,
            day_low=96.0,
            eod_close=98.0,
            break_bar=60,
            break_side="down",
            break_price=94.0,
        )
    )

    # Day 4: PDH=100.5, PDL=94.  No break: range stays inside.
    bars.extend(_make_day("2025-06-05", 100.0, 95.0, eod_close=97.0))

    # Day 5: PDH=100 (Day 4 high), PDL=95 (Day 4 low).  Both sides break in
    # the same minute -> excluded.  At break_bar=40, _make_day's "both" branch
    # sets h=break_price=101 (>PDH) and lo=day_low-1=94.5 (<PDL=95).
    bars.extend(
        _make_day(
            "2025-06-06",
            day_high=99.5,
            day_low=95.5,
            eod_close=97.0,
            break_bar=40,
            break_side="both",
            break_price=101.0,
        )
    )

    # Day 6: PDH=101 (Day 5's break), PDL=94 (Day 5's break).
    # Up break at bar 50 to 102, EOD 103 (continuation).
    bars.extend(
        _make_day(
            "2025-06-09",
            day_high=100.5,
            day_low=95.0,
            eod_close=103.0,
            break_bar=50,
            break_side="up",
            break_price=102.0,
        )
    )

    # Day 7: PDH=102 (Day 6 break), PDL=94 (Day 6 low envelope=95, but Day 5's
    # downside break to 93 means Day 5 day_low was 93. To keep it simple,
    # Day 6's day_low = 95 -> PDL_for_day7 = 95).
    # Down break at bar 70 to 94, EOD 92 (continuation, EOD < PDL).
    bars.extend(
        _make_day(
            "2025-06-10",
            day_high=100.5,
            day_low=95.5,
            eod_close=92.0,
            break_bar=70,
            break_side="down",
            break_price=94.0,
        )
    )

    # Day 8: short session — excluded.  Only 60 bars.
    bars.extend(
        _make_day(
            "2025-06-11",
            100.0,
            95.0,
            eod_close=97.0,
            num_bars=60,
        )
    )

    return pl.DataFrame(bars)


@pytest.fixture()
def fixture_bars() -> pl.DataFrame:
    return _build_fixture()


@pytest.fixture()
def result(fixture_bars: pl.DataFrame) -> dict[str, object]:
    params = ReportParams(symbol="TEST", lookback_days=500, recency_window_days=30)
    return compute(fixture_bars, params)  # type: ignore[return-value]


class TestPDHPDLCounts:
    def test_total_days(self, result: dict[str, object]) -> None:
        # Days 2, 3, 4, 6, 7 → 5.  (Day 1 dropped: no prior; Day 5 excluded:
        # both-side break; Day 8 excluded: short session.)
        summary = result["summary"]
        assert isinstance(summary, dict)
        assert summary["total_days"] == 5

    def test_breakout_days(self, result: dict[str, object]) -> None:
        # Days 2, 3, 6, 7 broke a side.  Day 4 did not.
        summary = result["summary"]
        assert isinstance(summary, dict)
        assert summary["breakout_days"] == 4

    def test_breakout_rate(self, result: dict[str, object]) -> None:
        summary = result["summary"]
        assert isinstance(summary, dict)
        assert summary["breakout_rate"] == round(4 / 5, 4)

    def test_upside_instances(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        up = buckets.filter(pl.col("breakout_direction") == "upside")
        # Day 2 + Day 6 = 2
        assert up.get_column("instances")[0] == 2

    def test_downside_instances(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        down = buckets.filter(pl.col("breakout_direction") == "downside")
        # Day 3 (false) + Day 7 (cont) = 2
        assert down.get_column("instances")[0] == 2


class TestPDHPDLRates:
    def test_upside_continuation_rate_full(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        up = buckets.filter(pl.col("breakout_direction") == "upside")
        # Both Day 2 (102 > 100) and Day 6 (103 > 101) continue.
        assert up.get_column("continuation_rate")[0] == 1.0

    def test_downside_continuation_rate_half(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        down = buckets.filter(pl.col("breakout_direction") == "downside")
        # Day 3 false (98 > 95), Day 7 cont (92 < 95).
        assert down.get_column("continuation_rate")[0] == 0.5

    def test_upside_false_break_rate(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        up = buckets.filter(pl.col("breakout_direction") == "upside")
        assert up.get_column("false_break_rate")[0] == 0.0

    def test_downside_false_break_rate(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        down = buckets.filter(pl.col("breakout_direction") == "downside")
        assert down.get_column("false_break_rate")[0] == 0.5

    def test_overall_continuation_rate(self, result: dict[str, object]) -> None:
        # 3 continuations (Day 2, 6, 7) out of 4 breakouts.
        summary = result["summary"]
        assert isinstance(summary, dict)
        assert summary["overall_continuation_rate"] == 0.75


class TestPDHPDLContSize:
    def test_upside_avg_continuation_size_pct(self, result: dict[str, object]) -> None:
        """Day 2: (102-100)/100 = 2.0; Day 6: (103-101)/101 ≈ 1.9802.
        Average ≈ 1.9901."""
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        up = buckets.filter(pl.col("breakout_direction") == "upside")
        avg = up.get_column("avg_continuation_size_pct")[0]
        assert avg is not None
        assert 1.98 < avg < 2.00


class TestPDHPDLEdgeCases:
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

    def test_single_day_no_prior(self) -> None:
        """One full session has no prior day, so report is empty."""
        bars = _make_day("2025-07-01", 100.0, 95.0, eod_close=97.0)
        df = pl.DataFrame(bars)
        params = ReportParams(symbol="X", lookback_days=500)
        r = compute(df, params)
        assert r["summary"]["total_days"] == 0

    def test_short_prior_day_skipped_for_pdh(self) -> None:
        """A short prior day (60 bars) is excluded as a prior reference;
        the next normal day's 'prior' should walk back to the previous valid
        day, not crash."""
        bars: list[dict[str, object]] = []
        bars.extend(_make_day("2025-08-01", 100.0, 95.0, eod_close=98.0))  # valid prior
        bars.extend(
            _make_day("2025-08-04", 105.0, 102.0, eod_close=103.0, num_bars=60)  # short, skipped
        )
        bars.extend(_make_day("2025-08-05", 102.0, 96.0, eod_close=97.0))  # uses Day 1 as prior
        df = pl.DataFrame(bars)
        params = ReportParams(symbol="X", lookback_days=500)
        r = compute(df, params)
        # Day 1 is dropped (no prior). Day 2 is dropped (short).
        # Day 3 has a prior valid day (Day 1).  total_days should be 1.
        assert r["summary"]["total_days"] == 1
