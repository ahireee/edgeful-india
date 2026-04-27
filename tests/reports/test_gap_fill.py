"""Hand-checked tests for the Gap Fill report.

The synthetic dataset covers 10 trading days with known gaps, fill statuses,
and bucket assignments.  Every count is verified by hand.

Day layout (375 bars per day, 09:15 - 15:29):
  Day 1: baseline — no previous close → excluded (first day)
  Day 2: up-gap 0.05%, day_low=10001 > prev_close 10000 → NOT filled
  Day 3: up-gap 0.20% → FILLED at bar 30 (30 min)
  Day 4: down-gap 0.15% → NOT filled
  Day 5: up-gap 0.60% → FILLED at bar 60 (60 min)
  Day 6: down-gap 1.50% → FILLED at bar 120 (120 min)
  Day 7: up-gap 0.08% → FILLED at bar 10 (10 min)
  Day 8: down-gap 2.50% → NOT filled
  Day 9: up-gap 0.30% → FILLED at bar 45 (45 min)
  Day 10: down-gap 0.40% → FILLED at bar 90 (90 min)

After exclusions (day 1: no prev day), we have 9 gap-days.
Day 2's 0.05% gap IS included (> MIN_GAP_PCT of 0.005%).

Expected bucket/direction breakdown:
  0.0-0.1%  up:    2 instances (day 2 + day 7), 1 fill (day 7) → rate = 0.5
  0.1-0.25% up:    1 instance (day 3), 1 fill  → rate = 1.0
  0.1-0.25% down:  1 instance (day 4), 0 fills → rate = 0.0
  0.25-0.5% up:    1 instance (day 9), 1 fill  → rate = 1.0
  0.25-0.5% down:  1 instance (day 10), 1 fill → rate = 1.0
  0.5-1.0%  up:    1 instance (day 5), 1 fill  → rate = 1.0
  1.0-2.0%  down:  1 instance (day 6), 1 fill  → rate = 1.0
  2.0%+     down:  1 instance (day 8), 0 fills → rate = 0.0
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from reports.base import ReportParams, wilson_ci
from reports.gap_fill import compute


def _make_day_bars(
    date_str: str,
    open_price: float,
    close_price: float,
    day_high: float,
    day_low: float,
    fill_bar: int | None = None,
    fill_price: float | None = None,
) -> list[dict[str, object]]:
    """Generate 375 synthetic 1-min bars for one trading day.

    All bars have OHLC = open_price except:
    - bar 0 (09:15): open = open_price
    - bar [fill_bar] onward: price reaches fill_price via high or low
    - last bar: close = close_price
    - global high/low envelope applied
    """
    base_dt = datetime.fromisoformat(f"{date_str}T09:15:00")
    bars: list[dict[str, object]] = []
    mid = (open_price + close_price) / 2

    for i in range(375):
        ts = base_dt + timedelta(minutes=i)
        o = open_price if i == 0 else mid
        c = close_price if i == 374 else mid
        h = mid
        lo = mid

        # At fill_bar, make the price reach fill_price
        if fill_bar is not None and fill_price is not None and i == fill_bar:
            if fill_price < mid:
                lo = fill_price
            else:
                h = fill_price

        # Enforce day high/low on all bars
        if h < day_high and i == 1:
            h = day_high
        if lo > day_low and i == 2:
            lo = day_low

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
    """Build the 10-day synthetic dataset described in the module docstring."""
    all_bars: list[dict[str, object]] = []

    # Day 1: baseline.  Close = 10000.0
    all_bars.extend(_make_day_bars("2025-01-06", 10000.0, 10000.0, 10010.0, 9990.0))

    # Day 2: up-gap 0.05% (tiny, excluded). open = 10000 * 1.0005 = 10005.0
    # prev_close = 10000.0 → gap = 0.05%
    all_bars.extend(_make_day_bars("2025-01-07", 10005.0, 10010.0, 10015.0, 10001.0))

    # Day 3: up-gap 0.20%. prev_close = 10010.0, open = 10010 * 1.002 = 10030.02
    # Fill when low touches 10010.0 → bar 30
    all_bars.extend(
        _make_day_bars(
            "2025-01-08",
            10030.0,
            10025.0,
            10035.0,
            10008.0,
            fill_bar=30,
            fill_price=10010.0,
        )
    )

    # Day 4: down-gap 0.15%. prev_close = 10025.0, open = 10025 * 0.9985 = 10009.96
    # NOT filled — high never reaches 10025.0
    all_bars.extend(_make_day_bars("2025-01-09", 10010.0, 10012.0, 10020.0, 10005.0))

    # Day 5: up-gap 0.60%. prev_close = 10012.0, open = 10012 * 1.006 = 10072.07
    # Fill when low touches 10012.0 → bar 60
    all_bars.extend(
        _make_day_bars(
            "2025-01-10",
            10072.0,
            10060.0,
            10080.0,
            10010.0,
            fill_bar=60,
            fill_price=10012.0,
        )
    )

    # Day 6: down-gap 1.50%. prev_close = 10060.0, open = 10060 * 0.985 = 9909.1
    # Fill when high reaches 10060.0 → bar 120
    all_bars.extend(
        _make_day_bars(
            "2025-01-13",
            9909.0,
            9950.0,
            10065.0,
            9900.0,
            fill_bar=120,
            fill_price=10060.0,
        )
    )

    # Day 7: up-gap 0.08%. prev_close = 9950.0, open = 9950 * 1.0008 = 9957.96
    # Fill when low touches 9950.0 → bar 10
    all_bars.extend(
        _make_day_bars(
            "2025-01-14",
            9958.0,
            9955.0,
            9965.0,
            9948.0,
            fill_bar=10,
            fill_price=9950.0,
        )
    )

    # Day 8: down-gap 2.50%. prev_close = 9955.0, open = 9955 * 0.975 = 9706.125
    # NOT filled — high stays below 9955.0
    all_bars.extend(_make_day_bars("2025-01-15", 9706.0, 9720.0, 9750.0, 9700.0))

    # Day 9: up-gap 0.30%. prev_close = 9720.0, open = 9720 * 1.003 = 9749.16
    # Fill when low touches 9720.0 → bar 45
    all_bars.extend(
        _make_day_bars(
            "2025-01-16",
            9749.0,
            9740.0,
            9755.0,
            9718.0,
            fill_bar=45,
            fill_price=9720.0,
        )
    )

    # Day 10: down-gap 0.40%. prev_close = 9740.0, open = 9740 * 0.996 = 9701.04
    # Fill when high reaches 9740.0 → bar 90
    all_bars.extend(
        _make_day_bars(
            "2025-01-17",
            9701.0,
            9730.0,
            9745.0,
            9695.0,
            fill_bar=90,
            fill_price=9740.0,
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


class TestGapFillCounts:
    """Verify bucket assignments and instance counts from the hand-checked fixture."""

    def test_total_gap_days(self, result: dict[str, object]) -> None:
        summary = result["summary"]
        assert isinstance(summary, dict)
        assert summary["total_gap_days"] == 9

    def test_total_fills(self, result: dict[str, object]) -> None:
        summary = result["summary"]
        assert isinstance(summary, dict)
        assert summary["total_fills"] == 6

    def test_overall_fill_rate(self, result: dict[str, object]) -> None:
        summary = result["summary"]
        assert isinstance(summary, dict)
        # 6 fills / 9 gap-days
        assert summary["overall_fill_rate"] == round(6 / 9, 4)

    def test_bucket_instance_counts(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)

        rows = {(r["bucket"], r["direction"]): r for r in buckets.iter_rows(named=True)}

        # 0.0-0.1% up: day 2 + day 7
        assert rows[("0.0-0.1%", "up")]["instances"] == 2

        # 0.1-0.25% up: day 3
        assert rows[("0.1-0.25%", "up")]["instances"] == 1

        # 0.1-0.25% down: day 4
        assert rows[("0.1-0.25%", "down")]["instances"] == 1

        # 0.25-0.5% up: day 9
        assert rows[("0.25-0.5%", "up")]["instances"] == 1

        # 0.25-0.5% down: day 10
        assert rows[("0.25-0.5%", "down")]["instances"] == 1

        # 0.5-1.0% up: day 5
        assert rows[("0.5-1.0%", "up")]["instances"] == 1

        # 1.0-2.0% down: day 6
        assert rows[("1.0-2.0%", "down")]["instances"] == 1

        # 2.0%+ down: day 8
        assert rows[("2.0%+", "down")]["instances"] == 1

    def test_fill_rates(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)

        rows = {(r["bucket"], r["direction"]): r for r in buckets.iter_rows(named=True)}

        # 100% fill rate buckets
        for key in [
            ("0.1-0.25%", "up"),
            ("0.25-0.5%", "up"),
            ("0.25-0.5%", "down"),
            ("0.5-1.0%", "up"),
            ("1.0-2.0%", "down"),
        ]:
            assert rows[key]["fill_rate"] == 1.0, f"Expected 100% fill for {key}"

        # 0.0-0.1% up: 1 fill out of 2 → 50%
        assert rows[("0.0-0.1%", "up")]["fill_rate"] == 0.5

        # Unfilled buckets
        assert rows[("0.1-0.25%", "down")]["fill_rate"] == 0.0
        assert rows[("2.0%+", "down")]["fill_rate"] == 0.0

    def test_bucket_row_count(self, result: dict[str, object]) -> None:
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        # 8 distinct (bucket, direction) combos
        assert buckets.height == 8


class TestGapFillEdgeCases:
    """Edge cases: tiny gaps, short sessions, empty data."""

    def test_includes_small_but_real_gaps(self, result: dict[str, object]) -> None:
        """Day 2 (0.05% gap) is above MIN_GAP_PCT so it IS included."""
        buckets = result["buckets"]
        assert isinstance(buckets, pl.DataFrame)
        rows = {(r["bucket"], r["direction"]): r for r in buckets.iter_rows(named=True)}
        # 0.0-0.1% up has 2 instances (day 2 + day 7)
        assert rows[("0.0-0.1%", "up")]["instances"] == 2

    def test_short_session_excluded(self) -> None:
        """A day with only 60 bars should be excluded from the report."""
        # Build a minimal 2-day dataset: day 1 normal, day 2 short (60 bars)
        bars_day1: list[dict[str, object]] = []
        base = datetime(2025, 1, 6, 9, 15)
        for i in range(375):
            bars_day1.append(
                {
                    "symbol": "X",
                    "ts_ist": base + timedelta(minutes=i),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1,
                }
            )

        bars_day2: list[dict[str, object]] = []
        base2 = datetime(2025, 1, 7, 9, 15)
        for i in range(60):
            bars_day2.append(
                {
                    "symbol": "X",
                    "ts_ist": base2 + timedelta(minutes=i),
                    "open": 101.0,
                    "high": 102.0,
                    "low": 100.5,
                    "close": 101.0,
                    "volume": 1,
                }
            )

        df = pl.DataFrame(bars_day1 + bars_day2)
        params = ReportParams(symbol="X", lookback_days=500)
        r = compute(df, params)
        # Day 2 is excluded (60 bars < 300), so no gaps at all
        assert r["summary"]["total_gap_days"] == 0

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
        assert r["summary"]["total_gap_days"] == 0


class TestWilsonCI:
    def test_zero_trials(self) -> None:
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_all_successes(self) -> None:
        lo, hi = wilson_ci(10, 10)
        assert lo > 0.6
        assert hi == 1.0

    def test_no_successes(self) -> None:
        lo, hi = wilson_ci(0, 10)
        assert lo == 0.0
        assert hi < 0.4

    def test_half(self) -> None:
        lo, hi = wilson_ci(50, 100)
        assert 0.39 < lo < 0.42
        assert 0.58 < hi < 0.61
