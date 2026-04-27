"""Hand-checked tests for the Session Bias report.

Fixture: 8 trading days with deliberately chosen open/close pairs so each
(gap-bucket, direction, green/red) combination is verified by hand.

Day 1: seed.  open=10000, close=10000 (used only as prev_close source).
Day 2: gap_up 0.20%, open=10020, close=10050 -> green.
       (bucket: 0.1-0.25%, dir: gap_up, green)
Day 3: gap_up 0.20%, open=10070, close=10040 -> red.
       (bucket: 0.1-0.25%, dir: gap_up, red)
Day 4: gap_down 0.30%, open=10010, close=10030 -> green.
       (bucket: 0.25-0.5%, dir: gap_down, green)
Day 5: gap_down 0.30%, open=10000, close=9990 -> red.
       (bucket: 0.25-0.5%, dir: gap_down, red)
Day 6: gap_up 0.05% (below MIN_GAP_PCT=0.005% — actually 0.05% IS above
       MIN_GAP_PCT, so this is the smallest bucket).  open=9995,
       close=10005 -> green.
       (bucket: 0.0-0.1%, dir: gap_up, green)
Day 7: gap_down 1.50%, open=9855, close=9700 -> red, large move.
       (bucket: 1.0-2.0%, dir: gap_down, red)
Day 8: short session (60 bars) — excluded entirely.

After exclusions: Day 1 (no prior), Day 8 (short).  6 gap days remain.

Expected per-bucket (note: each day's gap is vs the *previous trading day's*
close, so Day 5's gap is computed against Day 4's close, not Day 3's):

  0.0-0.1%   gap_up   : 1 inst (Day 6),       1 green, 0 red. green_rate=1.0
  0.1-0.25%  gap_up   : 2 inst (Day 2, 3),    1 green, 1 red. green_rate=0.5
  0.25-0.5%  gap_down : 2 inst (Day 4, 5),    1 green, 1 red. green_rate=0.5
  1.0-2.0%   gap_down : 1 inst (Day 7),       0 green, 1 red. green_rate=0.0
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from reports.base import ReportParams
from reports.session_bias import compute


def _make_day(
    date_str: str,
    open_price: float,
    close_price: float,
    *,
    num_bars: int = 375,
) -> list[dict[str, object]]:
    """Generate a synthetic day with a fixed open and close.  Intermediate
    bars sit at the midpoint; high/low envelope is unused by Session Bias."""
    base = datetime.fromisoformat(f"{date_str}T09:15:00")
    mid = (open_price + close_price) / 2
    bars: list[dict[str, object]] = []
    for i in range(num_bars):
        ts = base + timedelta(minutes=i)
        o = open_price if i == 0 else mid
        c = close_price if i == num_bars - 1 else mid
        bars.append(
            {
                "symbol": "TEST",
                "ts_ist": ts,
                "open": o,
                "high": max(o, c),
                "low": min(o, c),
                "close": c,
                "volume": 100,
            }
        )
    return bars


def _build_fixture() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(_make_day("2025-09-01", 10000.0, 10000.0))  # seed
    rows.extend(_make_day("2025-09-02", 10020.0, 10050.0))  # gap_up 0.20%, green
    rows.extend(_make_day("2025-09-03", 10070.0, 10040.0))  # gap_up 0.20%, red
    rows.extend(_make_day("2025-09-04", 10010.0, 10030.0))  # gap_down 0.30%, green
    rows.extend(_make_day("2025-09-05", 10000.0, 9990.0))  # gap_down 0.10%, red
    rows.extend(_make_day("2025-09-08", 9995.0, 10005.0))  # gap_up 0.05%, green
    rows.extend(_make_day("2025-09-09", 9855.0, 9700.0))  # gap_down 1.40%, red
    rows.extend(_make_day("2025-09-10", 9700.0, 9710.0, num_bars=60))  # short -> excluded
    return pl.DataFrame(rows)


@pytest.fixture()
def fixture_bars() -> pl.DataFrame:
    return _build_fixture()


@pytest.fixture()
def result(fixture_bars: pl.DataFrame) -> dict[str, object]:
    params = ReportParams(symbol="TEST", lookback_days=500, recency_window_days=30)
    return compute(fixture_bars, params)  # type: ignore[return-value]


def _row_map(buckets: pl.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    return {(r["bucket"], r["direction"]): r for r in buckets.iter_rows(named=True)}


class TestSessionBiasCounts:
    def test_total_gap_days(self, result: dict[str, object]) -> None:
        summary = result["summary"]
        assert isinstance(summary, dict)
        # Day 1 dropped (no prior), Day 8 dropped (short) → 6 gap-days.
        # Recompute: Day 2,3,4,5,6,7 = 6.
        assert summary["total_gap_days"] == 6

    def test_overall_green_red_split(self, result: dict[str, object]) -> None:
        summary = result["summary"]
        assert isinstance(summary, dict)
        # Greens: Day 2, 4, 6 = 3.  Reds: Day 3, 5, 7 = 3.
        assert summary["green_count"] == 3
        assert summary["red_count"] == 3
        assert summary["overall_green_rate"] == 0.5


class TestSessionBiasBuckets:
    def test_smallest_bucket_gap_up(self, result: dict[str, object]) -> None:
        rows = _row_map(result["buckets"])  # type: ignore[arg-type]
        # Day 6 is the only entry in (0.0-0.1%, gap_up) — green.
        r = rows[("0.0-0.1%", "gap_up")]
        assert r["instances"] == 1
        assert r["green_rate"] == 1.0
        assert r["red_rate"] == 0.0

    def test_mid_up_bucket_split(self, result: dict[str, object]) -> None:
        rows = _row_map(result["buckets"])  # type: ignore[arg-type]
        # Day 2 (green) + Day 3 (red) → split 50/50 in (0.1-0.25%, gap_up).
        r = rows[("0.1-0.25%", "gap_up")]
        assert r["instances"] == 2
        assert r["green_rate"] == 0.5
        assert r["red_rate"] == 0.5

    def test_down_bucket_split(self, result: dict[str, object]) -> None:
        rows = _row_map(result["buckets"])  # type: ignore[arg-type]
        # Day 4: prev_close=Day 3 close=10040, open=10010 -> gap ~-0.30%, green.
        # Day 5: prev_close=Day 4 close=10030, open=10000 -> gap ~-0.30%, red.
        # Both land in (0.25-0.5%, gap_down): 1 green + 1 red.
        r = rows[("0.25-0.5%", "gap_down")]
        assert r["instances"] == 2
        assert r["green_rate"] == 0.5
        assert r["red_rate"] == 0.5

    def test_large_gap_down_red(self, result: dict[str, object]) -> None:
        rows = _row_map(result["buckets"])  # type: ignore[arg-type]
        # Day 7: gap_down ~1.40%, big red day.
        r = rows[("1.0-2.0%", "gap_down")]
        assert r["instances"] == 1
        assert r["green_rate"] == 0.0
        assert r["red_rate"] == 1.0

    def test_avg_session_change_pct_signed(self, result: dict[str, object]) -> None:
        rows = _row_map(result["buckets"])  # type: ignore[arg-type]
        # Day 6: open=9995, close=10005 → +0.1001%.  Single instance.
        r = rows[("0.0-0.1%", "gap_up")]
        avg = r["avg_session_change_pct"]
        assert avg is not None
        assert isinstance(avg, float)
        assert 0.09 < avg < 0.11

    def test_large_gap_avg_session_change_negative(self, result: dict[str, object]) -> None:
        rows = _row_map(result["buckets"])  # type: ignore[arg-type]
        # Day 7: open=9855, close=9700 → (9700-9855)/9855 ≈ -1.573%.
        r = rows[("1.0-2.0%", "gap_down")]
        avg = r["avg_session_change_pct"]
        assert avg is not None
        assert isinstance(avg, float)
        assert -1.60 < avg < -1.55


class TestSessionBiasEdgeCases:
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

    def test_single_day_no_prior(self) -> None:
        bars = _make_day("2025-10-01", 100.0, 101.0)
        df = pl.DataFrame(bars)
        params = ReportParams(symbol="X", lookback_days=500)
        r = compute(df, params)
        assert r["summary"]["total_gap_days"] == 0

    def test_tiny_gap_excluded(self) -> None:
        """A gap below MIN_GAP_PCT (0.005%) is filtered out."""
        bars: list[dict[str, object]] = []
        bars.extend(_make_day("2025-10-01", 10000.0, 10000.0))
        # gap = (10000.001 - 10000) / 10000 * 100 = 0.00001% — well below threshold.
        bars.extend(_make_day("2025-10-02", 10000.001, 10000.5))
        df = pl.DataFrame(bars)
        params = ReportParams(symbol="X", lookback_days=500)
        r = compute(df, params)
        assert r["summary"]["total_gap_days"] == 0
