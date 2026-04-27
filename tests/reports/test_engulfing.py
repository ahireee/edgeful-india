"""Hand-checked tests for the Engulfing Candle Reversals report.

Strategy: build per-day 1-min bars whose 15-minute resample produces a
hand-designed sequence of OHLC candles, including known bullish/bearish
engulfing pairs and forward outcomes.

Helper ``_make_15m_candle`` writes 15 1-min bars whose first bar's open,
last bar's close, and the day's max/min cover the requested OHLC for the
candle (intermediate bars sit at the midpoint).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from reports.base import ReportParams
from reports.engulfing import compute


def _make_15m_candle(
    base: datetime,
    candle_idx: int,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> list[dict[str, object]]:
    """Generate 15 1-min bars whose resample yields exactly (open, high, low, close)."""
    start = base + timedelta(minutes=candle_idx * 15)
    rows: list[dict[str, object]] = []
    mid = (open_ + close) / 2
    for i in range(15):
        ts = start + timedelta(minutes=i)
        if i == 0:
            o, h, lo, c = open_, max(open_, mid), min(open_, mid), mid
        elif i == 1:
            o, h, lo, c = mid, high, mid, mid  # plant the high
        elif i == 2:
            o, h, lo, c = mid, mid, low, mid  # plant the low
        elif i == 14:
            o, h, lo, c = mid, max(close, mid), min(close, mid), close
        else:
            o, h, lo, c = mid, mid, mid, mid
        rows.append(
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
    return rows


def _make_day_from_candles(
    date_str: str, candles: list[tuple[float, float, float, float]], num_candles_target: int = 25
) -> list[dict[str, object]]:
    """Build a full 25-candle (375-bar) day from a list of 15m OHLC tuples.

    Any candles beyond ``len(candles)`` are filled as flat at the prior close
    so the day has at least 300 1-min bars (Muhurat threshold).
    """
    base = datetime.fromisoformat(f"{date_str}T09:15:00")
    rows: list[dict[str, object]] = []
    last_close = candles[0][3] if candles else 100.0
    for idx in range(num_candles_target):
        if idx < len(candles):
            o, h, lo, c = candles[idx]
        else:
            o = h = lo = c = last_close
        rows.extend(_make_15m_candle(base, idx, o, h, lo, c))
        last_close = c
    return rows


def _build_fixture() -> pl.DataFrame:
    """Two trading days, each carefully designed.

    Day 1 (2025-11-03) candles:
        idx=0: red    (open 100, close 98)
        idx=1: BULLISH ENGULFING green (open 97, close 102)  # body engulfs day-0 body
        idx=2: green  (close 103) — pushes price up
        idx=3: green  (close 104)
        idx=4: green  (close 105)  # forward N+3=4 close=105 > 102 → reversal CONFIRMED
        idx=5..: flat at 105.

    Day 2 (2025-11-04) candles:
        idx=0: green  (open 100, close 103)
        idx=1: BEARISH ENGULFING red (open 104, close 99)  # body engulfs day-0 body
        idx=2: green  (close 100)  # whipsaw up
        idx=3: green  (close 101)
        idx=4: green  (close 102)  # forward N+3=4 close=102 > 99 → reversal NOT confirmed
        idx=5..: flat at 102.

    Expected per-type rows:
      bullish: 1 instance, 1 reversal -> rate 1.0, avg_forward_pct ~ +2.94%
      bearish: 1 instance, 0 reversals -> rate 0.0, avg_forward_pct = (99-102)/99 = -3.03%
    """
    bars: list[dict[str, object]] = []

    # Day 1 — bullish engulfing at idx=1
    d1_candles: list[tuple[float, float, float, float]] = [
        (100.0, 100.5, 97.5, 98.0),  # idx=0: red, body 100->98
        (97.0, 102.5, 96.5, 102.0),  # idx=1: green, body 97->102 (engulfs 100->98)
        (102.0, 103.5, 101.8, 103.0),
        (103.0, 104.5, 102.8, 104.0),
        (104.0, 105.5, 103.8, 105.0),  # idx=4 close = forward target for N+3 from idx=1
    ]
    bars.extend(_make_day_from_candles("2025-11-03", d1_candles))

    # Day 2 — bearish engulfing at idx=1, false signal (no reversal)
    d2_candles = [
        (100.0, 103.5, 99.5, 103.0),  # idx=0: green, body 100->103
        (104.0, 104.5, 98.5, 99.0),  # idx=1: red, body 104->99 (engulfs 100->103)
        (99.0, 100.5, 98.8, 100.0),
        (100.0, 101.5, 99.8, 101.0),
        (101.0, 102.5, 100.8, 102.0),  # idx=4 close = 102 > 99 → no reversal
    ]
    bars.extend(_make_day_from_candles("2025-11-04", d2_candles))

    return pl.DataFrame(bars)


@pytest.fixture()
def fixture_bars() -> pl.DataFrame:
    return _build_fixture()


@pytest.fixture()
def result(fixture_bars: pl.DataFrame) -> dict[str, object]:
    params = ReportParams(
        symbol="TEST",
        lookback_days=500,
        recency_window_days=30,
        timeframe="15m",
        lookahead_candles=3,
    )
    return compute(fixture_bars, params)  # type: ignore[return-value]


def _row_map(buckets: pl.DataFrame) -> dict[str, dict[str, object]]:
    return {r["engulf_type"]: r for r in buckets.iter_rows(named=True)}


class TestEngulfingDetection:
    def test_total_engulfings(self, result: dict[str, object]) -> None:
        summary = result["summary"]
        assert isinstance(summary, dict)
        # One bullish (Day 1 idx=1) + one bearish (Day 2 idx=1) = 2.
        assert summary["total_engulfings"] == 2

    def test_bullish_instance_count(self, result: dict[str, object]) -> None:
        rows = _row_map(result["buckets"])  # type: ignore[arg-type]
        assert rows["bullish"]["instances"] == 1

    def test_bearish_instance_count(self, result: dict[str, object]) -> None:
        rows = _row_map(result["buckets"])  # type: ignore[arg-type]
        assert rows["bearish"]["instances"] == 1


class TestReversalConfirmation:
    def test_bullish_reversal_confirmed(self, result: dict[str, object]) -> None:
        rows = _row_map(result["buckets"])  # type: ignore[arg-type]
        # Day 1: engulf close = 102, N+3 close = 105 > 102 → reversal.
        assert rows["bullish"]["reversal_rate"] == 1.0

    def test_bearish_reversal_not_confirmed(self, result: dict[str, object]) -> None:
        rows = _row_map(result["buckets"])  # type: ignore[arg-type]
        # Day 2: engulf close = 99, N+3 close = 102 > 99 → no reversal.
        assert rows["bearish"]["reversal_rate"] == 0.0

    def test_avg_forward_pct_signed(self, result: dict[str, object]) -> None:
        rows = _row_map(result["buckets"])  # type: ignore[arg-type]
        # Bullish: (105-102)/102 ≈ +2.94%.
        b_avg = rows["bullish"]["avg_forward_pct"]
        assert b_avg is not None
        assert isinstance(b_avg, float)
        assert 2.9 < b_avg < 3.0
        # Bearish: forward_pct stored as (engulf - forward)/engulf for bearish,
        # so 99 -> 102 → (99-102)/99 ≈ -3.03%.
        be_avg = rows["bearish"]["avg_forward_pct"]
        assert be_avg is not None
        assert isinstance(be_avg, float)
        assert -3.1 < be_avg < -3.0


class TestEngulfingEdgeCases:
    def test_unknown_timeframe_raises(self, fixture_bars: pl.DataFrame) -> None:
        params = ReportParams(
            symbol="TEST", lookback_days=500, recency_window_days=30, timeframe="7m"
        )
        with pytest.raises(ValueError, match="Unknown timeframe"):
            compute(fixture_bars, params)

    def test_engulfing_too_close_to_eod_excluded(self) -> None:
        """An engulfing at the end of the day with no N+K candle is dropped."""
        # Build one day where the bullish engulfing is at idx=23, N+3 would
        # be idx=26 but max is 24 (25-candle day) → dropped.
        candles: list[tuple[float, float, float, float]] = [
            (100.0, 100.0, 100.0, 100.0)
        ] * 22  # flat 0..21
        candles.append((100.0, 100.5, 99.5, 99.0))  # idx=22: red
        candles.append((98.5, 102.5, 98.0, 102.0))  # idx=23: bullish engulfing
        candles.append((102.0, 102.5, 101.5, 102.0))  # idx=24: only one candle ahead
        bars = _make_day_from_candles("2025-12-01", candles)
        bars2 = _make_day_from_candles("2025-12-02", candles)  # second day for lookback >= 2
        df = pl.DataFrame(bars + bars2)
        params = ReportParams(
            symbol="X",
            lookback_days=500,
            recency_window_days=30,
            timeframe="15m",
            lookahead_candles=3,
        )
        r = compute(df, params)
        # Engulfing exists but is dropped due to insufficient lookahead.
        assert r["summary"]["total_engulfings"] == 0

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
        params = ReportParams(symbol="X", lookback_days=500, timeframe="15m")
        r = compute(df, params)
        assert r["summary"]["total_engulfings"] == 0

    def test_summary_carries_timeframe(self, result: dict[str, object]) -> None:
        summary = result["summary"]
        assert isinstance(summary, dict)
        assert summary["timeframe"] == "15m"
        assert summary["lookahead_candles"] == 3


class TestNoSpuriousEngulfing:
    def test_inside_candle_is_not_engulfing(self) -> None:
        """Two candles where the second is strictly inside the first (smaller
        body) must NOT be reported as an engulfing."""
        candles: list[tuple[float, float, float, float]] = [
            (100.0, 105.0, 95.0, 96.0),  # idx=0: red, big body 100->96
            (97.0, 99.5, 96.5, 99.0),  # idx=1: green, small body 97->99 — INSIDE prev body
        ]
        # Pad to a full session.
        candles += [(99.0, 99.5, 98.5, 99.0)] * 23
        bars = _make_day_from_candles("2025-12-08", candles)
        bars2 = _make_day_from_candles("2025-12-09", candles)
        df = pl.DataFrame(bars + bars2)
        params = ReportParams(symbol="X", lookback_days=500, timeframe="15m", lookahead_candles=3)
        r = compute(df, params)
        # Inside body (97->99) does NOT engulf (100->96), so no engulfing.
        assert r["summary"]["total_engulfings"] == 0
