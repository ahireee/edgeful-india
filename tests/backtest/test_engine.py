"""Hand-checked tests for the backtest engine.

The engine is independent of any particular strategy, so we hand-craft
``Signal`` values directly and verify entry / stop / target / EOD exits.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

import polars as pl
import pytest

from backtest.costs import CostModel
from backtest.engine import Signal, Trade, simulate


def _make_day(
    date_str: str,
    bars_spec: list[tuple[float, float, float, float]],
    *,
    pad_to: int = 375,
) -> list[dict[str, object]]:
    """Generate 1-min bars for one day from explicit (open, high, low, close)
    tuples for the first ``len(bars_spec)`` bars.  Remaining bars are flat
    at the previous close so the day clears the 300-bar Muhurat threshold."""
    base = datetime.fromisoformat(f"{date_str}T09:15:00")
    rows: list[dict[str, object]] = []
    last_close = bars_spec[0][3] if bars_spec else 100.0
    for i in range(pad_to):
        ts = base + timedelta(minutes=i)
        if i < len(bars_spec):
            o, h, lo, c = bars_spec[i]
        else:
            o = h = lo = c = last_close
        last_close = c
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


def _const_strategy(signal: Signal | None) -> object:
    """Return a strategy that always emits the given signal (or None)."""

    def strat(today_bars: pl.DataFrame, ctx: object) -> Signal | None:
        return signal

    return strat


_NO_COST = CostModel(
    is_index=False,
    slippage_pct_stock=0.0,
    stt_sell_pct=0.0,
    brokerage_per_side=0.0,
)


class TestEngineExits:
    def _two_day_frame(self, day2_spec: list[tuple[float, float, float, float]]) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        rows.extend(_make_day("2025-04-01", [(100.0, 100.0, 100.0, 100.0)]))
        rows.extend(_make_day("2025-04-02", day2_spec))
        return pl.DataFrame(rows)

    def test_long_target_hit(self) -> None:
        # Day 2 bar 0: range 100..101 hits target at 101.
        df = self._two_day_frame([(100.0, 101.0, 99.5, 100.5)])
        entry_ts = datetime(2025, 4, 2, 9, 15)
        sig = Signal("long", entry_ts, 100.0, stop=99.0, target=101.0)
        trades = simulate(df, _const_strategy(sig), "TEST", _NO_COST)  # type: ignore[arg-type]
        assert len(trades) == 1
        assert trades[0].exit_reason == "target"
        assert trades[0].exit_price == 101.0
        assert trades[0].pnl_pct_gross == pytest.approx(1.0)

    def test_long_stop_hit(self) -> None:
        # Day 2 bar 0: range 98..100.5 dips to 98 -> stop at 99.
        df = self._two_day_frame([(100.0, 100.5, 98.0, 99.5)])
        entry_ts = datetime(2025, 4, 2, 9, 15)
        sig = Signal("long", entry_ts, 100.0, stop=99.0, target=105.0)
        trades = simulate(df, _const_strategy(sig), "TEST", _NO_COST)  # type: ignore[arg-type]
        assert len(trades) == 1
        assert trades[0].exit_reason == "stop"
        assert trades[0].exit_price == 99.0
        assert trades[0].pnl_pct_gross == pytest.approx(-1.0)

    def test_long_eod_exit(self) -> None:
        # Bars never hit stop/target; EOD bar (15:25) closes at 100.5.
        spec = [(100.0, 100.4, 99.6, 100.0)] * 374
        # Bar at index 370 = 09:15 + 370 min = 15:25 IST. Make it close 100.5.
        spec[370] = (100.0, 100.5, 100.0, 100.5)
        df = self._two_day_frame(spec)
        entry_ts = datetime(2025, 4, 2, 9, 15)
        sig = Signal("long", entry_ts, 100.0, stop=95.0, target=105.0)
        trades = simulate(df, _const_strategy(sig), "TEST", _NO_COST)  # type: ignore[arg-type]
        assert len(trades) == 1
        assert trades[0].exit_reason == "eod"
        assert trades[0].exit_price == pytest.approx(100.5)
        assert trades[0].pnl_pct_gross == pytest.approx(0.5)

    def test_short_target_hit(self) -> None:
        df = self._two_day_frame([(100.0, 100.5, 99.0, 99.0)])
        entry_ts = datetime(2025, 4, 2, 9, 15)
        sig = Signal("short", entry_ts, 100.0, stop=101.0, target=99.0)
        trades = simulate(df, _const_strategy(sig), "TEST", _NO_COST)  # type: ignore[arg-type]
        assert len(trades) == 1
        assert trades[0].exit_reason == "target"
        assert trades[0].pnl_pct_gross == pytest.approx(1.0)

    def test_short_stop_hit(self) -> None:
        df = self._two_day_frame([(100.0, 101.5, 99.0, 100.0)])
        entry_ts = datetime(2025, 4, 2, 9, 15)
        sig = Signal("short", entry_ts, 100.0, stop=101.0, target=99.0)
        trades = simulate(df, _const_strategy(sig), "TEST", _NO_COST)  # type: ignore[arg-type]
        assert len(trades) == 1
        assert trades[0].exit_reason == "stop"
        assert trades[0].pnl_pct_gross == pytest.approx(-1.0)

    def test_both_side_bar_stop_wins(self) -> None:
        """When a single bar's range crosses both stop and target, the engine
        treats stop as having fired first (worst case for the strategy)."""
        df = self._two_day_frame([(100.0, 102.0, 98.0, 100.0)])
        entry_ts = datetime(2025, 4, 2, 9, 15)
        sig = Signal("long", entry_ts, 100.0, stop=99.0, target=101.0)
        trades = simulate(df, _const_strategy(sig), "TEST", _NO_COST)  # type: ignore[arg-type]
        assert len(trades) == 1
        assert trades[0].exit_reason == "stop"


class TestEngineCostApplication:
    def test_cost_subtracts_from_gross(self) -> None:
        rows: list[dict[str, object]] = []
        rows.extend(_make_day("2025-04-01", [(100.0, 100.0, 100.0, 100.0)]))
        rows.extend(_make_day("2025-04-02", [(100.0, 101.0, 99.5, 100.5)]))
        df = pl.DataFrame(rows)
        entry_ts = datetime(2025, 4, 2, 9, 15)
        sig = Signal("long", entry_ts, 100.0, stop=99.0, target=101.0)

        cm = CostModel(is_index=False)  # ~0.1525% round-trip
        trades = simulate(df, _const_strategy(sig), "TEST", cm)  # type: ignore[arg-type]
        assert len(trades) == 1
        t = trades[0]
        assert t.pnl_pct_gross == pytest.approx(1.0)
        # Net = gross - cost.
        expected_net = 1.0 - cm.round_trip_cost_pct(100.0)
        assert t.pnl_pct_net == pytest.approx(expected_net, rel=1e-6)
        assert t.pnl_pct_net < t.pnl_pct_gross


class TestEngineSkipsAndShortDays:
    def test_short_session_day_excluded(self) -> None:
        rows: list[dict[str, object]] = []
        rows.extend(_make_day("2025-04-01", [(100.0, 100.0, 100.0, 100.0)]))
        # Day 2 has only 60 bars -> skipped (Muhurat threshold = 300).
        rows.extend(_make_day("2025-04-02", [(100.0, 101.0, 99.0, 100.0)], pad_to=60))
        df = pl.DataFrame(rows)
        entry_ts = datetime(2025, 4, 2, 9, 15)
        sig = Signal("long", entry_ts, 100.0, stop=99.0, target=101.0)
        trades = simulate(df, _const_strategy(sig), "TEST", _NO_COST)  # type: ignore[arg-type]
        assert trades == []

    def test_no_signal_returns_no_trade(self) -> None:
        rows: list[dict[str, object]] = []
        rows.extend(_make_day("2025-04-01", [(100.0, 100.0, 100.0, 100.0)]))
        rows.extend(_make_day("2025-04-02", [(100.0, 101.0, 99.5, 100.5)]))
        df = pl.DataFrame(rows)
        trades = simulate(df, _const_strategy(None), "TEST", _NO_COST)  # type: ignore[arg-type]
        assert trades == []

    def test_first_day_skipped_no_prior(self) -> None:
        """A single-day dataset has no prior day -> no trades emitted."""
        rows = _make_day("2025-04-01", [(100.0, 101.0, 99.0, 100.0)])
        df = pl.DataFrame(rows)
        entry_ts = datetime(2025, 4, 1, 9, 15)
        sig = Signal("long", entry_ts, 100.0, stop=99.0, target=101.0)
        trades = simulate(df, _const_strategy(sig), "TEST", _NO_COST)  # type: ignore[arg-type]
        assert trades == []

    def test_eod_time_configurable(self) -> None:
        """Custom EOD time is honoured."""
        # Build a day where price drifts up and never hits stop/target.
        spec = [(100.0, 100.4, 99.8, 100.0)] * 30
        spec[10] = (100.0, 100.0, 100.0, 100.2)  # 09:25 close = 100.2
        rows: list[dict[str, object]] = []
        rows.extend(_make_day("2025-04-01", [(100.0, 100.0, 100.0, 100.0)]))
        rows.extend(_make_day("2025-04-02", spec))
        df = pl.DataFrame(rows)
        entry_ts = datetime(2025, 4, 2, 9, 15)
        sig = Signal("long", entry_ts, 100.0, stop=95.0, target=105.0)
        trades = simulate(
            df,
            _const_strategy(sig),  # type: ignore[arg-type]
            "TEST",
            _NO_COST,
            eod_time=time(9, 25),
        )
        assert len(trades) == 1
        assert trades[0].exit_reason == "eod"
        assert trades[0].exit_price == pytest.approx(100.2)


class TestTradeRecordShape:
    def test_records_all_fields(self) -> None:
        rows: list[dict[str, object]] = []
        rows.extend(_make_day("2025-04-01", [(100.0, 100.0, 100.0, 100.0)]))
        rows.extend(_make_day("2025-04-02", [(100.0, 101.0, 99.5, 100.5)]))
        df = pl.DataFrame(rows)
        entry_ts = datetime(2025, 4, 2, 9, 15)
        sig = Signal("long", entry_ts, 100.0, stop=99.0, target=101.0)
        trades = simulate(df, _const_strategy(sig), "TEST", _NO_COST)  # type: ignore[arg-type]
        t = trades[0]
        assert isinstance(t, Trade)
        assert t.symbol == "TEST"
        assert t.direction == "long"
        assert t.entry_time == entry_ts
        assert t.entry_price == 100.0
        assert t.exit_reason == "target"
