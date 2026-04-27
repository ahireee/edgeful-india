"""Bar-replay backtest engine.

Pure function: takes 1-min bars + a strategy function + a cost model,
returns the list of trades.  No I/O; no global state.

Per-day flow:
  1. Build a StrategyContext from the previous day's stats.
  2. Hand today's bars + context to the strategy. It returns a Signal or None.
  3. Walk forward from the entry timestamp managing stop / target / EOD exit.
  4. Apply the cost model and append a Trade record.

Both the entry bar and subsequent bars are checked for stop/target hits.
When a single bar's range crosses both stop and target, we conservatively
treat the stop as having fired first (worst-case for the strategy).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time

import polars as pl

from backtest.costs import CostModel

_MIN_SESSION_BARS: int = 300
DEFAULT_EOD: time = time(15, 25)


@dataclass(frozen=True)
class StrategyContext:
    """Pre-computed inputs handed to a strategy for one trading day."""

    symbol: str
    trade_date: date
    prev_open: float
    prev_high: float
    prev_low: float
    prev_close: float
    today_open: float
    today_open_ts: datetime


@dataclass(frozen=True)
class Signal:
    """An entry order returned by a strategy."""

    direction: str  # "long" | "short"
    entry_time: datetime  # timestamp of the bar at which entry triggers
    entry_price: float
    stop: float
    target: float


@dataclass(frozen=True)
class Trade:
    """A completed round-trip with cost-adjusted return in percent."""

    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str  # "stop" | "target" | "eod"
    pnl_pct_gross: float
    pnl_pct_net: float


StrategyFn = Callable[[pl.DataFrame, StrategyContext], Signal | None]


def simulate(
    bars: pl.DataFrame,
    strategy: StrategyFn,
    symbol: str,
    cost_model: CostModel,
    *,
    eod_time: time = DEFAULT_EOD,
) -> list[Trade]:
    """Replay ``bars`` through ``strategy`` and return the resulting trades.

    Parameters
    ----------
    bars : pl.DataFrame
        1-min bars with columns: symbol, ts_ist, open, high, low, close, volume.
        Should already be filtered to the requested ``symbol``.
    strategy : StrategyFn
        A pure function that, given today's bars + context, returns a Signal
        or None.  No mutation expected.
    symbol : str
        Symbol label, copied onto every Trade record.
    cost_model : CostModel
        Round-trip cost in percent gets subtracted from every gross trade.
    eod_time : datetime.time
        Force-close any open position at this time (default 15:25 IST).
    """
    if bars.height == 0:
        return []

    bars = bars.sort("ts_ist").with_columns(pl.col("ts_ist").cast(pl.Date).alias("trade_date"))

    daily_counts = (
        bars.group_by("trade_date")
        .agg(pl.len().alias("bar_count"))
        .filter(pl.col("bar_count") >= _MIN_SESSION_BARS)
        .sort("trade_date")
    )
    valid_dates: list[date] = daily_counts.get_column("trade_date").to_list()
    if len(valid_dates) < 2:
        return []

    trades: list[Trade] = []
    for i in range(1, len(valid_dates)):
        td = valid_dates[i]
        prev_td = valid_dates[i - 1]

        prev_bars = bars.filter(pl.col("trade_date") == prev_td).sort("ts_ist")
        today_bars = bars.filter(pl.col("trade_date") == td).sort("ts_ist")
        if today_bars.height == 0 or prev_bars.height == 0:
            continue

        ctx = StrategyContext(
            symbol=symbol,
            trade_date=td,
            prev_open=float(prev_bars.get_column("open")[0]),
            prev_high=float(prev_bars.get_column("high").max()),  # type: ignore[arg-type]
            prev_low=float(prev_bars.get_column("low").min()),  # type: ignore[arg-type]
            prev_close=float(prev_bars.get_column("close")[-1]),
            today_open=float(today_bars.get_column("open")[0]),
            today_open_ts=today_bars.get_column("ts_ist")[0],
        )

        signal = strategy(today_bars, ctx)
        if signal is None:
            continue

        trade = _execute(today_bars, signal, symbol, cost_model, eod_time)
        if trade is not None:
            trades.append(trade)

    return trades


def _execute(
    today_bars: pl.DataFrame,
    signal: Signal,
    symbol: str,
    cost_model: CostModel,
    eod_time: time,
) -> Trade | None:
    """Walk bars forward from signal.entry_time, exiting on stop / target / EOD."""
    post = today_bars.filter(pl.col("ts_ist") >= signal.entry_time)
    if post.height == 0:
        return None

    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None

    for row in post.iter_rows(named=True):
        ts = row["ts_ist"]
        h = float(row["high"])
        lo = float(row["low"])
        c = float(row["close"])

        if signal.direction == "long":
            # Stop checked before target (worst-case on a both-side bar).
            if lo <= signal.stop:
                exit_time, exit_price, exit_reason = ts, signal.stop, "stop"
                break
            if h >= signal.target:
                exit_time, exit_price, exit_reason = ts, signal.target, "target"
                break
        else:  # short
            if h >= signal.stop:
                exit_time, exit_price, exit_reason = ts, signal.stop, "stop"
                break
            if lo <= signal.target:
                exit_time, exit_price, exit_reason = ts, signal.target, "target"
                break

        if ts.time() >= eod_time:
            exit_time, exit_price, exit_reason = ts, c, "eod"
            break

    if exit_time is None or exit_price is None or exit_reason is None:
        # Day ended before EOD bar (synthetic / partial session) → use last bar's close.
        last = post.tail(1).row(0, named=True)
        exit_time = last["ts_ist"]
        exit_price = float(last["close"])
        exit_reason = "eod"

    if signal.direction == "long":
        pnl_pct_gross = (exit_price - signal.entry_price) / signal.entry_price * 100.0
    else:
        pnl_pct_gross = (signal.entry_price - exit_price) / signal.entry_price * 100.0

    cost_pct = cost_model.round_trip_cost_pct(signal.entry_price)
    pnl_pct_net = pnl_pct_gross - cost_pct

    return Trade(
        symbol=symbol,
        direction=signal.direction,
        entry_time=signal.entry_time,
        entry_price=signal.entry_price,
        exit_time=exit_time,
        exit_price=exit_price,
        exit_reason=exit_reason,
        pnl_pct_gross=round(pnl_pct_gross, 6),
        pnl_pct_net=round(pnl_pct_net, 6),
    )
