"""Performance metrics for a list of trades.

Returns are expressed in percent of entry price (``Trade.pnl_pct_net``).
Equity is built by compounding net per-trade returns starting from 1.0.

Annualisation uses 252 trading days for daily-aggregated metrics.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date

from backtest.engine import Trade

_TRADING_DAYS_PER_YEAR: int = 252


@dataclass(frozen=True)
class Metrics:
    total_trades: int
    wins: int
    losses: int
    win_rate: float  # 0..1
    avg_win_pct: float  # mean win, percent (positive)
    avg_loss_pct: float  # mean loss, percent (negative)
    profit_factor: float  # sum(wins) / abs(sum(losses)); inf if no losses
    total_return_pct: float  # cumulative compounded return in percent
    annualised_return_pct: float
    sharpe: float  # annualised, based on daily returns
    sortino: float  # annualised, based on daily returns and downside deviation
    max_drawdown_pct: float  # most negative peak-to-trough on the equity curve
    calmar: float  # annualised_return / |max_drawdown|; 0 if dd=0
    monthly_returns: dict[str, float]  # "YYYY-MM" -> percent


def compute(trades: list[Trade]) -> Metrics:
    """Compute the full metrics block from a chronological trade list."""
    if not trades:
        return _empty()

    sorted_trades = sorted(trades, key=lambda t: t.entry_time)
    nets = [t.pnl_pct_net for t in sorted_trades]
    n = len(nets)
    wins = [r for r in nets if r > 0]
    losses = [r for r in nets if r < 0]

    win_rate = len(wins) / n if n > 0 else 0.0
    avg_win = statistics.mean(wins) if wins else 0.0
    avg_loss = statistics.mean(losses) if losses else 0.0
    sum_wins = sum(wins)
    sum_losses = sum(losses)
    profit_factor = (sum_wins / abs(sum_losses)) if sum_losses < 0 else math.inf

    # Equity curve (compounded). Start at 1.0, multiply by (1 + r/100).
    equity = [1.0]
    for r in nets:
        equity.append(equity[-1] * (1.0 + r / 100.0))
    total_return_pct = (equity[-1] - 1.0) * 100.0

    # Annualised return over the trade-date span.
    first_date = sorted_trades[0].entry_time.date()
    last_date = sorted_trades[-1].entry_time.date()
    span_days = max((last_date - first_date).days, 1)
    years = span_days / 365.0
    if years > 0 and equity[-1] > 0:
        annualised_return_pct = (equity[-1] ** (1.0 / years) - 1.0) * 100.0
    else:
        annualised_return_pct = 0.0

    # Daily returns for Sharpe / Sortino: aggregate trade returns per date
    # (compounded), then take simple mean / stdev across days.
    by_day: dict[date, list[float]] = {}
    for t in sorted_trades:
        by_day.setdefault(t.entry_time.date(), []).append(t.pnl_pct_net)
    daily_returns = []
    for _td, day_returns in sorted(by_day.items()):
        product = 1.0
        for r in day_returns:
            product *= 1.0 + r / 100.0
        daily_returns.append((product - 1.0) * 100.0)

    if len(daily_returns) > 1:
        d_mean = statistics.mean(daily_returns)
        d_std = statistics.stdev(daily_returns)
        sharpe = (d_mean / d_std) * math.sqrt(_TRADING_DAYS_PER_YEAR) if d_std > 0 else 0.0
        downside = [r for r in daily_returns if r < 0]
        if len(downside) > 1:
            d_down_std = statistics.stdev(downside)
            sortino = (
                (d_mean / d_down_std) * math.sqrt(_TRADING_DAYS_PER_YEAR) if d_down_std > 0 else 0.0
            )
        else:
            sortino = 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    # Max drawdown from the equity curve (peak-to-trough).
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100.0
        if dd < max_dd:
            max_dd = dd
    calmar = annualised_return_pct / abs(max_dd) if max_dd < 0 else 0.0

    # Monthly returns.
    by_month: dict[str, float] = {}
    month_eq: dict[str, float] = {}
    for t in sorted_trades:
        key = t.entry_time.strftime("%Y-%m")
        if key not in month_eq:
            month_eq[key] = 1.0
        month_eq[key] *= 1.0 + t.pnl_pct_net / 100.0
    for key, eq in month_eq.items():
        by_month[key] = round((eq - 1.0) * 100.0, 4)

    return Metrics(
        total_trades=n,
        wins=len(wins),
        losses=len(losses),
        win_rate=round(win_rate, 4),
        avg_win_pct=round(avg_win, 4),
        avg_loss_pct=round(avg_loss, 4),
        profit_factor=round(profit_factor, 4) if profit_factor != math.inf else math.inf,
        total_return_pct=round(total_return_pct, 4),
        annualised_return_pct=round(annualised_return_pct, 4),
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4),
        max_drawdown_pct=round(max_dd, 4),
        calmar=round(calmar, 4),
        monthly_returns=by_month,
    )


def _empty() -> Metrics:
    return Metrics(
        total_trades=0,
        wins=0,
        losses=0,
        win_rate=0.0,
        avg_win_pct=0.0,
        avg_loss_pct=0.0,
        profit_factor=0.0,
        total_return_pct=0.0,
        annualised_return_pct=0.0,
        sharpe=0.0,
        sortino=0.0,
        max_drawdown_pct=0.0,
        calmar=0.0,
        monthly_returns={},
    )
