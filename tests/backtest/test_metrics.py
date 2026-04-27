"""Hand-checked tests for the metrics module."""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from backtest.engine import Trade
from backtest.metrics import compute


def _trade(date: str, gross: float, net: float | None = None) -> Trade:
    """Convenience factory for a Trade with controllable returns."""
    if net is None:
        net = gross
    entry = datetime.fromisoformat(f"{date}T09:15:00")
    exit_ = datetime.fromisoformat(f"{date}T15:25:00")
    return Trade(
        symbol="TEST",
        direction="long",
        entry_time=entry,
        entry_price=100.0,
        exit_time=exit_,
        exit_price=100.0 * (1.0 + net / 100.0),
        exit_reason="eod",
        pnl_pct_gross=gross,
        pnl_pct_net=net,
    )


class TestEmptyAndTrivial:
    def test_empty_trade_list(self) -> None:
        m = compute([])
        assert m.total_trades == 0
        assert m.win_rate == 0.0
        assert m.sharpe == 0.0
        assert m.monthly_returns == {}

    def test_single_winning_trade(self) -> None:
        m = compute([_trade("2025-01-15", 1.0)])
        assert m.total_trades == 1
        assert m.wins == 1
        assert m.losses == 0
        assert m.win_rate == 1.0
        assert m.profit_factor == math.inf
        assert m.total_return_pct == pytest.approx(1.0)


class TestCounts:
    def _trades(self) -> list[Trade]:
        # Three wins (1%, 2%, 0.5%) and two losses (-1%, -0.5%) on five
        # different dates so daily-aggregation is one return per day.
        return [
            _trade("2025-01-02", 1.0),
            _trade("2025-01-03", -1.0),
            _trade("2025-01-06", 2.0),
            _trade("2025-01-07", -0.5),
            _trade("2025-01-08", 0.5),
        ]

    def test_total_and_win_rate(self) -> None:
        m = compute(self._trades())
        assert m.total_trades == 5
        assert m.wins == 3
        assert m.losses == 2
        assert m.win_rate == 0.6

    def test_avg_win_avg_loss(self) -> None:
        m = compute(self._trades())
        # Avg win = (1 + 2 + 0.5) / 3 = 1.1667
        assert m.avg_win_pct == pytest.approx(1.1667, abs=1e-3)
        # Avg loss = (-1 + -0.5) / 2 = -0.75
        assert m.avg_loss_pct == pytest.approx(-0.75, abs=1e-3)

    def test_profit_factor(self) -> None:
        m = compute(self._trades())
        # sum_wins = 3.5, sum_losses = -1.5, PF = 3.5/1.5 = 2.333
        assert m.profit_factor == pytest.approx(3.5 / 1.5, abs=1e-3)


class TestEquityAndReturns:
    def test_total_return_compounds_correctly(self) -> None:
        # Three +1% trades compound to (1.01)^3 - 1 = 3.0301%.
        trades = [
            _trade("2025-01-02", 1.0),
            _trade("2025-01-03", 1.0),
            _trade("2025-01-06", 1.0),
        ]
        m = compute(trades)
        expected = ((1.01**3) - 1.0) * 100.0
        assert m.total_return_pct == pytest.approx(expected, abs=1e-3)

    def test_loss_then_win_pulls_below_simple_sum(self) -> None:
        # +5% then -5% compounds to (1.05*0.95) - 1 = -0.25%, not 0%.
        m = compute(
            [
                _trade("2025-01-02", 5.0),
                _trade("2025-01-03", -5.0),
            ]
        )
        assert m.total_return_pct == pytest.approx(-0.25, abs=1e-3)


class TestDrawdown:
    def test_max_drawdown_after_loss_streak(self) -> None:
        # +5% then two -5% trades: equity 1.0 -> 1.05 -> 0.9975 -> 0.9476.
        # Peak 1.05; trough 0.9476 -> dd = (0.9476 - 1.05)/1.05 ≈ -9.76%.
        trades = [
            _trade("2025-01-02", 5.0),
            _trade("2025-01-03", -5.0),
            _trade("2025-01-06", -5.0),
        ]
        m = compute(trades)
        # Compute expected DD manually.
        eq = [1.0, 1.05, 1.05 * 0.95, 1.05 * 0.95 * 0.95]
        peak = max(eq)
        trough = min(eq[eq.index(peak) :])
        expected_dd = (trough - peak) / peak * 100.0
        assert m.max_drawdown_pct == pytest.approx(expected_dd, abs=1e-3)

    def test_no_drawdown_on_monotonic_wins(self) -> None:
        m = compute(
            [
                _trade("2025-01-02", 1.0),
                _trade("2025-01-03", 1.0),
                _trade("2025-01-06", 1.0),
            ]
        )
        assert m.max_drawdown_pct == 0.0


class TestMonthlyReturns:
    def test_monthly_buckets(self) -> None:
        m = compute(
            [
                _trade("2025-01-15", 1.0),
                _trade("2025-01-22", -0.5),
                _trade("2025-02-04", 2.0),
            ]
        )
        # Jan: 1.01 * 0.995 - 1 = 0.495%
        assert m.monthly_returns["2025-01"] == pytest.approx((1.01 * 0.995 - 1.0) * 100.0, abs=1e-3)
        assert m.monthly_returns["2025-02"] == pytest.approx(2.0, abs=1e-3)


class TestSharpeAndSortino:
    def test_sharpe_positive_for_winning_streak(self) -> None:
        # Varied positive returns -> non-zero stdev, positive mean,
        # high Sharpe.  Daily values: 1.0, 1.5, 0.8, 1.2, 0.9, 1.1, 1.3, 0.7,
        # 1.0, 1.4 (mean ~1.09, stdev small).
        returns = [1.0, 1.5, 0.8, 1.2, 0.9, 1.1, 1.3, 0.7, 1.0, 1.4]
        trades = [_trade(f"2025-01-{i:02d}", r) for i, r in zip(range(2, 12), returns, strict=True)]
        m = compute(trades)
        assert m.sharpe > 5.0  # mean~1.09, low stdev, annualised by sqrt(252)

    def test_sharpe_zero_when_all_same_return(self) -> None:
        # Identical daily returns -> stdev 0 -> Sharpe defined as 0.
        trades = [_trade(f"2025-01-{i:02d}", 1.0) for i in range(2, 12)]
        m = compute(trades)
        assert m.sharpe == 0.0

    def test_sortino_uses_only_downside(self) -> None:
        # Mix of varied wins and varied losses so both total stdev and
        # downside stdev are non-zero.  Sortino should be positive and
        # finite; we check the ordering against Sharpe with this asymmetric
        # series where wins are bigger than losses.
        trades = [
            _trade("2025-01-02", 3.0),
            _trade("2025-01-03", -0.5),
            _trade("2025-01-06", 2.5),
            _trade("2025-01-07", -1.0),
            _trade("2025-01-08", 3.5),
            _trade("2025-01-09", -0.7),
        ]
        m = compute(trades)
        assert m.sortino > 0
        # Total stdev includes both upside (3.0, 2.5, 3.5) and downside
        # (-0.5, -1.0, -0.7).  Downside-only stdev is much smaller, so
        # sortino > sharpe.
        assert m.sortino > m.sharpe
