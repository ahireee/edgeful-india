"""Hand-checked tests for the transaction-cost model."""

from __future__ import annotations

import pytest

from backtest.costs import CostModel, for_symbol


class TestIndexCostModel:
    def test_round_trip_breakdown_for_nifty_like_price(self) -> None:
        # Index instrument at price 22000.  Tick = 0.05.
        # Slippage per side = 0.05 / 22000 * 100 = 0.0002273%
        # Both sides            = 0.0004545%
        # STT (sell side)       = 0.0125%
        # Brokerage 2 * 20 / 100000 * 100 = 0.04%
        # Total                ≈ 0.0529%
        cm = CostModel(is_index=True)
        cost = cm.round_trip_cost_pct(22000.0)
        # Component-level checks.
        slip_both = (cm.tick_size * cm.slippage_ticks / 22000.0) * 100.0 * 2
        broker = (2 * cm.brokerage_per_side / cm.notional_per_trade) * 100.0
        expected = slip_both + cm.stt_sell_pct + broker
        assert cost == pytest.approx(expected, rel=1e-9)
        assert 0.05 < cost < 0.06

    def test_index_cost_decreases_with_price(self) -> None:
        """At higher price, fixed-tick slippage shrinks in percent terms."""
        cm = CostModel(is_index=True)
        c_low = cm.round_trip_cost_pct(10000.0)
        c_high = cm.round_trip_cost_pct(50000.0)
        assert c_high < c_low


class TestStockCostModel:
    def test_default_components(self) -> None:
        # Stock at price 3000.  Slippage 0.05% per side -> 0.10% both sides.
        # STT 0.0125%, brokerage 0.04% (default notional).  Total 0.1525%.
        cm = CostModel(is_index=False)
        cost = cm.round_trip_cost_pct(3000.0)
        expected = 0.10 + 0.0125 + 0.04
        assert cost == pytest.approx(expected, rel=1e-9)


class TestCostModelOverrides:
    def test_brokerage_override(self) -> None:
        cm = CostModel(is_index=False, brokerage_per_side=0.0)
        cm_default = CostModel(is_index=False)
        # Zero brokerage is exactly 0.04% lower than the default model.
        assert cm_default.round_trip_cost_pct(1000.0) - cm.round_trip_cost_pct(1000.0) == (
            pytest.approx(0.04, rel=1e-9)
        )

    def test_invalid_entry_price_raises(self) -> None:
        cm = CostModel(is_index=True)
        with pytest.raises(ValueError):
            cm.round_trip_cost_pct(0.0)
        with pytest.raises(ValueError):
            cm.round_trip_cost_pct(-100.0)


class TestForSymbolFactory:
    def test_index_factory(self) -> None:
        cm = for_symbol("NIFTY", is_index=True)
        assert cm.is_index is True

    def test_stock_factory_with_override(self) -> None:
        cm = for_symbol("RELIANCE", is_index=False, brokerage_per_side=10.0)
        assert cm.is_index is False
        assert cm.brokerage_per_side == 10.0
