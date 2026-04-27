"""Transaction-cost model for the backtest engine.

Returns are expressed in percent of entry price.  The cost model converts
brokerage / STT / slippage into a single round-trip percentage that the
engine subtracts from the gross trade return.

Defaults reflect typical Indian discount-broker pricing for intraday equity:
  - STT 0.0125% on the sell side
  - Brokerage Rs.20 flat per executed order (Rs.40 round-trip)
  - Slippage 1 tick (0.05) for index-based instruments, 0.05% for stocks

"Notional per trade" lets us amortise the flat-fee brokerage into a
percentage; otherwise a Rs.40 round-trip on a single share would dominate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """All-in transaction cost model.  Per-trade percentages are computed
    against ``entry_price``; flat brokerage is amortised against
    ``notional_per_trade``."""

    is_index: bool = False
    tick_size: float = 0.05
    slippage_ticks: int = 1  # only used when ``is_index`` is True
    slippage_pct_stock: float = 0.05  # 0.05% per side, used when ``is_index`` is False
    stt_sell_pct: float = 0.0125  # 0.0125% on sell-side notional (intraday equity)
    brokerage_per_side: float = 20.0  # rupees, flat per order
    notional_per_trade: float = 100_000.0  # rupees, used to amortise flat brokerage

    def round_trip_cost_pct(self, entry_price: float) -> float:
        """Return total round-trip cost as a percentage of entry price.

        Components:
          - 2x slippage (entry + exit)
          - STT on sell side (approximated against entry price for symmetry)
          - 2x brokerage (entry + exit), amortised against notional_per_trade
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        if self.is_index:
            slip_per_side_pct = (self.tick_size * self.slippage_ticks / entry_price) * 100.0
        else:
            slip_per_side_pct = self.slippage_pct_stock

        slip_total = 2.0 * slip_per_side_pct
        stt = self.stt_sell_pct
        brokerage = (2.0 * self.brokerage_per_side / self.notional_per_trade) * 100.0
        return slip_total + stt + brokerage


def for_symbol(symbol: str, *, is_index: bool, **overrides: float) -> CostModel:
    """Build a CostModel sensibly defaulted for the given symbol.

    The ``is_index`` flag selects fixed-tick slippage vs. percent slippage.
    Any other field can be overridden via keyword arguments."""
    _ = symbol  # symbol kept for future per-instrument tuning (lot size, etc.)
    return CostModel(is_index=is_index, **overrides)  # type: ignore[arg-type]
