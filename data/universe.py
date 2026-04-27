"""
Trading universe for edgeful-india.

12 instruments: NIFTY 50 + Bank NIFTY indices, plus the top 10 NIFTY 50 stocks
by index weight. Weights as of 2025 — verify against the current NSE NIFTY 50
factsheet before locking for production. The top 10 typically account for
~55-60% of NIFTY weight.

Upstox uses an "instrument key" format: SEGMENT|ISIN. ISINs are India's
permanent security identifiers and never change for a given company, so these
strings are stable across time. Indices use a special key format like
"NSE_INDEX|Nifty 50".

Why we track the index, not the future:
- For probability reports (gap fill, ORB, PDH/PDL) the index level is what
  matters. Backtests using futures for actual P&L can resolve the active
  contract at runtime via the Expired Instruments API.
- This avoids the messy contract-rollover logic in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Segment(StrEnum):
    NSE_EQ = "NSE_EQ"  # NSE cash equity
    NSE_INDEX = "NSE_INDEX"  # NSE index (NIFTY, BANKNIFTY)
    NFO_FUT = "NFO_FUT"  # NSE F&O futures (resolved at runtime)


@dataclass(frozen=True)
class Instrument:
    symbol: str  # human-readable, e.g. "NIFTY", "RELIANCE"
    segment: Segment
    instrument_key: str  # Upstox format: "SEGMENT|ISIN" or "NSE_INDEX|Name"
    description: str

    @property
    def is_index(self) -> bool:
        return self.segment == Segment.NSE_INDEX


# ISINs sourced from NSE/BSE; these are permanent identifiers.
# If a stock is replaced in NIFTY 50, update this list and rerun backfill for
# the new ticker.
UNIVERSE: list[Instrument] = [
    Instrument("NIFTY", Segment.NSE_INDEX, "NSE_INDEX|Nifty 50", "NIFTY 50 index"),
    Instrument("BANKNIFTY", Segment.NSE_INDEX, "NSE_INDEX|Nifty Bank", "Bank NIFTY index"),
    Instrument("RELIANCE", Segment.NSE_EQ, "NSE_EQ|INE002A01018", "Reliance Industries"),
    Instrument("HDFCBANK", Segment.NSE_EQ, "NSE_EQ|INE040A01034", "HDFC Bank"),
    Instrument("ICICIBANK", Segment.NSE_EQ, "NSE_EQ|INE090A01021", "ICICI Bank"),
    Instrument("INFY", Segment.NSE_EQ, "NSE_EQ|INE009A01021", "Infosys"),
    Instrument("TCS", Segment.NSE_EQ, "NSE_EQ|INE467B01029", "Tata Consultancy Services"),
    Instrument("BHARTIARTL", Segment.NSE_EQ, "NSE_EQ|INE397D01024", "Bharti Airtel"),
    Instrument("ITC", Segment.NSE_EQ, "NSE_EQ|INE154A01025", "ITC Ltd"),
    Instrument("LT", Segment.NSE_EQ, "NSE_EQ|INE018A01030", "Larsen & Toubro"),
    Instrument("AXISBANK", Segment.NSE_EQ, "NSE_EQ|INE238A01034", "Axis Bank"),
    Instrument("KOTAKBANK", Segment.NSE_EQ, "NSE_EQ|INE237A01036", "Kotak Mahindra Bank"),
]


def by_symbol(symbol: str) -> Instrument:
    """Look up an instrument by its human symbol. Raises KeyError if unknown."""
    for inst in UNIVERSE:
        if inst.symbol == symbol:
            return inst
    raise KeyError(f"Unknown symbol: {symbol}. Known: {[i.symbol for i in UNIVERSE]}")


def by_instrument_key(key: str) -> Instrument:
    """Look up an instrument by its Upstox instrument_key."""
    for inst in UNIVERSE:
        if inst.instrument_key == key:
            return inst
    raise KeyError(f"Unknown instrument_key: {key}")
