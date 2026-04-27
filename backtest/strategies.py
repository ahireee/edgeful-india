"""Reference strategies for the backtest engine.

Each strategy is a pure function: ``(today_bars, ctx) -> Signal | None``.
None means "no trade today".  All entry-bar detection lives in the strategy;
the engine only handles position management once a Signal is returned.
"""

from __future__ import annotations

from datetime import time

import polars as pl

from backtest.engine import Signal, StrategyContext

# Wait until this time before allowing PDH/PDL or ORB-style entries to fire.
_AFTER_OPEN_TIME: time = time(9, 30)
_OR_MINUTES: int = 15


def pdh_breakout(today_bars: pl.DataFrame, ctx: StrategyContext) -> Signal | None:
    """Long when price breaks PDH after 09:30 IST.

    Entry: PDH (the breakout level).
    Stop: PDL.
    Target: PDH + 1.5 * (PDH - PDL) (1.5R).
    EOD exit handled by the engine (default 15:25).

    Days where today's first bar already opens above PDH are still eligible
    if PDH gets touched again post-09:30; otherwise no breakout fires.
    """
    eligible = today_bars.filter(pl.col("ts_ist").dt.time() >= _AFTER_OPEN_TIME)
    breakouts = eligible.filter(pl.col("high") > ctx.prev_high)
    if breakouts.height == 0:
        return None

    entry_ts = breakouts.get_column("ts_ist")[0]
    entry = ctx.prev_high
    stop = ctx.prev_low
    if stop >= entry:  # malformed prev day; skip
        return None
    risk = entry - stop
    target = entry + 1.5 * risk

    return Signal(
        direction="long",
        entry_time=entry_ts,
        entry_price=entry,
        stop=stop,
        target=target,
    )


def gap_fill_fade(today_bars: pl.DataFrame, ctx: StrategyContext) -> Signal | None:
    """Long the open on a 0.10%-0.25% down gap, target = prev_close (the fill).

    No stop (we model with a far-away level to keep the engine simple).
    Exit is either ``target`` (price touches prev_close) or EOD.
    """
    if ctx.prev_close <= 0:
        return None
    gap_pct = (ctx.today_open - ctx.prev_close) / ctx.prev_close * 100.0
    if not (-0.25 <= gap_pct <= -0.10):
        return None

    entry = ctx.today_open
    target = ctx.prev_close  # fill the gap
    # No real stop; place far below entry so it never fires.
    far_stop = entry * 0.5

    return Signal(
        direction="long",
        entry_time=ctx.today_open_ts,
        entry_price=entry,
        stop=far_stop,
        target=target,
    )


def orb_continuation(today_bars: pl.DataFrame, ctx: StrategyContext) -> Signal | None:
    """Trade the first break of the 15-min opening range.

    Long if price breaks OR-high first; short if OR-low breaks first.
    Stop = the opposite side of the OR.  Target = entry +/- 1.5 * range.
    Days where both sides break in the same 1-min bar are skipped (matches
    the ORB report's exclusion rule).
    """
    if today_bars.height < _OR_MINUTES:
        return None

    or_bars = today_bars.head(_OR_MINUTES)
    or_high = float(or_bars.get_column("high").max())  # type: ignore[arg-type]
    or_low = float(or_bars.get_column("low").min())  # type: ignore[arg-type]
    if or_high <= or_low:
        return None

    post = today_bars.slice(_OR_MINUTES, today_bars.height - _OR_MINUTES)
    up_breaks = post.filter(pl.col("high") > or_high)
    down_breaks = post.filter(pl.col("low") < or_low)
    up_ts = up_breaks.get_column("ts_ist")[0] if up_breaks.height > 0 else None
    down_ts = down_breaks.get_column("ts_ist")[0] if down_breaks.height > 0 else None

    if up_ts is None and down_ts is None:
        return None
    if up_ts is not None and down_ts is not None and up_ts == down_ts:
        return None  # both sides break same minute -- ambiguous

    range_size = or_high - or_low
    if down_ts is None or (up_ts is not None and up_ts < down_ts):
        return Signal(
            direction="long",
            entry_time=up_ts,  # type: ignore[arg-type]
            entry_price=or_high,
            stop=or_low,
            target=or_high + 1.5 * range_size,
        )
    return Signal(
        direction="short",
        entry_time=down_ts,
        entry_price=or_low,
        stop=or_high,
        target=or_low - 1.5 * range_size,
    )


# Registry so the CLI can resolve a strategy by name.
STRATEGIES: dict[str, object] = {
    "pdh_breakout": pdh_breakout,
    "gap_fill_fade": gap_fill_fade,
    "orb_continuation": orb_continuation,
}
