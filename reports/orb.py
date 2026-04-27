"""Opening Range Breakout (ORB) report.

Computes breakout rate, continuation rate, and false-break rate for a
configurable opening-range window.  Pure function on a Polars DataFrame.

See docs/REPORTS.md section 2 for the full methodology.
"""

from __future__ import annotations

from datetime import time, timedelta

import polars as pl

from reports.base import ReportParams, ReportResult, wilson_ci

# Minimum bars for a valid session (exclude Muhurat / short days).
_MIN_SESSION_BARS: int = 300

# Market open time.
_MARKET_OPEN: time = time(9, 15)


def compute(bars: pl.DataFrame, params: ReportParams) -> ReportResult:
    """Compute the ORB probability table.

    Parameters
    ----------
    bars : pl.DataFrame
        1-min bars with columns: symbol, ts_ist, open, high, low, close, volume.
        Should already be filtered to the requested symbol.
    params : ReportParams
        Controls lookback, recency window, and ``or_minutes`` (opening range
        window in minutes, default 15).
    """
    or_minutes = params.or_minutes

    # ------------------------------------------------------------------
    # 1. Per-day aggregation: bar_count, day_close (last bar's close)
    # ------------------------------------------------------------------
    daily_meta = (
        bars.sort("ts_ist")
        .group_by(pl.col("ts_ist").cast(pl.Date).alias("trade_date"))
        .agg(
            pl.len().alias("bar_count"),
            pl.col("close").last().alias("day_close"),
        )
        .sort("trade_date")
        .filter(pl.col("bar_count") >= _MIN_SESSION_BARS)
        .tail(params.lookback_days)
    )

    if daily_meta.height == 0:
        return _empty_result(params)

    trade_dates = daily_meta.get_column("trade_date").to_list()
    day_closes = dict(
        zip(
            daily_meta.get_column("trade_date").to_list(),
            daily_meta.get_column("day_close").to_list(),
            strict=True,
        )
    )

    # ------------------------------------------------------------------
    # 2. For each trading day, compute OR range and detect breakout
    # ------------------------------------------------------------------
    records: list[dict[str, object]] = []

    for td in trade_dates:
        day_bars = bars.filter(pl.col("ts_ist").cast(pl.Date) == td).sort("ts_ist")
        if day_bars.height == 0:
            continue

        first_ts = day_bars.get_column("ts_ist")[0]
        or_end_ts = first_ts + timedelta(minutes=or_minutes)

        # Opening-range bars: 09:15 .. 09:15 + or_minutes (exclusive)
        or_bars = day_bars.filter(pl.col("ts_ist") < or_end_ts)
        if or_bars.height == 0:
            continue

        or_high: float = or_bars.get_column("high").max()  # type: ignore[assignment]
        or_low: float = or_bars.get_column("low").min()  # type: ignore[assignment]

        # Post-OR bars (from or_end_ts onward)
        post_bars = day_bars.filter(pl.col("ts_ist") >= or_end_ts)
        if post_bars.height == 0:
            # All bars are within the opening range (very short session)
            records.append(
                {
                    "trade_date": td,
                    "breakout": False,
                    "direction": None,
                    "continuation": False,
                    "cont_size_pct": None,
                }
            )
            continue

        # Find first bar that breaks above OR high or below OR low
        up_break_bars = post_bars.filter(pl.col("high") > or_high)
        down_break_bars = post_bars.filter(pl.col("low") < or_low)

        up_ts = up_break_bars.get_column("ts_ist")[0] if up_break_bars.height > 0 else None
        down_ts = down_break_bars.get_column("ts_ist")[0] if down_break_bars.height > 0 else None

        if up_ts is None and down_ts is None:
            # No breakout
            records.append(
                {
                    "trade_date": td,
                    "breakout": False,
                    "direction": None,
                    "continuation": False,
                    "cont_size_pct": None,
                }
            )
            continue

        if up_ts is not None and down_ts is not None and up_ts == down_ts:
            # Both sides break in the same minute -- exclude
            continue

        # Determine which side broke first
        if down_ts is None or (up_ts is not None and up_ts < down_ts):
            direction = "upside"
            breakout_level = or_high
        else:
            direction = "downside"
            breakout_level = or_low

        eod_close = day_closes[td]
        if direction == "upside":
            continuation = eod_close > breakout_level
            cont_size_pct = (eod_close - breakout_level) / breakout_level * 100.0
        else:
            continuation = eod_close < breakout_level
            cont_size_pct = (breakout_level - eod_close) / breakout_level * 100.0

        records.append(
            {
                "trade_date": td,
                "breakout": True,
                "direction": direction,
                "continuation": continuation,
                "cont_size_pct": cont_size_pct,
            }
        )

    if not records:
        return _empty_result(params)

    df = pl.DataFrame(records)
    total_days = df.height
    breakout_days = df.filter(pl.col("breakout")).height

    # ------------------------------------------------------------------
    # 3. Recency cutoff
    # ------------------------------------------------------------------
    all_dates = df.get_column("trade_date").sort()
    if all_dates.len() > params.recency_window_days:
        recency_cutoff = all_dates[-(params.recency_window_days)]
    else:
        recency_cutoff = all_dates[0]

    # ------------------------------------------------------------------
    # 4. Per-direction stats
    # ------------------------------------------------------------------
    breakouts = df.filter(pl.col("breakout"))
    rows_out: list[dict[str, object]] = []

    for direction in ("upside", "downside"):
        subset = breakouts.filter(pl.col("direction") == direction)
        n = subset.height
        cont_n = subset.filter(pl.col("continuation")).height
        false_n = n - cont_n

        cont_rate = cont_n / n if n > 0 else 0.0
        cont_ci = wilson_ci(cont_n, n)
        false_rate = false_n / n if n > 0 else 0.0
        false_ci = wilson_ci(false_n, n)

        cont_series = subset.filter(pl.col("continuation")).get_column("cont_size_pct")
        avg_cont: float | None = float(cont_series.mean()) if cont_series.len() > 0 else None  # type: ignore[arg-type]

        # Recent window
        recent = subset.filter(pl.col("trade_date") >= recency_cutoff)
        recent_n = recent.height
        recent_cont = recent.filter(pl.col("continuation")).height
        recent_cont_rate = recent_cont / recent_n if recent_n > 0 else None

        bo_rate = n / total_days if total_days > 0 else 0.0
        bo_ci = wilson_ci(n, total_days)

        rows_out.append(
            {
                "breakout_direction": direction,
                "instances": n,
                "breakout_rate": round(bo_rate, 4),
                "breakout_rate_ci_low": round(bo_ci[0], 4),
                "breakout_rate_ci_high": round(bo_ci[1], 4),
                "continuation_rate": round(cont_rate, 4),
                "continuation_rate_ci_low": round(cont_ci[0], 4),
                "continuation_rate_ci_high": round(cont_ci[1], 4),
                "false_break_rate": round(false_rate, 4),
                "false_break_rate_ci_low": round(false_ci[0], 4),
                "false_break_rate_ci_high": round(false_ci[1], 4),
                "avg_continuation_size_pct": round(avg_cont, 4) if avg_cont is not None else None,
                "recent_30d_continuation_rate": (
                    round(recent_cont_rate, 4) if recent_cont_rate is not None else None
                ),
            }
        )

    result_df = pl.DataFrame(rows_out)

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    bo_rate_all = breakout_days / total_days if total_days > 0 else 0.0
    total_cont = breakouts.filter(pl.col("continuation")).height

    summary: dict[str, object] = {
        "symbol": params.symbol,
        "or_minutes": or_minutes,
        "lookback_days": params.lookback_days,
        "total_days": total_days,
        "breakout_days": breakout_days,
        "breakout_rate": round(bo_rate_all, 4),
        "continuation_count": total_cont,
        "overall_continuation_rate": (
            round(total_cont / breakout_days, 4) if breakout_days > 0 else 0.0
        ),
        "date_range": (str(all_dates[0]), str(all_dates[-1])),
    }

    return ReportResult(
        buckets=result_df,
        summary=summary,
        methodology=_methodology(or_minutes),
    )


def _empty_result(params: ReportParams) -> ReportResult:
    return ReportResult(
        buckets=pl.DataFrame(),
        summary={
            "symbol": params.symbol,
            "total_days": 0,
            "breakout_days": 0,
            "breakout_rate": 0.0,
        },
        methodology=_methodology(params.or_minutes),
    )


def _methodology(or_minutes: int) -> str:
    return f"""\
Opening Range Breakout (ORB) Report
====================================
Opening range = high and low of the first {or_minutes} minutes after market
open (09:15-09:{15 + or_minutes - 1:02d} IST, {or_minutes} bars).

A breakout occurs when, after the opening range closes, price trades above
the opening-range high (upside) or below the opening-range low (downside).
Whichever side breaks first determines the direction.  If both sides break
in the same 1-min bar, the day is excluded (ambiguous).

Continuation: EOD close is on the breakout side of the breakout level.
False break: breakout occurred but EOD close is on the other side.

Days with fewer than {_MIN_SESSION_BARS} bars are excluded (Muhurat / short sessions).
"""
