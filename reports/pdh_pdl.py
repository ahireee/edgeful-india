"""Previous Day High / Previous Day Low (PDH/PDL) breakout report.

For each trading day, the PDH and PDL are the prior trading day's intraday
high and low.  During today's session, we ask: which side (if either) broke
first, and did the day close on the breakout side (continuation) or return
back inside the prior range (false break)?

The output schema matches ORB/IB so the same printer can render it.
See SPEC.md section 4.
"""

from __future__ import annotations

import polars as pl

from reports.base import ReportParams, ReportResult, wilson_ci

_MIN_SESSION_BARS: int = 300


def compute(bars: pl.DataFrame, params: ReportParams) -> ReportResult:
    """Compute the PDH/PDL breakout probability table.

    Parameters
    ----------
    bars : pl.DataFrame
        1-min bars with columns: symbol, ts_ist, open, high, low, close, volume.
        Should already be filtered to the requested symbol.
    params : ReportParams
        Lookback / recency window.  ``or_minutes`` is ignored.
    """
    # ------------------------------------------------------------------
    # 1. Daily aggregates (high, low, close, bar_count)
    # ------------------------------------------------------------------
    daily = (
        bars.sort("ts_ist")
        .group_by(pl.col("ts_ist").cast(pl.Date).alias("trade_date"))
        .agg(
            pl.col("high").max().alias("day_high"),
            pl.col("low").min().alias("day_low"),
            pl.col("close").last().alias("day_close"),
            pl.len().alias("bar_count"),
        )
        .sort("trade_date")
        .filter(pl.col("bar_count") >= _MIN_SESSION_BARS)
    )

    if daily.height < 2:
        return _empty_result(params)

    # Prior day's high/low become today's PDH/PDL.
    daily = daily.with_columns(
        daily.get_column("day_high").shift(1).alias("pdh"),
        daily.get_column("day_low").shift(1).alias("pdl"),
    ).filter(pl.col("pdh").is_not_null())

    # Apply lookback after PDH/PDL are attached so we don't drop a valid
    # day just because its prior day got trimmed.
    daily = daily.tail(params.lookback_days)

    if daily.height == 0:
        return _empty_result(params)

    trade_dates = daily.get_column("trade_date").to_list()
    pdhs = daily.get_column("pdh").to_list()
    pdls = daily.get_column("pdl").to_list()
    closes = daily.get_column("day_close").to_list()

    # ------------------------------------------------------------------
    # 2. Per-day breakout detection on the intraday bars
    # ------------------------------------------------------------------
    records: list[dict[str, object]] = []

    for td, pdh, pdl, eod_close in zip(trade_dates, pdhs, pdls, closes, strict=True):
        day_bars = bars.filter(pl.col("ts_ist").cast(pl.Date) == td).sort("ts_ist")
        if day_bars.height == 0:
            continue

        up_break_bars = day_bars.filter(pl.col("high") > pdh)
        down_break_bars = day_bars.filter(pl.col("low") < pdl)

        up_ts = up_break_bars.get_column("ts_ist")[0] if up_break_bars.height > 0 else None
        down_ts = down_break_bars.get_column("ts_ist")[0] if down_break_bars.height > 0 else None

        if up_ts is None and down_ts is None:
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
            # Both sides break in the same 1-min bar -- ambiguous, exclude.
            continue

        if down_ts is None or (up_ts is not None and up_ts < down_ts):
            direction = "upside"
            level = pdh
        else:
            direction = "downside"
            level = pdl

        if direction == "upside":
            continuation = eod_close > level
            cont_size_pct = (eod_close - level) / level * 100.0
        else:
            continuation = eod_close < level
            cont_size_pct = (level - eod_close) / level * 100.0

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
    # 5. Summary (same shape as ORB/IB so the printer can be reused)
    # ------------------------------------------------------------------
    bo_rate_all = breakout_days / total_days if total_days > 0 else 0.0
    total_cont = breakouts.filter(pl.col("continuation")).height

    summary: dict[str, object] = {
        "symbol": params.symbol,
        "or_minutes": None,
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
        methodology=_METHODOLOGY,
    )


def _empty_result(params: ReportParams) -> ReportResult:
    return ReportResult(
        buckets=pl.DataFrame(),
        summary={
            "symbol": params.symbol,
            "or_minutes": None,
            "total_days": 0,
            "breakout_days": 0,
            "breakout_rate": 0.0,
        },
        methodology=_METHODOLOGY,
    )


_METHODOLOGY = f"""\
Previous Day High/Low (PDH/PDL) Breakout Report
=================================================
For each trading day, the PDH (previous day high) and PDL (previous day low)
are taken from the prior trading day's intraday bars.  We then scan today's
1-min bars and ask which side broke first:

  - Upside break: today's bar high > PDH (today trades above yesterday's high)
  - Downside break: today's bar low < PDL (today trades below yesterday's low)

Whichever side breaks first sets the day's direction.  If both sides break
in the same 1-min bar, the day is excluded (ambiguous).  If neither side
ever breaks, the day is "no breakout".

Continuation: EOD close on the breakout side of the breached level
              (close > PDH for upside, close < PDL for downside).
False break: breakout occurred but EOD closed back inside the prior range.

Days with fewer than {_MIN_SESSION_BARS} bars are excluded (Muhurat / short sessions).
The first day in the lookback window is dropped (no prior day available).
"""
