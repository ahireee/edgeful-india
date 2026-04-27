"""Engulfing Candle Reversals report.

Resamples 1-min bars into a chosen timeframe (5m / 15m / 1h), detects
bullish and bearish engulfing pairs, and measures how often the engulfing
candle is followed by price moving in the signal direction over the next K
candles -- i.e. the engulfing actually marked a reversal.

See SPEC.md section 6.
"""

from __future__ import annotations

import polars as pl

from reports.base import ReportParams, ReportResult, wilson_ci

_MIN_SESSION_BARS: int = 300

_TIMEFRAME_MINUTES: dict[str, int] = {"5m": 5, "15m": 15, "1h": 60}


def _resample_to_timeframe(bars: pl.DataFrame, minutes: int) -> pl.DataFrame:
    """Aggregate 1-min bars into ``minutes``-minute candles, per trading day.

    Buckets are chunked by index from each day's first bar so the first
    candle is 09:15-09:14+minutes regardless of any global alignment.
    """
    bars = bars.sort("ts_ist").with_columns(pl.col("ts_ist").cast(pl.Date).alias("trade_date"))

    bars = bars.with_columns(pl.int_range(pl.len()).over("trade_date").alias("intraday_idx"))
    bars = bars.with_columns((pl.col("intraday_idx") // minutes).alias("candle_idx"))

    candles = (
        bars.group_by(["trade_date", "candle_idx"])
        .agg(
            pl.col("ts_ist").first().alias("ts"),
            pl.col("open").first().alias("c_open"),
            pl.col("high").max().alias("c_high"),
            pl.col("low").min().alias("c_low"),
            pl.col("close").last().alias("c_close"),
            pl.len().alias("bar_count"),
        )
        .sort(["trade_date", "candle_idx"])
    )

    # Drop any partial trailing candle that has fewer 1-min bars than the
    # timeframe -- it would distort the OHLC envelope.
    candles = candles.filter(pl.col("bar_count") == minutes)
    return candles


def compute(bars: pl.DataFrame, params: ReportParams) -> ReportResult:
    """Compute the Engulfing Candle Reversal probability table."""
    tf = params.timeframe
    if tf not in _TIMEFRAME_MINUTES:
        raise ValueError(f"Unknown timeframe {tf!r}; expected one of {list(_TIMEFRAME_MINUTES)}.")
    minutes = _TIMEFRAME_MINUTES[tf]
    k = params.lookahead_candles

    # ------------------------------------------------------------------
    # 1. Filter to valid trading days first (exclude short / Muhurat).
    # ------------------------------------------------------------------
    daily_bar_counts = (
        bars.sort("ts_ist")
        .group_by(pl.col("ts_ist").cast(pl.Date).alias("trade_date"))
        .agg(pl.len().alias("bar_count"))
        .filter(pl.col("bar_count") >= _MIN_SESSION_BARS)
        .sort("trade_date")
        .tail(params.lookback_days)
    )

    if daily_bar_counts.height < 2:
        return _empty_result(params)

    valid_dates = set(daily_bar_counts.get_column("trade_date").to_list())
    bars = bars.filter(pl.col("ts_ist").cast(pl.Date).is_in(list(valid_dates)))

    # ------------------------------------------------------------------
    # 2. Resample to the requested timeframe.
    # ------------------------------------------------------------------
    candles = _resample_to_timeframe(bars, minutes)
    if candles.height == 0:
        return _empty_result(params)

    # ------------------------------------------------------------------
    # 3. Detect engulfing pairs within each trading day.
    # ------------------------------------------------------------------
    candles = candles.with_columns(
        pl.col("c_open").shift(1).over("trade_date").alias("prev_open"),
        pl.col("c_close").shift(1).over("trade_date").alias("prev_close"),
        pl.col("candle_idx").shift(1).over("trade_date").alias("prev_idx"),
    )

    # First candle of each day has null prev_*; drop it.
    candles = candles.filter(pl.col("prev_idx").is_not_null())

    bullish = (
        (pl.col("prev_close") < pl.col("prev_open"))  # prev red
        & (pl.col("c_close") > pl.col("c_open"))  # curr green
        & (pl.col("c_open") <= pl.col("prev_close"))  # curr body engulfs prev body
        & (pl.col("c_close") >= pl.col("prev_open"))
    )
    bearish = (
        (pl.col("prev_close") > pl.col("prev_open"))
        & (pl.col("c_close") < pl.col("c_open"))
        & (pl.col("c_open") >= pl.col("prev_close"))
        & (pl.col("c_close") <= pl.col("prev_open"))
    )

    candles = candles.with_columns(
        pl.when(bullish)
        .then(pl.lit("bullish"))
        .when(bearish)
        .then(pl.lit("bearish"))
        .otherwise(None)
        .alias("engulf_type"),
    )

    engulfings = candles.filter(pl.col("engulf_type").is_not_null())
    if engulfings.height == 0:
        return _empty_result(params)

    # ------------------------------------------------------------------
    # 4. For each engulfing, look ahead K candles within the same day to
    #    determine whether the reversal followed through.
    # ------------------------------------------------------------------
    # Build a (trade_date, candle_idx) -> close lookup from the resampled
    # candle table (pre-shift) so we can resolve forward closes cheaply.
    fwd = _resample_to_timeframe(bars, minutes).select(["trade_date", "candle_idx", "c_close"])
    fwd_map = {(r["trade_date"], r["candle_idx"]): r["c_close"] for r in fwd.iter_rows(named=True)}
    last_idx_per_day = fwd.group_by("trade_date").agg(pl.col("candle_idx").max().alias("max_idx"))
    max_idx = {r["trade_date"]: r["max_idx"] for r in last_idx_per_day.iter_rows(named=True)}

    records: list[dict[str, object]] = []
    for row in engulfings.iter_rows(named=True):
        td = row["trade_date"]
        idx = row["candle_idx"]
        if td not in max_idx or idx + k > max_idx[td]:
            # Insufficient forward lookahead within the same trading day.
            continue
        engulf_close = row["c_close"]
        forward_close = fwd_map[(td, idx + k)]
        if row["engulf_type"] == "bullish":
            reversal = forward_close > engulf_close
            forward_pct = (forward_close - engulf_close) / engulf_close * 100.0
        else:
            reversal = forward_close < engulf_close
            forward_pct = (engulf_close - forward_close) / engulf_close * 100.0

        records.append(
            {
                "trade_date": td,
                "engulf_type": row["engulf_type"],
                "reversal": reversal,
                "forward_pct": forward_pct,
            }
        )

    if not records:
        return _empty_result(params)

    df = pl.DataFrame(records)

    # ------------------------------------------------------------------
    # 5. Recency cutoff
    # ------------------------------------------------------------------
    all_dates = df.get_column("trade_date").sort()
    if all_dates.len() > params.recency_window_days:
        recency_cutoff = all_dates[-(params.recency_window_days)]
    else:
        recency_cutoff = all_dates[0]

    # ------------------------------------------------------------------
    # 6. Aggregate by engulfing type
    # ------------------------------------------------------------------
    rows_out: list[dict[str, object]] = []
    for etype in ("bullish", "bearish"):
        subset = df.filter(pl.col("engulf_type") == etype)
        n = subset.height
        rev_n = subset.filter(pl.col("reversal")).height
        rev_rate = rev_n / n if n > 0 else 0.0
        rev_ci = wilson_ci(rev_n, n)

        avg_fwd: float | None = (
            float(subset.get_column("forward_pct").mean()) if n > 0 else None  # type: ignore[arg-type]
        )

        recent = subset.filter(pl.col("trade_date") >= recency_cutoff)
        recent_n = recent.height
        recent_rev = recent.filter(pl.col("reversal")).height
        recent_rate = recent_rev / recent_n if recent_n > 0 else None

        rows_out.append(
            {
                "engulf_type": etype,
                "instances": n,
                "reversal_rate": round(rev_rate, 4),
                "reversal_rate_ci_low": round(rev_ci[0], 4),
                "reversal_rate_ci_high": round(rev_ci[1], 4),
                "avg_forward_pct": round(avg_fwd, 4) if avg_fwd is not None else None,
                "recent_30d_reversal_rate": (
                    round(recent_rate, 4) if recent_rate is not None else None
                ),
            }
        )

    result_df = pl.DataFrame(rows_out)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    total = df.height
    total_rev = df.filter(pl.col("reversal")).height
    overall_rate = total_rev / total if total > 0 else 0.0

    summary: dict[str, object] = {
        "symbol": params.symbol,
        "timeframe": tf,
        "lookahead_candles": k,
        "lookback_days": params.lookback_days,
        "total_engulfings": total,
        "total_reversals": total_rev,
        "overall_reversal_rate": round(overall_rate, 4),
        "date_range": (str(all_dates[0]), str(all_dates[-1])),
    }

    return ReportResult(buckets=result_df, summary=summary, methodology=_methodology(tf, k))


def _empty_result(params: ReportParams) -> ReportResult:
    return ReportResult(
        buckets=pl.DataFrame(),
        summary={
            "symbol": params.symbol,
            "timeframe": params.timeframe,
            "lookahead_candles": params.lookahead_candles,
            "total_engulfings": 0,
            "total_reversals": 0,
            "overall_reversal_rate": 0.0,
        },
        methodology=_methodology(params.timeframe, params.lookahead_candles),
    )


def _methodology(tf: str, k: int) -> str:
    return f"""\
Engulfing Candle Reversals Report
==================================
Timeframe: {tf}.  1-min bars are resampled into non-overlapping {tf} candles,
per trading day (no candle spans two days; trailing partial candles dropped).

Engulfing detection on candle N (with N-1 the prior candle within the same day):

  Bullish engulfing:
    - N-1 is red (close < open)
    - N is green (close > open)
    - N.open <= N-1.close  AND  N.close >= N-1.open  (body engulfs body)

  Bearish engulfing: symmetric (N-1 green, N red, body engulfs body in reverse).

Reversal confirmation: for each engulfing at candle N, look at candle N+{k}
within the same trading day.  The reversal is "confirmed" if N+{k}.close is
on the signal side of N.close (above for bullish, below for bearish).
Engulfings too close to EOD (no N+{k} candle in-day) are excluded.

For each engulfing type we report instance count, reversal rate with 95%
Wilson CI, average forward move (signed in the signal direction), and a
30-trading-day recency rate.
"""
