"""Gap Fill report.

Computes the historical probability of intraday gap fills, bucketed by gap
size and direction.  Pure function on a Polars DataFrame — no I/O.

See docs/REPORTS.md section 1 for the full methodology.
"""

from __future__ import annotations

import polars as pl

from reports.base import MIN_GAP_PCT, ReportParams, ReportResult, wilson_ci

# Gap-size bucket edges (absolute %).  The last bucket is open-ended.
_EDGES: list[float] = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
_LABELS: list[str] = [
    "0.0-0.1%",
    "0.1-0.25%",
    "0.25-0.5%",
    "0.5-1.0%",
    "1.0-2.0%",
    "2.0%+",
]

# Minimum bars expected in a normal trading session (375 = 09:15-15:29).
# Days with fewer than this threshold are excluded (Muhurat / short sessions).
_MIN_SESSION_BARS: int = 300


def _assign_bucket(abs_gap_pct: float) -> str:
    """Return the bucket label for an absolute gap size in percent."""
    for i in range(len(_EDGES) - 1):
        if abs_gap_pct < _EDGES[i + 1]:
            return _LABELS[i]
    return _LABELS[-1]


def _minutes_between(ts_open: object, ts_fill: object) -> float | None:
    """Compute minutes between two timestamps.  Returns None if ts_fill is None."""
    if ts_fill is None:
        return None
    from datetime import datetime

    t0 = ts_open if isinstance(ts_open, datetime) else datetime.fromisoformat(str(ts_open))
    t1 = ts_fill if isinstance(ts_fill, datetime) else datetime.fromisoformat(str(ts_fill))
    return (t1 - t0).total_seconds() / 60.0


def compute(bars: pl.DataFrame, params: ReportParams) -> ReportResult:
    """Compute the gap fill probability table.

    Parameters
    ----------
    bars : pl.DataFrame
        1-min bars with columns: symbol, ts_ist, open, high, low, close, volume.
        Should already be filtered to the requested symbol.
    params : ReportParams
        Controls lookback and recency window.

    Returns
    -------
    ReportResult with ``buckets`` DataFrame containing the probability table.
    """
    # ------------------------------------------------------------------
    # 1. Build daily summaries from 1-min bars
    # ------------------------------------------------------------------
    daily = (
        bars.sort("ts_ist")
        .group_by(pl.col("ts_ist").cast(pl.Date).alias("trade_date"))
        .agg(
            pl.col("open").first().alias("day_open"),
            pl.col("high").max().alias("day_high"),
            pl.col("low").min().alias("day_low"),
            pl.col("close").last().alias("day_close"),
            pl.len().alias("bar_count"),
        )
        .sort("trade_date")
    )

    # Exclude short sessions (Muhurat days, etc.)
    daily = daily.filter(pl.col("bar_count") >= _MIN_SESSION_BARS)

    # Keep only the most recent `lookback_days` trading days
    daily = daily.tail(params.lookback_days)

    if daily.height < 2:
        return ReportResult(
            buckets=pl.DataFrame(),
            summary={"total_gap_days": 0, "total_fills": 0, "overall_fill_rate": 0.0},
            methodology=_METHODOLOGY,
        )

    # ------------------------------------------------------------------
    # 2. Compute gaps: today_open vs prev_close
    # ------------------------------------------------------------------
    prev_close = daily.get_column("day_close").shift(1)
    gap_pct = (daily.get_column("day_open") - prev_close) / prev_close * 100.0

    daily = daily.with_columns(
        prev_close.alias("prev_close"),
        gap_pct.alias("gap_pct"),
        gap_pct.abs().alias("abs_gap_pct"),
    )

    # Drop the first row (no previous day) and gaps smaller than MIN_GAP_PCT
    daily = daily.filter(
        pl.col("prev_close").is_not_null() & (pl.col("abs_gap_pct") >= MIN_GAP_PCT)
    )

    # ------------------------------------------------------------------
    # 3. Determine fill status per day using the 1-min bars
    # ------------------------------------------------------------------
    # For each gap day, scan 1-min bars to find if/when the gap fills.
    trade_dates = daily.get_column("trade_date").to_list()
    prev_closes = daily.get_column("prev_close").to_list()
    gap_pcts = daily.get_column("gap_pct").to_list()
    day_opens_ts: list[object] = []
    filled_list: list[bool] = []
    fill_minute_list: list[float | None] = []

    for td, pc, gp in zip(trade_dates, prev_closes, gap_pcts, strict=True):
        day_bars = bars.filter(pl.col("ts_ist").cast(pl.Date) == td).sort("ts_ist")
        if day_bars.height == 0:
            day_opens_ts.append(None)
            filled_list.append(False)
            fill_minute_list.append(None)
            continue

        open_ts = day_bars.get_column("ts_ist")[0]
        day_opens_ts.append(open_ts)

        if gp > 0:
            # Up-gap: filled when low touches prev_close
            fill_bars = day_bars.filter(pl.col("low") <= pc)
        else:
            # Down-gap: filled when high touches prev_close
            fill_bars = day_bars.filter(pl.col("high") >= pc)

        if fill_bars.height > 0:
            filled_list.append(True)
            fill_ts = fill_bars.get_column("ts_ist")[0]
            fill_minute_list.append(_minutes_between(open_ts, fill_ts))
        else:
            filled_list.append(False)
            fill_minute_list.append(None)

    daily = daily.with_columns(
        pl.Series("filled", filled_list),
        pl.Series("minutes_to_fill", fill_minute_list),
    )

    # ------------------------------------------------------------------
    # 4. Assign buckets and directions
    # ------------------------------------------------------------------
    daily = daily.with_columns(
        pl.col("abs_gap_pct").map_elements(_assign_bucket, return_dtype=pl.Utf8).alias("bucket"),
        pl.when(pl.col("gap_pct") > 0)
        .then(pl.lit("up"))
        .otherwise(pl.lit("down"))
        .alias("direction"),
    )

    # ------------------------------------------------------------------
    # 5. Identify the recency window
    # ------------------------------------------------------------------
    all_dates_sorted = daily.get_column("trade_date").sort()
    if all_dates_sorted.len() > params.recency_window_days:
        recency_cutoff = all_dates_sorted[-(params.recency_window_days)]
    else:
        recency_cutoff = all_dates_sorted[0]

    # ------------------------------------------------------------------
    # 6. Aggregate into the output table
    # ------------------------------------------------------------------
    bucket_order = {label: i for i, label in enumerate(_LABELS)}
    groups = daily.group_by(["bucket", "direction"]).agg(
        pl.len().alias("instances"),
        pl.col("filled").sum().alias("fills"),
        pl.col("minutes_to_fill").mean().alias("avg_minutes_to_fill"),
        pl.col("minutes_to_fill").median().alias("median_minutes_to_fill"),
    )

    # Recent window stats
    recent = (
        daily.filter(pl.col("trade_date") >= recency_cutoff)
        .group_by(["bucket", "direction"])
        .agg(
            pl.len().alias("recent_instances"),
            pl.col("filled").sum().alias("recent_fills"),
        )
    )

    result = groups.join(recent, on=["bucket", "direction"], how="left").with_columns(
        pl.col("recent_instances").fill_null(0),
        pl.col("recent_fills").fill_null(0),
    )

    # Compute rates and CIs row by row
    fill_rates: list[float] = []
    ci_lows: list[float] = []
    ci_highs: list[float] = []
    recent_rates: list[float | None] = []

    for row in result.iter_rows(named=True):
        n = row["instances"]
        k = row["fills"]
        rate = k / n if n > 0 else 0.0
        ci_lo, ci_hi = wilson_ci(k, n)
        fill_rates.append(round(rate, 4))
        ci_lows.append(round(ci_lo, 4))
        ci_highs.append(round(ci_hi, 4))

        rn = row["recent_instances"]
        rk = row["recent_fills"]
        recent_rates.append(round(rk / rn, 4) if rn > 0 else None)

    result = result.with_columns(
        pl.Series("fill_rate", fill_rates),
        pl.Series("fill_rate_ci_low", ci_lows),
        pl.Series("fill_rate_ci_high", ci_highs),
        pl.Series("recent_30d_fill_rate", recent_rates),
    )

    # Select and order output columns
    result = (
        result.select(
            "bucket",
            "direction",
            "instances",
            "fill_rate",
            "fill_rate_ci_low",
            "fill_rate_ci_high",
            "avg_minutes_to_fill",
            "median_minutes_to_fill",
            "recent_30d_fill_rate",
        )
        .with_columns(
            pl.col("bucket")
            .map_elements(lambda b: bucket_order.get(b, 99), return_dtype=pl.Int64)
            .alias("_order")
        )
        .sort(["_order", "direction"])
        .drop("_order")
    )

    # ------------------------------------------------------------------
    # 7. Summary stats
    # ------------------------------------------------------------------
    total_gaps = daily.height
    total_fills = daily.filter(pl.col("filled")).height
    overall_rate = total_fills / total_gaps if total_gaps > 0 else 0.0

    summary: dict[str, object] = {
        "symbol": params.symbol,
        "lookback_days": params.lookback_days,
        "total_gap_days": total_gaps,
        "total_fills": total_fills,
        "overall_fill_rate": round(overall_rate, 4),
        "date_range": (
            str(daily.get_column("trade_date").min()),
            str(daily.get_column("trade_date").max()),
        ),
    }

    return ReportResult(
        buckets=result,
        summary=summary,
        methodology=_METHODOLOGY,
    )


_METHODOLOGY = """\
Gap Fill Report
===============
A "gap" is the difference between today's opening price (09:15 IST bar open)
and yesterday's closing price (15:29 IST bar close).

Gap size = (today_open - prev_close) / prev_close * 100, signed.

A gap is "filled" if at any point during today's session the price touches
yesterday's close:
  - Up-gap filled when today's intraday low <= prev_close
  - Down-gap filled when today's intraday high >= prev_close

Gaps smaller than {min_gap:.3f}% are excluded (noise / rounding).
Days with fewer than {min_bars} bars are excluded (Muhurat / short sessions).

Buckets by absolute gap size: {buckets}

Each bucket reports: instance count, fill rate with 95% Wilson confidence
interval, average and median minutes to fill, and a 30-trading-day recency
rate.  A divergence of >15pp between the full-window and recency rates
flags a potential regime change.
""".format(
    min_gap=MIN_GAP_PCT,
    min_bars=_MIN_SESSION_BARS,
    buckets=", ".join(_LABELS),
)
