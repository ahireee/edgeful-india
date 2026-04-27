"""Session Bias report.

Given the open price relative to yesterday's close (i.e. the gap), what is
the probability that today closes "green" (close > open) vs "red" (close <
open)?  Pure function on a Polars DataFrame.

We bucket gap days by size and direction (mirroring the Gap Fill report's
bucket schema for consistency) and report green/red rates per bucket.

See SPEC.md section 5.
"""

from __future__ import annotations

import polars as pl

from reports.base import MIN_GAP_PCT, ReportParams, ReportResult, wilson_ci

_EDGES: list[float] = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
_LABELS: list[str] = [
    "0.0-0.1%",
    "0.1-0.25%",
    "0.25-0.5%",
    "0.5-1.0%",
    "1.0-2.0%",
    "2.0%+",
]

_MIN_SESSION_BARS: int = 300


def _assign_bucket(abs_gap_pct: float) -> str:
    for i in range(len(_EDGES) - 1):
        if abs_gap_pct < _EDGES[i + 1]:
            return _LABELS[i]
    return _LABELS[-1]


def compute(bars: pl.DataFrame, params: ReportParams) -> ReportResult:
    """Compute the Session Bias probability table.

    For every trading day with a meaningful overnight gap, we record whether
    the session closed green (close > open) or red (close < open), then
    group by gap-size bucket and gap direction.
    """
    # ------------------------------------------------------------------
    # 1. Daily aggregates (open, close, bar_count)
    # ------------------------------------------------------------------
    daily = (
        bars.sort("ts_ist")
        .group_by(pl.col("ts_ist").cast(pl.Date).alias("trade_date"))
        .agg(
            pl.col("open").first().alias("day_open"),
            pl.col("close").last().alias("day_close"),
            pl.len().alias("bar_count"),
        )
        .sort("trade_date")
        .filter(pl.col("bar_count") >= _MIN_SESSION_BARS)
        .tail(params.lookback_days)
    )

    if daily.height < 2:
        return _empty_result(params)

    prev_close = daily.get_column("day_close").shift(1)
    gap_pct = (daily.get_column("day_open") - prev_close) / prev_close * 100.0

    daily = daily.with_columns(
        prev_close.alias("prev_close"),
        gap_pct.alias("gap_pct"),
        gap_pct.abs().alias("abs_gap_pct"),
        (
            (daily.get_column("day_close") - daily.get_column("day_open"))
            / daily.get_column("day_open")
            * 100.0
        ).alias("session_change_pct"),
    ).filter(pl.col("prev_close").is_not_null() & (pl.col("abs_gap_pct") >= MIN_GAP_PCT))

    if daily.height == 0:
        return _empty_result(params)

    # ------------------------------------------------------------------
    # 2. Bucket + direction labels, green / red flags
    # ------------------------------------------------------------------
    daily = daily.with_columns(
        pl.col("abs_gap_pct").map_elements(_assign_bucket, return_dtype=pl.Utf8).alias("bucket"),
        pl.when(pl.col("gap_pct") > 0)
        .then(pl.lit("gap_up"))
        .otherwise(pl.lit("gap_down"))
        .alias("direction"),
        (pl.col("session_change_pct") > 0).alias("green"),
        (pl.col("session_change_pct") < 0).alias("red"),
    )

    # ------------------------------------------------------------------
    # 3. Recency cutoff
    # ------------------------------------------------------------------
    all_dates = daily.get_column("trade_date").sort()
    if all_dates.len() > params.recency_window_days:
        recency_cutoff = all_dates[-(params.recency_window_days)]
    else:
        recency_cutoff = all_dates[0]

    # ------------------------------------------------------------------
    # 4. Aggregate per (bucket, direction)
    # ------------------------------------------------------------------
    groups = daily.group_by(["bucket", "direction"]).agg(
        pl.len().alias("instances"),
        pl.col("green").sum().alias("green_count"),
        pl.col("red").sum().alias("red_count"),
        pl.col("session_change_pct").mean().alias("avg_session_change_pct"),
    )

    recent = (
        daily.filter(pl.col("trade_date") >= recency_cutoff)
        .group_by(["bucket", "direction"])
        .agg(
            pl.len().alias("recent_instances"),
            pl.col("green").sum().alias("recent_green"),
        )
    )

    result = groups.join(recent, on=["bucket", "direction"], how="left").with_columns(
        pl.col("recent_instances").fill_null(0),
        pl.col("recent_green").fill_null(0),
    )

    # ------------------------------------------------------------------
    # 5. Compute rates + CIs
    # ------------------------------------------------------------------
    green_rates: list[float] = []
    green_ci_low: list[float] = []
    green_ci_high: list[float] = []
    red_rates: list[float] = []
    recent_green_rates: list[float | None] = []

    for row in result.iter_rows(named=True):
        n = row["instances"]
        gk = row["green_count"]
        rk = row["red_count"]
        gr = gk / n if n > 0 else 0.0
        lo, hi = wilson_ci(gk, n)
        green_rates.append(round(gr, 4))
        green_ci_low.append(round(lo, 4))
        green_ci_high.append(round(hi, 4))
        red_rates.append(round(rk / n, 4) if n > 0 else 0.0)

        rn = row["recent_instances"]
        rg = row["recent_green"]
        recent_green_rates.append(round(rg / rn, 4) if rn > 0 else None)

    result = result.with_columns(
        pl.Series("green_rate", green_rates),
        pl.Series("green_rate_ci_low", green_ci_low),
        pl.Series("green_rate_ci_high", green_ci_high),
        pl.Series("red_rate", red_rates),
        pl.Series("recent_30d_green_rate", recent_green_rates),
    )

    bucket_order = {label: i for i, label in enumerate(_LABELS)}
    result = (
        result.select(
            "bucket",
            "direction",
            "instances",
            "green_rate",
            "green_rate_ci_low",
            "green_rate_ci_high",
            "red_rate",
            "avg_session_change_pct",
            "recent_30d_green_rate",
        )
        .with_columns(
            pl.col("bucket")
            .map_elements(lambda b: bucket_order.get(b, 99), return_dtype=pl.Int64)
            .alias("_order"),
            pl.col("avg_session_change_pct").round(4),
        )
        .sort(["_order", "direction"])
        .drop("_order")
    )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    total_days = daily.height
    green_total = daily.filter(pl.col("green")).height
    red_total = daily.filter(pl.col("red")).height
    overall_green_rate = green_total / total_days if total_days > 0 else 0.0

    summary: dict[str, object] = {
        "symbol": params.symbol,
        "lookback_days": params.lookback_days,
        "total_gap_days": total_days,
        "green_count": green_total,
        "red_count": red_total,
        "overall_green_rate": round(overall_green_rate, 4),
        "date_range": (str(all_dates[0]), str(all_dates[-1])),
    }

    return ReportResult(buckets=result, summary=summary, methodology=_METHODOLOGY)


def _empty_result(params: ReportParams) -> ReportResult:
    return ReportResult(
        buckets=pl.DataFrame(),
        summary={
            "symbol": params.symbol,
            "total_gap_days": 0,
            "green_count": 0,
            "red_count": 0,
            "overall_green_rate": 0.0,
        },
        methodology=_METHODOLOGY,
    )


_METHODOLOGY = f"""\
Session Bias Report
====================
For every trading day with a meaningful overnight gap (|gap| >= {MIN_GAP_PCT:.3f}%),
we record:

  - gap_pct      = (today_open - prev_close) / prev_close * 100
  - direction    = gap_up if gap_pct > 0 else gap_down
  - bucket       = absolute-size bucket (same edges as Gap Fill)
  - session_chg  = (today_close - today_open) / today_open * 100
  - green        = session_chg > 0  (close > open)
  - red          = session_chg < 0  (close < open)

For each (bucket, direction) we report:
  - N instances, green_rate with 95% Wilson CI, red_rate
  - average session_change_pct (signed)
  - 30-trading-day recent green_rate (decay flag if |full - recent| > 15pp)

The question this answers: given today's open is X% above (or below)
yesterday's close, does that bias today toward closing in the direction of
the gap, against it, or neither?

Days with fewer than {_MIN_SESSION_BARS} bars are excluded (Muhurat / short sessions).
Buckets: {", ".join(_LABELS)}
"""
