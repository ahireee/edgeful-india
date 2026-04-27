"""Streamlit dashboard for edgeful-india probability reports.

Single-page app: pick a symbol + lookback, select one or more reports,
and the page renders each report's probability table plus headline stats.
All computation reuses the pure compute() functions in reports/*.py.

Launch:
    uv run streamlit run dashboard/app.py
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import polars as pl
import streamlit as st

from data.db import get_conn
from data.universe import UNIVERSE
from reports.base import ReportParams, ReportResult
from reports.engulfing import compute as compute_engulfing
from reports.gap_fill import compute as compute_gap_fill
from reports.ib import compute as compute_ib
from reports.orb import compute as compute_orb
from reports.pdh_pdl import compute as compute_pdh_pdl
from reports.session_bias import compute as compute_session_bias


# ---------------------------------------------------------------------------
# Report registry: keeps all per-report metadata in one place so the rest of
# the page is dumb rendering.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReportSpec:
    key: str
    title: str
    blurb: str
    compute_fn: Callable[[pl.DataFrame, ReportParams], ReportResult]
    headline_keys: tuple[tuple[str, str, str], ...]
    """List of (summary_key, label, fmt) for the headline-stat row.
    fmt is one of "int", "pct", "raw"."""


REPORTS: list[ReportSpec] = [
    ReportSpec(
        key="gap_fill",
        title="Gap Fill",
        blurb=(
            "How often an overnight gap (today_open vs prev_close) closes back to "
            "yesterday's close during the session, bucketed by gap size."
        ),
        compute_fn=compute_gap_fill,
        headline_keys=(
            ("total_gap_days", "Gap days", "int"),
            ("total_fills", "Fills", "int"),
            ("overall_fill_rate", "Overall fill rate", "pct"),
        ),
    ),
    ReportSpec(
        key="orb",
        title="Opening Range Breakout (15-min)",
        blurb=(
            "First 15 minutes form the opening range. Per direction we measure "
            "breakout, continuation (close on the breakout side), and false-break rates."
        ),
        compute_fn=compute_orb,
        headline_keys=(
            ("total_days", "Total days", "int"),
            ("breakout_days", "Breakout days", "int"),
            ("overall_continuation_rate", "Continuation rate", "pct"),
        ),
    ),
    ReportSpec(
        key="ib",
        title="Initial Balance Breakout (60-min)",
        blurb=(
            "Same as ORB but with a 60-minute opening window. Tests whether a longer "
            "window filters morning noise into more reliable continuation."
        ),
        compute_fn=compute_ib,
        headline_keys=(
            ("total_days", "Total days", "int"),
            ("breakout_days", "Breakout days", "int"),
            ("overall_continuation_rate", "Continuation rate", "pct"),
        ),
    ),
    ReportSpec(
        key="pdh_pdl",
        title="Previous Day High / Low Breaks",
        blurb=(
            "Did today break yesterday's high or low first, and did it then continue "
            "(close on the breakout side) or fade back inside the prior range?"
        ),
        compute_fn=compute_pdh_pdl,
        headline_keys=(
            ("total_days", "Total days", "int"),
            ("breakout_days", "Breakout days", "int"),
            ("overall_continuation_rate", "Continuation rate", "pct"),
        ),
    ),
    ReportSpec(
        key="session_bias",
        title="Session Bias",
        blurb=(
            "Given the open relative to yesterday's close (the gap), what is the "
            "probability the session closes green (close > open) vs red?"
        ),
        compute_fn=compute_session_bias,
        headline_keys=(
            ("total_gap_days", "Gap days", "int"),
            ("green_count", "Green", "int"),
            ("red_count", "Red", "int"),
            ("overall_green_rate", "Overall green rate", "pct"),
        ),
    ),
    ReportSpec(
        key="engulfing",
        title="Engulfing Candle Reversals (15m, K=3)",
        blurb=(
            "On 15-minute candles, when a bullish or bearish engulfing pattern fires, "
            "how often does price follow through three candles later?"
        ),
        compute_fn=compute_engulfing,
        headline_keys=(
            ("total_engulfings", "Total signals", "int"),
            ("total_reversals", "Confirmed reversals", "int"),
            ("overall_reversal_rate", "Reversal rate", "pct"),
        ),
    ),
]


REPORTS_BY_KEY: dict[str, ReportSpec] = {r.key: r for r in REPORTS}


# ---------------------------------------------------------------------------
# Data loading -- cached per (symbol, lookback) so flipping reports is fast.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)  # type: ignore[untyped-decorator]
def load_bars(symbol: str) -> pl.DataFrame:
    """Load all 1-min bars for ``symbol`` from DuckDB. Lookback windowing
    happens inside each report's compute(), so we can cache just by symbol."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT symbol, ts_ist, open, high, low, close, volume "
        "FROM bars_1min WHERE symbol = ? ORDER BY ts_ist",
        [symbol],
    ).fetchall()
    conn.close()
    return pl.DataFrame(
        rows,
        schema=["symbol", "ts_ist", "open", "high", "low", "close", "volume"],
        orient="row",
    )


@st.cache_data(show_spinner=False)  # type: ignore[untyped-decorator]
def last_data_date(symbol: str) -> date | None:
    """Most recent ``trade_date`` in bars_daily for the symbol (None if missing)."""
    conn = get_conn()
    row = conn.execute(
        "SELECT MAX(trade_date) FROM bars_daily WHERE symbol = ?", [symbol]
    ).fetchone()
    conn.close()
    if row is None or row[0] is None:
        return None
    return row[0]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _fmt_metric(value: object, fmt: str) -> str:
    if value is None:
        return "—"
    if fmt == "pct" and isinstance(value, int | float):
        return f"{value:.1%}"
    if fmt == "int" and isinstance(value, int | float):
        return f"{int(value):,}"
    return str(value)


def render_report(spec: ReportSpec, bars: pl.DataFrame, params: ReportParams) -> None:
    st.subheader(spec.title)
    st.caption(spec.blurb)

    try:
        result = spec.compute_fn(bars, params)
    except Exception as exc:  # narrow surface: report failed, keep page alive
        st.error(f"Report failed to compute: {exc}")
        return

    summary = result["summary"]
    buckets = result["buckets"]

    # Headline row
    headline_cols = st.columns(len(spec.headline_keys))
    for col, (key, label, fmt) in zip(headline_cols, spec.headline_keys, strict=True):
        col.metric(label, _fmt_metric(summary.get(key), fmt))

    if buckets.height == 0:
        st.info("No rows returned for the current parameters.")
        return

    display_df, column_config = _prepare_for_display(buckets)
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config=column_config,
    )


# Columns whose values are stored as 0-1 fractions and should display as
# percentages.  Scale to 0-100 before rendering so st.column_config's
# "%.1f%%" formatter produces "62.5%" rather than "0.6%".
_RATE_COLS: frozenset[str] = frozenset(
    {
        "fill_rate",
        "fill_rate_ci_low",
        "fill_rate_ci_high",
        "recent_30d_fill_rate",
        "breakout_rate",
        "breakout_rate_ci_low",
        "breakout_rate_ci_high",
        "continuation_rate",
        "continuation_rate_ci_low",
        "continuation_rate_ci_high",
        "false_break_rate",
        "false_break_rate_ci_low",
        "false_break_rate_ci_high",
        "recent_30d_continuation_rate",
        "green_rate",
        "green_rate_ci_low",
        "green_rate_ci_high",
        "red_rate",
        "recent_30d_green_rate",
        "reversal_rate",
        "reversal_rate_ci_low",
        "reversal_rate_ci_high",
        "recent_30d_reversal_rate",
    }
)

# Columns that are already in percentage points (signed deltas / sizes), no scaling.
_PCT_POINT_COLS: frozenset[str] = frozenset(
    {
        "avg_continuation_size_pct",
        "avg_session_change_pct",
        "avg_forward_pct",
    }
)


def _prepare_for_display(
    buckets: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, st.column_config.Column]]:
    """Scale 0-1 rate columns to 0-100 and build a column_config that renders
    each column with appropriate formatting and widths."""
    rate_cols_present = [c for c in buckets.columns if c in _RATE_COLS]
    if rate_cols_present:
        buckets = buckets.with_columns([(pl.col(c) * 100.0).alias(c) for c in rate_cols_present])

    cfg: dict[str, st.column_config.Column] = {}
    for col in buckets.columns:
        if col in _RATE_COLS:
            cfg[col] = st.column_config.NumberColumn(col, format="%.1f%%", width="small")
        elif col in _PCT_POINT_COLS:
            cfg[col] = st.column_config.NumberColumn(col, format="%+.2f%%", width="small")
        elif col in {"avg_minutes_to_fill", "median_minutes_to_fill"}:
            cfg[col] = st.column_config.NumberColumn(col, format="%.1f", width="small")
        elif col == "instances":
            cfg[col] = st.column_config.NumberColumn(col, format="%d", width="small")
        elif col in {"bucket", "direction", "engulf_type", "breakout_direction"}:
            cfg[col] = st.column_config.TextColumn(col, width="small")

    return buckets, cfg


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="edgeful-india",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---- Sidebar ---------------------------------------------------------
    st.sidebar.title("edgeful-india")
    st.sidebar.caption("Probability reports for NIFTY, BANKNIFTY, and the top 10 NIFTY stocks.")

    symbols = [inst.symbol for inst in UNIVERSE]
    symbol = st.sidebar.selectbox("Symbol", symbols, index=0)

    lookback = st.sidebar.slider(
        "Lookback (trading days)",
        min_value=60,
        max_value=500,
        value=180,
        step=10,
    )

    selected_keys = st.sidebar.multiselect(
        "Reports",
        options=[r.key for r in REPORTS],
        default=["gap_fill", "orb", "pdh_pdl"],
        format_func=lambda k: REPORTS_BY_KEY[k].title,
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "Reports use a 30-day recency window inside the lookback. "
        "A divergence above 15 percentage points between full-window and "
        "recent rates surfaces in the table itself."
    )

    # ---- Main page header ------------------------------------------------
    st.title("edgeful-india")
    st.write(
        "Pre-computed historical probabilities for recurring intraday setups on "
        "Indian markets. Data is sourced from the Upstox API and stored locally "
        "in DuckDB."
    )

    last_date = last_data_date(symbol)
    last_str = last_date.isoformat() if last_date is not None else "no data"
    info_cols = st.columns([2, 1, 1])
    info_cols[0].markdown(f"**Symbol:** `{symbol}`")
    info_cols[1].markdown(f"**Lookback:** {lookback} days")
    info_cols[2].markdown(f"**Last data update:** {last_str}")

    if last_date is None:
        st.error(f"No bars found for {symbol}. Run the backfill script first.")
        return

    if not selected_keys:
        st.info("Select one or more reports in the sidebar to begin.")
        return

    # ---- Load bars (cached) ---------------------------------------------
    with st.spinner(f"Loading bars for {symbol}..."):
        bars = load_bars(symbol)
    st.caption(f"{bars.height:,} 1-minute bars loaded.")

    # ---- Render selected reports ----------------------------------------
    params = ReportParams(symbol=symbol, lookback_days=lookback)
    for key in selected_keys:
        st.divider()
        render_report(REPORTS_BY_KEY[key], bars, params)

    # ---- Footer ----------------------------------------------------------
    st.divider()
    st.caption(
        "Methodology: see `docs/SPEC.md` (sections 1-6) for the canonical definition "
        "of each report and `docs/REPORTS.md` for per-report write-ups as they land."
    )


if __name__ == "__main__":
    main()
