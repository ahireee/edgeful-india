"""CLI for running reports against the local DuckDB warehouse.

Usage:
    uv run python scripts/run_report.py gap-fill --symbol NIFTY --lookback 180
"""

from __future__ import annotations

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from data.db import get_conn
from reports.base import ReportParams

app = typer.Typer(help="Run edgeful-india reports from the CLI.")
console = Console()

# Force subcommand mode even with a single command.
# Without this callback, typer collapses the single command into the root.


@app.callback()
def _callback() -> None:
    """Run edgeful-india reports from the CLI."""


def _load_bars(symbol: str) -> pl.DataFrame:
    """Load all 1-min bars for *symbol* from DuckDB into a Polars DataFrame."""
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


def _fmt_minutes(v: float | None) -> str:
    """Format a minutes-to-fill value for display.

    A genuine instant fill (gap closes inside the 09:15 bar) records 0.0 and
    must render as "<1" — distinct from "—", which means no gap in the bucket
    ever filled and the avg/median is undefined.
    """
    if v is None:
        return "—"
    if v < 1:
        return "<1"
    return f"{v:.0f}"


def _print_gap_fill(result: dict[str, object]) -> None:
    """Pretty-print a Gap Fill ReportResult with rich."""
    summary = result["summary"]
    buckets = result["buckets"]
    assert isinstance(summary, dict)
    assert isinstance(buckets, pl.DataFrame)

    console.print(f"\n[bold]Gap Fill Report — {summary.get('symbol', '?')}[/bold]")
    dr = summary.get("date_range", ("?", "?"))
    assert isinstance(dr, tuple)
    console.print(f"  Lookback: {summary.get('lookback_days')} trading days ({dr[0]} → {dr[1]})")
    console.print(
        f"  Gap-days: {summary.get('total_gap_days')}  |  "
        f"Fills: {summary.get('total_fills')}  |  "
        f"Overall fill rate: {summary.get('overall_fill_rate', 0):.1%}\n"
    )

    if buckets.height == 0:
        console.print("[yellow]No gap data for the requested parameters.[/yellow]")
        return

    table = Table(title="Gap Fill Probabilities", show_lines=True)
    table.add_column("Bucket", style="cyan")
    table.add_column("Dir", style="bold")
    table.add_column("N", justify="right")
    table.add_column("Fill%", justify="right", style="green")
    table.add_column("CI 95%", justify="right")
    table.add_column("Avg min", justify="right")
    table.add_column("Med min", justify="right")
    table.add_column("Recent 30d", justify="right")

    for row in buckets.iter_rows(named=True):
        fill_pct = f"{row['fill_rate']:.0%}"
        ci = f"{row['fill_rate_ci_low']:.0%}-{row['fill_rate_ci_high']:.0%}"
        avg_m = _fmt_minutes(row["avg_minutes_to_fill"])
        med_m = _fmt_minutes(row["median_minutes_to_fill"])
        recent = (
            f"{row['recent_30d_fill_rate']:.0%}" if row["recent_30d_fill_rate"] is not None else "—"
        )

        # Flag divergence > 15pp between full and recent
        flag = ""
        if row["recent_30d_fill_rate"] is not None:
            diff = abs(row["recent_30d_fill_rate"] - row["fill_rate"])
            if diff > 0.15:
                flag = " ⚠"

        table.add_row(
            row["bucket"],
            row["direction"],
            str(row["instances"]),
            fill_pct,
            ci,
            avg_m,
            med_m,
            recent + flag,
        )

    console.print(table)


@app.command("gap-fill")
def gap_fill(
    symbol: str = typer.Option("NIFTY", help="Ticker symbol"),
    lookback: int = typer.Option(180, help="Lookback in trading days"),
    recency: int = typer.Option(30, help="Recency window in trading days"),
) -> None:
    """Run the Gap Fill report."""
    from reports.gap_fill import compute

    console.print(f"Loading bars for [bold]{symbol}[/bold]...")
    bars = _load_bars(symbol)
    if bars.height == 0:
        console.print(f"[red]No bars found for {symbol}.[/red]")
        raise typer.Exit(1)

    console.print(f"  {bars.height:,} bars loaded.")
    params = ReportParams(symbol=symbol, lookback_days=lookback, recency_window_days=recency)
    result = compute(bars, params)
    _print_gap_fill(result)  # type: ignore[arg-type]


if __name__ == "__main__":
    app()
