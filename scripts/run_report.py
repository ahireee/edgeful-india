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


def _print_orb(result: dict[str, object], report_name: str = "ORB") -> None:
    """Pretty-print an ORB-shaped ReportResult with rich.

    Used for both ORB and IB, which share an identical output schema —
    the only difference is the header label and the OR window length.
    """
    summary = result["summary"]
    buckets = result["buckets"]
    assert isinstance(summary, dict)
    assert isinstance(buckets, pl.DataFrame)

    console.print(f"\n[bold]{report_name} Report — {summary.get('symbol', '?')}[/bold]")
    dr = summary.get("date_range", ("?", "?"))
    assert isinstance(dr, tuple)
    console.print(
        f"  OR window: {summary.get('or_minutes')} min  |  "
        f"Lookback: {summary.get('lookback_days')} days ({dr[0]} -> {dr[1]})"
    )
    console.print(
        f"  Total days: {summary.get('total_days')}  |  "
        f"Breakout days: {summary.get('breakout_days')}  |  "
        f"Breakout rate: {summary.get('breakout_rate', 0):.1%}  |  "
        f"Continuation rate: {summary.get('overall_continuation_rate', 0):.1%}\n"
    )

    if buckets.height == 0:
        console.print("[yellow]No breakout data for the requested parameters.[/yellow]")
        return

    table = Table(title=f"{report_name} Breakout Statistics", show_lines=True)
    table.add_column("Direction", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("BO rate", justify="right")
    table.add_column("Cont%", justify="right", style="green")
    table.add_column("Cont CI", justify="right")
    table.add_column("False%", justify="right", style="red")
    table.add_column("False CI", justify="right")
    table.add_column("Avg cont%", justify="right")
    table.add_column("Recent 30d", justify="right")

    for row in buckets.iter_rows(named=True):
        avg_c = (
            f"{row['avg_continuation_size_pct']:.2f}%"
            if row["avg_continuation_size_pct"] is not None
            else "—"
        )
        recent = (
            f"{row['recent_30d_continuation_rate']:.0%}"
            if row["recent_30d_continuation_rate"] is not None
            else "—"
        )
        flag = ""
        if row["recent_30d_continuation_rate"] is not None:
            diff = abs(row["recent_30d_continuation_rate"] - row["continuation_rate"])
            if diff > 0.15:
                flag = " ⚠"

        table.add_row(
            row["breakout_direction"],
            str(row["instances"]),
            f"{row['breakout_rate']:.0%}",
            f"{row['continuation_rate']:.0%}",
            f"{row['continuation_rate_ci_low']:.0%}-{row['continuation_rate_ci_high']:.0%}",
            f"{row['false_break_rate']:.0%}",
            f"{row['false_break_rate_ci_low']:.0%}-{row['false_break_rate_ci_high']:.0%}",
            avg_c,
            recent + flag,
        )

    console.print(table)


@app.command("orb")
def orb(
    symbol: str = typer.Option("NIFTY", help="Ticker symbol"),
    lookback: int = typer.Option(180, help="Lookback in trading days"),
    recency: int = typer.Option(30, help="Recency window in trading days"),
    or_minutes: int = typer.Option(15, "--or-minutes", help="Opening range window in minutes"),
) -> None:
    """Run the Opening Range Breakout report."""
    from reports.orb import compute

    console.print(f"Loading bars for [bold]{symbol}[/bold]...")
    bars = _load_bars(symbol)
    if bars.height == 0:
        console.print(f"[red]No bars found for {symbol}.[/red]")
        raise typer.Exit(1)

    console.print(f"  {bars.height:,} bars loaded.")
    params = ReportParams(
        symbol=symbol,
        lookback_days=lookback,
        recency_window_days=recency,
        or_minutes=or_minutes,
    )
    result = compute(bars, params)
    _print_orb(result)  # type: ignore[arg-type]


@app.command("ib")
def ib(
    symbol: str = typer.Option("NIFTY", help="Ticker symbol"),
    lookback: int = typer.Option(180, help="Lookback in trading days"),
    recency: int = typer.Option(30, help="Recency window in trading days"),
) -> None:
    """Run the Initial Balance Breakout report (60-minute window)."""
    from reports.ib import compute

    console.print(f"Loading bars for [bold]{symbol}[/bold]...")
    bars = _load_bars(symbol)
    if bars.height == 0:
        console.print(f"[red]No bars found for {symbol}.[/red]")
        raise typer.Exit(1)

    console.print(f"  {bars.height:,} bars loaded.")
    params = ReportParams(symbol=symbol, lookback_days=lookback, recency_window_days=recency)
    result = compute(bars, params)
    _print_orb(result, report_name="IB")  # type: ignore[arg-type]


if __name__ == "__main__":
    app()
