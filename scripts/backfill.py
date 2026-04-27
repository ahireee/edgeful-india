"""Backfill historical 1-min bars from Upstox into DuckDB.

Usage:
    uv run python scripts/backfill.py             # full 24-month backfill
    uv run python scripts/backfill.py --months 6  # 6 months
    uv run python scripts/backfill.py --dry-run   # 1 month of NIFTY only
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

import duckdb
import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from data.db import get_conn
from data.universe import UNIVERSE, Instrument
from data.upstox_client import get_history_api, upstox_retry

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="Backfill historical 1-min bars from Upstox.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _month_ranges(months_back: int) -> list[tuple[str, date, date]]:
    """Return (year_month, from_date, to_date) for each calendar month to fetch.

    Goes from *months_back* months ago up to today's month.
    """
    today = date.today()
    ranges: list[tuple[str, date, date]] = []

    # Start from the 1st of the month N months ago
    start = (today.replace(day=1) - timedelta(days=months_back * 30)).replace(day=1)
    cursor = start

    while cursor <= today:
        year_month = cursor.strftime("%Y-%m")
        first = cursor
        # Last day: jump to next month's 1st, subtract a day
        if cursor.month == 12:
            last = date(cursor.year + 1, 1, 1) - timedelta(days=1)
        else:
            last = date(cursor.year, cursor.month + 1, 1) - timedelta(days=1)
        # Cap to today
        if last > today:
            last = today
        ranges.append((year_month, first, last))
        # Advance to next month
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)

    return ranges


def _already_loaded(conn: duckdb.DuckDBPyConnection, symbol: str, year_month: str) -> bool:
    """Check if this (symbol, year_month) is in backfill_log."""
    row = conn.execute(
        "SELECT 1 FROM backfill_log WHERE symbol = ? AND year_month = ?",
        [symbol, year_month],
    ).fetchone()
    return row is not None


@upstox_retry
def _fetch_candles(instrument_key: str, from_date: date, to_date: date) -> list[list[object]]:
    """Fetch 1-min candles from Upstox for the given date range."""
    api = get_history_api()
    resp = api.get_historical_candle_data1(
        instrument_key=instrument_key,
        unit="minutes",
        interval="1",
        to_date=to_date.isoformat(),
        from_date=from_date.isoformat(),
    )
    candles: list[list[object]] = resp.data.candles
    return candles


def _parse_candle_ts(ts_raw: object) -> str:
    """Strip timezone from ISO timestamp to get a naive IST string for DuckDB."""
    # Input: '2026-04-24T15:29:00+05:30'  ->  '2026-04-24 15:29:00'
    s = str(ts_raw)
    # Strip the +05:30 suffix
    if "+" in s:
        s = s[: s.rfind("+")]
    return s.replace("T", " ")


def _insert_candles(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    candles: list[list[object]],
) -> int:
    """Insert candles into bars_1min. Returns number of new rows inserted."""
    if not candles:
        return 0

    count_before: int = conn.execute(
        "SELECT count(*) FROM bars_1min WHERE symbol = ?", [symbol]
    ).fetchone()[0]  # type: ignore[index]

    rows = [
        (
            symbol,
            _parse_candle_ts(c[0]),
            float(str(c[1])),
            float(str(c[2])),
            float(str(c[3])),
            float(str(c[4])),
            int(str(c[5])),
        )
        for c in candles
    ]

    conn.executemany(
        "INSERT OR IGNORE INTO bars_1min (symbol, ts_ist, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )

    count_after: int = conn.execute(
        "SELECT count(*) FROM bars_1min WHERE symbol = ?", [symbol]
    ).fetchone()[0]  # type: ignore[index]

    return count_after - count_before


def _log_backfill(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    year_month: str,
    row_count: int,
) -> None:
    """Record that a (symbol, year_month) chunk has been loaded."""
    conn.execute(
        "INSERT OR IGNORE INTO backfill_log (symbol, year_month, completed_at, row_count) "
        "VALUES (?, ?, ?, ?)",
        [symbol, year_month, datetime.now().isoformat(), row_count],
    )


def _materialize_daily(conn: duckdb.DuckDBPyConnection) -> int:
    """Derive bars_daily from bars_1min. Returns number of daily rows."""
    conn.execute("DELETE FROM bars_daily")
    conn.execute("""
        INSERT INTO bars_daily (symbol, trade_date, open, high, low, close, volume, bar_count)
        SELECT
            symbol,
            CAST(ts_ist AS DATE) AS trade_date,
            FIRST(open ORDER BY ts_ist)   AS open,
            MAX(high)                     AS high,
            MIN(low)                      AS low,
            LAST(close ORDER BY ts_ist)   AS close,
            SUM(volume)                   AS volume,
            COUNT(*)                      AS bar_count
        FROM bars_1min
        GROUP BY symbol, CAST(ts_ist AS DATE)
    """)
    count: int = conn.execute("SELECT count(*) FROM bars_daily").fetchone()[0]  # type: ignore[index]
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _backfill(
    conn: duckdb.DuckDBPyConnection,
    instruments: list[Instrument],
    months: int,
) -> None:
    """Run the backfill for the given instruments and month range."""
    month_ranges = _month_ranges(months)
    total_chunks = len(instruments) * len(month_ranges)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Backfilling", total=total_chunks)

        for inst in instruments:
            for year_month, from_date, to_date in month_ranges:
                progress.update(
                    task,
                    description=f"{inst.symbol} {year_month}",
                )

                if _already_loaded(conn, inst.symbol, year_month):
                    logger.debug("Skipping %s %s (already loaded)", inst.symbol, year_month)
                    progress.advance(task)
                    continue

                candles = _fetch_candles(inst.instrument_key, from_date, to_date)
                new_rows = _insert_candles(conn, inst.symbol, candles)
                _log_backfill(conn, inst.symbol, year_month, new_rows)

                logger.debug(
                    "%s %s: %d candles fetched, %d new rows",
                    inst.symbol,
                    year_month,
                    len(candles),
                    new_rows,
                )
                progress.advance(task)

                # Rate-limit: 200ms between API calls
                time.sleep(0.2)

    # Materialize daily bars
    console.print("\n[bold]Materializing bars_daily from bars_1min...[/bold]")
    daily_count = _materialize_daily(conn)
    console.print(f"[green]bars_daily: {daily_count} rows[/green]")


@app.command()
def main(
    months: int = typer.Option(24, help="Number of months to backfill"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Fetch 1 month of NIFTY only"),
    db_path: str = typer.Option("", help="Override DUCKDB_PATH (empty = use env/default)"),
) -> None:
    """Backfill historical 1-min bars from Upstox into DuckDB."""
    logging.basicConfig(level=logging.INFO)

    conn = get_conn(db_path or None)

    if dry_run:
        from data.universe import by_symbol

        console.print("[bold yellow]DRY RUN: fetching 1 month of NIFTY only[/bold yellow]")
        instruments = [by_symbol("NIFTY")]
        _backfill(conn, instruments, months=1)
    else:
        instruments = list(UNIVERSE)
        _backfill(conn, instruments, months=months)

    # Summary
    total_1min: int = conn.execute("SELECT count(*) FROM bars_1min").fetchone()[0]  # type: ignore[index]
    total_daily: int = conn.execute("SELECT count(*) FROM bars_daily").fetchone()[0]  # type: ignore[index]
    console.print(
        f"\n[bold green]Done.[/bold green] bars_1min={total_1min}, bars_daily={total_daily}"
    )

    if dry_run:
        console.print("\n[bold]First 5 rows (bars_1min):[/bold]")
        for row in conn.execute("SELECT * FROM bars_1min ORDER BY ts_ist ASC LIMIT 5").fetchall():
            console.print(f"  {row}")

        console.print("\n[bold]Last 5 rows (bars_1min):[/bold]")
        for row in conn.execute("SELECT * FROM bars_1min ORDER BY ts_ist DESC LIMIT 5").fetchall():
            console.print(f"  {row}")

    conn.close()


if __name__ == "__main__":
    app()
