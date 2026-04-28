"""Fetch historical India VIX daily OHLC from Yahoo Finance and save to CSV.

Usage:
    uv run python scripts/fetch_india_vix.py
    uv run python scripts/fetch_india_vix.py --start 2024-04-01 --end 2026-04-28

Source is Yahoo Finance ticker ``^INDIAVIX``. NSE's own API sits behind
Akamai bot-manager and rejects non-browser TLS fingerprints, so for a
research project we pull from Yahoo instead — same series, no auth.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
import yfinance as yf

logger = logging.getLogger(__name__)

app = typer.Typer(help="Fetch India VIX historical daily OHLC from Yahoo Finance.")

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "raw" / "india_vix.csv"
TICKER = "^INDIAVIX"


@app.command()
def main(
    start: Annotated[str, typer.Option("--start", help="YYYY-MM-DD inclusive")] = "2024-04-01",
    end: Annotated[
        str, typer.Option("--end", help="YYYY-MM-DD exclusive (Yahoo convention)")
    ] = date.today().isoformat(),
    out: Annotated[Path, typer.Option("--out")] = OUT_PATH,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("downloading %s daily bars %s -> %s from Yahoo Finance", TICKER, start, end)
    raw = yf.download(
        TICKER,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise SystemExit(f"yfinance returned no data for {TICKER} {start}..{end}")

    # yfinance returns a MultiIndex column frame (("Open", "^INDIAVIX"), ...).
    # Flatten to just the field names.
    if hasattr(raw.columns, "nlevels") and raw.columns.nlevels > 1:
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()  # Date out of the index
    df = pl.from_pandas(raw[["Date", "Open", "High", "Low", "Close"]])
    df = df.rename(
        {
            "Date": "trade_date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
        }
    ).with_columns(pl.col("trade_date").cast(pl.Date))
    df = df.drop_nulls().sort("trade_date").unique(subset=["trade_date"], keep="first")

    df.write_csv(out)
    logger.info("wrote %d rows -> %s", df.height, out)
    typer.echo(f"\nrows: {df.height}")
    typer.echo(f"\nfirst 5:\n{df.head(5)}")
    typer.echo(f"\nlast 5:\n{df.tail(5)}")


if __name__ == "__main__":
    app()
