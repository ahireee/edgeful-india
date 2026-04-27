"""Smoke test: fetch the latest NIFTY 50 quote via the Upstox API and print it."""

from __future__ import annotations

import sys

from rich import print as rprint


def main() -> int:
    # Import here so module-level errors are caught by the try/except.
    from data.upstox_client import get_market_quote_api, upstox_retry

    instrument_key = "NSE_INDEX|Nifty 50"

    @upstox_retry
    def fetch_ltp() -> object:
        api = get_market_quote_api()
        return api.get_ltp(instrument_key=instrument_key)

    try:
        response = fetch_ltp()
        rprint("[bold green]Connection OK[/bold green]")
        rprint(response)
        return 0
    except Exception as exc:
        rprint(f"[bold red]Connection FAILED:[/bold red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
