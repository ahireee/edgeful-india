"""CLI for running backtests against the local DuckDB warehouse.

Usage:
    uv run python scripts/run_backtest.py pdh-breakout --symbol NIFTY --lookback 24
    uv run python scripts/run_backtest.py gap-fill-fade --symbol NIFTY --lookback 24
    uv run python scripts/run_backtest.py orb-continuation --symbol NIFTY --lookback 24

``lookback`` is in months.  The engine pulls all 1-min bars for the symbol,
filters to the last N months, runs the strategy, and reports metrics for
the train/validate/test 60/20/20 split.  A WARN is printed if the test
metrics diverge meaningfully from train -- a coarse overfitting signal.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from backtest.costs import for_symbol as cost_model_for
from backtest.engine import Trade, simulate
from backtest.metrics import Metrics
from backtest.metrics import compute as compute_metrics
from backtest.splits import Split, split_60_20_20
from backtest.strategies import STRATEGIES
from data.db import get_conn
from data.universe import by_symbol

app = typer.Typer(help="Run backtests against the local DuckDB warehouse.")
console = Console()


@app.callback()
def _callback() -> None:
    """Run edgeful-india backtests from the CLI."""


def _load_bars(symbol: str, months: int) -> pl.DataFrame:
    """Load 1-min bars for ``symbol`` covering the last ``months`` months."""
    conn = get_conn()
    cutoff = date.today() - timedelta(days=months * 31)  # generous upper bound
    rows = conn.execute(
        "SELECT symbol, ts_ist, open, high, low, close, volume "
        "FROM bars_1min WHERE symbol = ? AND ts_ist >= ? ORDER BY ts_ist",
        [symbol, cutoff],
    ).fetchall()
    conn.close()
    return pl.DataFrame(
        rows,
        schema=["symbol", "ts_ist", "open", "high", "low", "close", "volume"],
        orient="row",
    )


def _filter_trades_to_split(trades: list[Trade], split: Split) -> list[Trade]:
    return [t for t in trades if split.contains(t.entry_time.date())]


def _print_metrics_table(splits_metrics: list[tuple[Split, Metrics]]) -> None:
    table = Table(title="Backtest Metrics by Split", show_lines=True)
    table.add_column("Metric", style="cyan")
    for split, _ in splits_metrics:
        table.add_column(
            f"{split.name}\n{split.start} → {split.end}",
            justify="right",
        )

    rows: list[tuple[str, list[str]]] = [
        ("Trades", [str(m.total_trades) for _, m in splits_metrics]),
        ("Win rate", [f"{m.win_rate:.1%}" for _, m in splits_metrics]),
        ("Avg win", [f"{m.avg_win_pct:+.2f}%" for _, m in splits_metrics]),
        ("Avg loss", [f"{m.avg_loss_pct:+.2f}%" for _, m in splits_metrics]),
        ("Profit factor", [_fmt_pf(m.profit_factor) for _, m in splits_metrics]),
        ("Total return", [f"{m.total_return_pct:+.2f}%" for _, m in splits_metrics]),
        ("Annualised", [f"{m.annualised_return_pct:+.2f}%" for _, m in splits_metrics]),
        ("Sharpe", [f"{m.sharpe:.2f}" for _, m in splits_metrics]),
        ("Sortino", [f"{m.sortino:.2f}" for _, m in splits_metrics]),
        ("Max DD", [f"{m.max_drawdown_pct:.2f}%" for _, m in splits_metrics]),
        ("Calmar", [f"{m.calmar:.2f}" for _, m in splits_metrics]),
    ]
    for label, values in rows:
        table.add_row(label, *values)
    console.print(table)


def _fmt_pf(pf: float) -> str:
    import math

    if pf == math.inf:
        return "∞"
    return f"{pf:.2f}"


def _check_overfitting(train: Metrics, test: Metrics) -> None:
    """Flag if test metrics decay materially vs. train.

    Heuristic: warn if test win-rate drops by more than 10pp vs train, OR
    if test annualised return is negative while train was positive."""
    warnings: list[str] = []
    if (train.win_rate - test.win_rate) > 0.10:
        warnings.append(
            f"win_rate decay: train {train.win_rate:.1%} → test {test.win_rate:.1%} (drop > 10pp)"
        )
    if train.annualised_return_pct > 0 and test.annualised_return_pct < 0:
        warnings.append(
            f"return flip: train {train.annualised_return_pct:+.2f}% → "
            f"test {test.annualised_return_pct:+.2f}% (positive on train, negative on test)"
        )
    if (train.sharpe - test.sharpe) > 1.0:
        warnings.append(
            f"sharpe decay: train {train.sharpe:.2f} → test {test.sharpe:.2f} (drop > 1.0)"
        )

    if warnings:
        console.print("\n[yellow]⚠ Possible overfitting:[/yellow]")
        for w in warnings:
            console.print(f"  - {w}")
    else:
        console.print(
            "\n[green]✓ Train → test metrics are stable (no obvious overfitting signal).[/green]"
        )


def _run(strategy_name: str, symbol: str, lookback_months: int) -> None:
    if strategy_name not in STRATEGIES:
        console.print(f"[red]Unknown strategy: {strategy_name}[/red]")
        raise typer.Exit(1)

    inst = by_symbol(symbol)
    cost_model = cost_model_for(symbol, is_index=inst.is_index)
    strategy = STRATEGIES[strategy_name]

    console.print(f"Loading {lookback_months}-month bars for [bold]{symbol}[/bold]...")
    bars = _load_bars(symbol, lookback_months)
    if bars.height == 0:
        console.print(f"[red]No bars found for {symbol}.[/red]")
        raise typer.Exit(1)

    console.print(
        f"  {bars.height:,} bars  |  {symbol}  |  cost model "
        f"({'index' if inst.is_index else 'equity'})"
    )

    all_dates = bars.get_column("ts_ist").cast(pl.Date).unique().sort().to_list()
    if len(all_dates) < 5:
        console.print(f"[red]Not enough trading days ({len(all_dates)}) for a split.[/red]")
        raise typer.Exit(1)
    period_start: date = all_dates[0]
    period_end: date = all_dates[-1] + timedelta(days=1)
    train, validate, test = split_60_20_20(period_start, period_end)

    console.print(
        f"  Splits: train {train.start}→{train.end}  |  "
        f"validate {validate.start}→{validate.end}  |  "
        f"test {test.start}→{test.end}\n"
    )

    trades = simulate(bars, strategy, symbol, cost_model)  # type: ignore[arg-type]
    console.print(f"Strategy [bold]{strategy_name}[/bold]: {len(trades)} total trades.")

    splits_metrics: list[tuple[Split, Metrics]] = []
    for split in (train, validate, test):
        sub = _filter_trades_to_split(trades, split)
        splits_metrics.append((split, compute_metrics(sub)))

    _print_metrics_table(splits_metrics)
    _check_overfitting(splits_metrics[0][1], splits_metrics[2][1])


@app.command("pdh-breakout")
def pdh_breakout(
    symbol: Annotated[str, typer.Option(help="Ticker symbol")] = "NIFTY",
    lookback: Annotated[int, typer.Option(help="Lookback in months")] = 24,
) -> None:
    """Run the PDH breakout strategy."""
    _run("pdh_breakout", symbol, lookback)


@app.command("gap-fill-fade")
def gap_fill_fade(
    symbol: Annotated[str, typer.Option(help="Ticker symbol")] = "NIFTY",
    lookback: Annotated[int, typer.Option(help="Lookback in months")] = 24,
) -> None:
    """Run the gap-fill fade strategy."""
    _run("gap_fill_fade", symbol, lookback)


@app.command("orb-continuation")
def orb_continuation(
    symbol: Annotated[str, typer.Option(help="Ticker symbol")] = "NIFTY",
    lookback: Annotated[int, typer.Option(help="Lookback in months")] = 24,
) -> None:
    """Run the ORB continuation strategy."""
    _run("orb_continuation", symbol, lookback)


if __name__ == "__main__":
    app()
