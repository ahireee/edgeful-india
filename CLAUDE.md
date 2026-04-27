# CLAUDE.md

This file is loaded automatically by Claude Code at session start. It defines project conventions, architecture, and current priorities. Read `docs/SPEC.md` for the full product spec and `docs/ROADMAP.md` for build order.

## Project: edgeful-india

A probability-based trading edge platform for Indian markets (NIFTY/BANKNIFTY indices + top 10 NIFTY stocks). Inspired by edgeful.com, but built on Indian market data via the Upstox API. The core idea is **probability stacking** rather than traditional strategy backtesting: pre-compute the historical probabilities of recurring intraday setups (gap fills, opening range breakouts, etc.) over rolling 6-month to 2-year windows, then let the user combine them.

This is both a personal trading tool and a portfolio piece for hedge fund / prop trading internship applications. Code quality, test coverage, and documentation matter as much as features. Every report and backtest must be reproducible from a clean clone.

## Tech stack

- **Python 3.11+**, managed with `uv` (fast, modern; falls back to pip if unavailable)
- **DuckDB** for the local data warehouse — single file, fast OLAP, zero ops
- **Polars** for data manipulation (faster than pandas, lazy evaluation, easier joins)
- **upstox-python-sdk** (Upstox official SDK, v2.21+) for market data. Use the
  Analytics Token (1-year read-only) rather than the daily-OAuth flow whenever
  possible — this is a research project, not a live-trading one, so we don't
  need write scopes.
- **FastAPI** for the report/backtest API layer
- **Streamlit** for the dashboard MVP (swap to Next.js later if it becomes a real product)
- **pytest** + **hypothesis** for tests
- **ruff** for lint/format, **mypy** for type checking

Do not introduce pandas, SQLAlchemy, Django, or Flask without justification. Stick to the stack.

## Repo layout

```
edgeful-india/
  data/         # ingestion, storage, schemas (DuckDB)
  reports/      # one module per report (gap_fill.py, orb.py, ib.py, ...)
  backtest/     # strategy execution engine, equity curves, metrics
  api/          # FastAPI app exposing reports + backtest endpoints
  dashboard/    # Streamlit UI
  scripts/      # one-off CLI utilities (initial data load, daily refresh)
  tests/        # pytest suite, mirrors source layout
  docs/         # SPEC.md, ROADMAP.md, ARCHITECTURE.md, REPORTS.md
```

Each report is a self-contained module with the same interface (see `docs/REPORTS.md`). Adding a new report should not touch any other report's code.

## Conventions

- **Type hints everywhere.** `mypy --strict` should pass.
- **Functions over classes** unless state genuinely needs to live somewhere. The report computation layer is pure functions on DataFrames.
- **No I/O inside report logic.** Reports take a Polars DataFrame in, return a DataFrame + a stats dict out. The data layer fetches; the report layer computes. This is what makes everything testable.
- **Time zones: everything in IST internally.** Convert at the API boundary if needed. Use `zoneinfo`, not `pytz`.
- **Trading calendar matters.** Use `mcal` (`pandas_market_calendars`) for NSE holidays. Never assume Mon–Fri.
- **Money is `Decimal`, prices are `float`.** Tick size for NIFTY futures is 0.05; round all displayed prices.
- **Lookback windows are configurable, default to 6 months.** Per the edgeful philosophy, do not default to long histories — they overfit.

## Current priorities (read ROADMAP.md for the full list)

1. Get historical 1-min bars for NIFTY + BANKNIFTY indices and the 10 stocks loaded into DuckDB via Upstox
2. Build the Gap Fill report end-to-end as the reference implementation
3. Add ORB, IB Breakout, PDH/PDL, Session Bias, Engulfing
4. Wire up FastAPI endpoints + Streamlit dashboard
5. Live screener that runs reports against today's bars

## What "done" looks like for each report

A report is not done until:
- Pure compute function with type hints, in `reports/<name>.py`
- Unit tests with hand-checked fixtures, in `tests/reports/test_<name>.py`
- A CLI entry in `scripts/run_report.py --report <name> --ticker <symbol>`
- A FastAPI endpoint at `/api/reports/<name>`
- A Streamlit page rendering the probability table + filters
- A short writeup in `docs/REPORTS.md` explaining the methodology and any edge cases

## Things to ask the user before doing

- Adding new dependencies
- Designing the database schema for a new entity
- Picking a hosting/deployment approach
- Anything involving real money or live trading (this project is research/backtest only until explicitly told otherwise)

## Things NOT to do

- Do not paper over data quality issues. If a ticker has missing bars, log it loudly and skip rather than forward-filling.
- Do not silently fall back from real Upstox data to synthetic data. If the API is down, fail and tell the user.
- Do not add ML to any report without explicit approval. The whole point of this project is *clean, transparent statistics*. ML can come later.
- Do not commit secrets. `.env` is gitignored; use it for `UPSTOX_CLIENT_ID`, `UPSTOX_CLIENT_SECRET`, `UPSTOX_REDIRECT_URI`, and either `UPSTOX_ACCESS_TOKEN` (daily OAuth) or `UPSTOX_ANALYTICS_TOKEN` (preferred — 1-year read-only).
