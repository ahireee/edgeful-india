# edgeful-india

[![CI](https://github.com/ahireee/edgeful-india/actions/workflows/ci.yml/badge.svg)](https://github.com/ahireee/edgeful-india/actions/workflows/ci.yml)

Probability-based trading edge platform for Indian markets. Pre-computes the historical probabilities of recurring intraday setups (gap fills, opening range breakouts, etc.) on NIFTY/BANKNIFTY and the top 10 NIFTY stocks, using real Upstox API data on rolling 6-month to 2-year windows.

Inspired by [edgeful.com](https://www.edgeful.com) for US futures. There's no equivalent for Indian markets — this is the gap.

## Status

🚧 In active development. See [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Quickstart (once Phase 0 is done)

```bash
git clone https://github.com/yashahire/edgeful-india.git
cd edgeful-india
uv sync
cp .env.example .env  # paste your Upstox Analytics Token
uv run python scripts/backfill.py  # one-time, ~30-45 min for 2 years × 12 symbols
uv run streamlit run dashboard/app.py
```

## Documentation

- [`CLAUDE.md`](CLAUDE.md) — conventions, stack, things-not-to-do (read this first if you're contributing)
- [`docs/SPEC.md`](docs/SPEC.md) — what the product is and isn't
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — phased build order
- [`docs/REPORTS.md`](docs/REPORTS.md) — exact methodology for each report

## Why this exists

Discretionary intraday traders in India trade gap fills, opening range breakouts, and previous-day-high/low breaks every day — but they trade them on vibes, not on data. Indian charting tools show you the chart; algo platforms make you write code. Nobody publishes the simple statistic: *of the last 180 trading days, when NIFTY gapped down 0.3%, what fraction of those gaps got filled?*

This tool answers that question for six categories of setup, on twelve instruments, with proper confidence intervals and a recency check that flags when an edge is decaying.

## License

MIT (planned).
