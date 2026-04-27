# Roadmap

Six phases. Don't skip ahead — each phase produces something working you can demo. Phase 1 is the longest because it sets up everything else.

## Phase 0 — Project bootstrap (2–4 hours)

- [ ] Initialise `uv` project, `pyproject.toml` with the deps from CLAUDE.md
- [ ] Set up `ruff`, `mypy --strict`, `pytest`, `pre-commit`
- [ ] `.gitignore` covering `.env`, `*.duckdb`, `__pycache__`, `.venv`, data dumps
- [ ] `.env.example` listing Upstox env vars (no real values) — see the file for OAuth vs Analytics Token options
- [ ] GitHub Actions: lint + test on push
- [ ] README skeleton with a "what this is" paragraph and screenshot placeholder

## Phase 1 — Data layer (1–2 weeks, this is where most projects die — do it carefully)

### 1A. Auth + client setup (1–2 days)

- [ ] Decide upfront: are you using OAuth (daily refresh) or Analytics Token (1-year)? **Strongly prefer Analytics Token** — generate it once from the developer portal and forget about it.
- [ ] If using OAuth: `scripts/login.py` runs a tiny local Flask/FastAPI server on `localhost:3000`, opens the Upstox auth URL in a browser, captures the redirect's `code` param, exchanges it for an access token, writes it to `.env`. Run this script every morning before market open.
- [ ] If using Analytics Token: just paste the token into `.env` and you're done.
- [ ] `data/upstox_client.py` — thin wrapper that:
  - Loads token from env (Analytics first, falls back to OAuth access token)
  - Configures `upstox_client.Configuration` with the token
  - Exposes `get_history_api()`, `get_market_quote_api()`, `get_intraday_api()` as cached singletons
  - Adds tenacity-based retry on 5xx and 429 (rate limit)
  - Logs every API call with duration to a `data_quality` table for debugging

### 1B. Universe + instrument keys (half a day)

- [ ] Verify the ISINs in `data/universe.py` against current NSE data. ISINs don't change but stock-replacement events do change which 10 names are top-weighted.
- [ ] `data/universe.py` is already populated — just run a sanity check by fetching today's quote for each instrument key.

### 1C. Schema + DuckDB setup (half a day)

- [ ] `data/schema.sql` already drafted. Apply via `duckdb edgeful.duckdb < data/schema.sql`.
- [ ] `data/db.py` — a `get_conn()` helper that returns a DuckDB connection, ensures schema is applied, and handles the "first run" case.

### 1D. Historical backfill (the actual hard part — 3–5 days)

Upstox's 1-min historical endpoint returns **at most 1 month of data per call**, regardless of the from_date you specify. So backfilling 2 years means ~24 calls per symbol × 12 symbols = ~288 calls. Build for this from the start.

- [ ] `scripts/backfill.py` — for each instrument, for each calendar month from N months ago to today, call `HistoryV3Api.get_historical_candle_data1(instrument_key, "minutes", "1", to_date, from_date)`.
- [ ] After each call, dedupe by `(symbol, ts_ist)` and insert into `bars_1min`. Use DuckDB's `INSERT OR IGNORE` semantics (or `INSERT ... ON CONFLICT DO NOTHING`).
- [ ] Idempotent: re-running should be a no-op for already-loaded months. Track per-(symbol, year_month) loaded state in a `backfill_log` table or just check existing row counts.
- [ ] Rate limit: sleep 200ms between calls. Upstox's exact published limits are vague but conservative pacing avoids 429s.
- [ ] After all 1-min bars are in, run an aggregation query that materializes `bars_daily` from `bars_1min`. Don't trust Upstox's daily endpoint — derive it yourself so it's always consistent with the intraday data the reports use.

### 1E. Daily refresh (half a day)

- [ ] `scripts/daily_refresh.py` — runs after market close. Pulls today's 1-min bars (one call per symbol since today fits in <1 month), inserts, regenerates today's `bars_daily` row, runs the data-quality check.
- [ ] Add a cron entry or a GitHub Action on schedule (15:45 IST = 10:15 UTC).

### 1F. Calendar + data quality (1 day)

- [ ] `data/calendar.py` — NSE trading calendar wrapper around `pandas_market_calendars`. Used to compute "expected_bars" for any given date (typically 375 = 9:15 to 15:30 inclusive).
- [ ] `data/quality.py` — for each (symbol, trade_date) computes expected vs actual, identifies gaps (consecutive missing minutes), writes to `data_quality`. Flag any day where `actual_bars / expected_bars < 0.95`.
- [ ] Tests using a fixed CSV fixture (don't hit the live API in tests).

## Phase 2 — First report end-to-end: Gap Fill (3–5 days)

This is the reference implementation. Get this right and the next five are mechanical.

- [ ] `reports/base.py` — defines the report interface:
  ```python
  class ReportResult(TypedDict):
      buckets: pl.DataFrame   # the probability table
      summary: dict           # headline stats
      methodology: str        # human-readable description
  ```
- [ ] `reports/gap_fill.py` — pure function `compute(bars: pl.DataFrame, **params) -> ReportResult`
- [ ] Hand-checked test fixtures: 20–30 days of NIFTY data with known gaps and known fills, asserting the report returns the right counts
- [ ] `scripts/run_report.py --report gap_fill --symbol NIFTY --lookback 180`
- [ ] `docs/REPORTS.md` entry: definition of "gap," fill criteria (touches prev close vs closes through it), edge cases (no overnight gap = excluded, holiday-adjacent days = flagged)

## Phase 3 — Remaining five reports (1 per 2–3 days)

In order: ORB → IB Breakout → PDH/PDL → Session Bias → Engulfing. Each follows the Phase 2 pattern. Refactor `reports/base.py` only when the second or third report makes a clear pattern obvious — don't pre-abstract.

## Phase 4 — API + dashboard (1 week)

- [ ] FastAPI app in `api/main.py` with `/api/reports/{name}?symbol=...&lookback=...` endpoints
- [ ] Streamlit dashboard in `dashboard/app.py`:
  - Sidebar: symbol picker, lookback slider, report multiselect
  - Main: probability table + bar chart per report
  - Stack page: pick 2–3 reports and see joint probability
- [ ] Deploy: Streamlit Community Cloud or Railway. Public URL goes in README.

## Phase 5 — Backtest engine (1–2 weeks)

- [ ] `backtest/engine.py` — bar replay loop, position state, equity curve
- [ ] `backtest/costs.py` — NSE STT, brokerage (configurable; default Zerodha), slippage in ticks
- [ ] `backtest/metrics.py` — Sharpe, Sortino, max DD, Calmar, win rate, profit factor (you've already written most of this for the options engine — port it)
- [ ] `backtest/splits.py` — 60/20/20 time-series split
- [ ] One end-to-end backtest: ORB + bullish session bias on NIFTY futures, last 12 months, equity curve in the dashboard

## Phase 6 — Live screener + writeup (1 week)

- [ ] `scripts/screener.py` — runs every N minutes during market hours (9:15–15:30 IST), computes which reports are firing right now, writes to a `live_signals` table
- [ ] Dashboard "Today" page that polls `live_signals`
- [ ] LinkedIn/blog writeup, ~1500 words: motivation, methodology, three things that surprised you about NIFTY's intraday structure, link to repo and live demo
- [ ] Pin the post to your LinkedIn, link from your resume

## Stretch (post-internship-applications)

- Options-specific extension reusing your existing options engine
- Multi-user version with auth
- Mobile-friendly UI
- Push notifications when a stacked setup fires
