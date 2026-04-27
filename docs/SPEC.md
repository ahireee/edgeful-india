# edgeful-india — Product Spec

## What this is

An "edgeful for Indian markets." A web tool where you pick an asset (NIFTY futures, BANKNIFTY futures, or one of the top 10 NIFTY stocks), pick a setup (gap fill, opening range breakout, etc.), and see the historical probability that setup played out — based on the last 6 months to 2 years of real Indian market data.

## Why it doesn't exist yet

The US futures market has edgeful, BuildAlpha, Quantified Strategies, TraderEdge, etc. The Indian market has charting tools (TradingView, Sensibull) and full-blown algo platforms (AlgoTest, Streak), but nothing in the middle: a clean, focused probability layer for intraday setups on Indian instruments. NIFTY and BANKNIFTY are among the world's most-traded index derivatives by volume, so the data is there — nobody has packaged it this way.

## Target user (v1)

Yourself, plus discretionary intraday traders in India who want statistical backing for the setups they already trade. Long-term: Indian prop traders and retail quants.

## The core loop

1. User opens the dashboard, picks `NIFTY` from the asset dropdown.
2. Picks `Gap Fill` from the report library.
3. Sees a table:
   - Gap size bucket | Instances (last 6mo) | Fill rate | Avg time to fill
   - 0.0% – 0.1%   | 87                    | 92%       | 38 min
   - 0.1% – 0.25%  | 64                    | 81%       | 1h 12min
   - ...
4. Filters by gap direction (up/down), day of week, weekday vs expiry day.
5. Optionally stacks a second report (e.g. ORB direction) and sees the joint probability when both setups align.
6. Hits "backtest this combination" and gets an equity curve, trade log, win rate, and drawdown stats.

## v1 reports (in build order)

1. **Gap Fill** — overnight gap between previous close and today's open. What fraction of gaps of size X get filled intraday? Filterable by gap size bucket, direction, and day type.
2. **Opening Range Breakout (ORB)** — define the opening range as the first 15 minutes. What fraction of days break that range, and of those, what fraction continue in the breakout direction by EOD?
3. **Initial Balance Breakout (IB)** — first hour's range. Same logic as ORB but with a wider window. Used heavily by prop firm traders.
4. **Previous Day High/Low (PDH/PDL) breaks** — when today breaks yesterday's high (or low), does it tend to continue (breakout) or revert (false break)?
5. **Session Bias** — given the open price relative to yesterday's close, what's the probability the session closes green vs red?
6. **Engulfing Candle Reversals** — on a chosen timeframe (5m / 15m / 1h), how often does a bullish/bearish engulfing candle actually mark a reversal?

Each report ships with a configurable lookback (default 180 trading days). Each shows: instance count, raw probability, 95% Wilson confidence interval, and a recency check (last 30 days vs full window — flags if the edge has decayed).

## Probability stacking (v1.5)

A "stack" page where you select 2–3 reports and the system computes the joint probability over the same historical window — i.e. the fraction of days where Setup A's signal *and* Setup B's signal *and* Setup C's signal all fired in the same direction, and what happened next. This is the headline differentiator vs single-strategy backtesters.

## Backtest engine (v2)

A small bar-replay engine that takes a stacked setup, generates entry signals from it, applies a simple exit rule (fixed RR, EOD, or ATR stop), and produces an equity curve with proper transaction costs (NSE STT, brokerage, slippage). 60/20/20 in-sample / out-of-sample / forward split — the same discipline already in use in the existing NIFTY options engine.

## Live screener (v2.5)

Runs all reports against the current trading day's bars and flags which setups are firing right now, across all tracked tickers. This is the feature that turns the tool from a research toy into something usable during market hours.

## Out of scope for v1

- Order execution / live trading
- Options-specific reports (already covered by the existing options engine)
- Multi-day swing setups
- Anything ML-based (resist the temptation; explainable stats first)
- Multi-user accounts, billing, auth — single-user local install only

## Success criteria

**For self-use:** You consult it before placing intraday trades and can point to specific times the probability data made you skip a bad trade or take a setup you'd otherwise have missed.

**For recruiters:** Live demo URL, public GitHub repo with clean commit history, README with screenshots, and a writeup post (LinkedIn or personal blog) explaining the methodology — emphasising rolling windows, confidence intervals, and the no-overfitting philosophy. The portfolio narrative is: *"I noticed Indian traders had no equivalent to edgeful, so I built one on Upstox's API. Here's how it works and what I learned about NIFTY's intraday structure."*
