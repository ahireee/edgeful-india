# Reports — Methodology

Every report shares the same interface and the same statistical framing. This file pins down the *exact* definitions so two engineers (or two Claude sessions) implement them the same way.

## Shared rules

- **Lookback default: 180 trading days.** Configurable per call. Never default to more than 500 trading days.
- **Bar source:** 1-min bars from `bars_1min`, aggregated as needed. Daily bars (`bars_daily`) are derived, never trusted as a separate source.
- **Session window:** 09:15:00 IST to 15:30:00 IST. Bars outside this window are excluded.
- **Excluded days:** holidays (per `nse_holidays`), days with > 5% missing 1-min bars (per `data_quality`), and the day after any data-quality flag.
- **Confidence intervals:** every probability is reported with a 95% Wilson score interval. This matters more for low-instance buckets.
- **Recency check:** alongside the headline probability over the full lookback, also compute the same probability over the last 30 trading days. If they differ by > 15 percentage points, flag the row as "edge decaying" or "edge strengthening."
- **All filters compose.** A request like "gap fill, gap size 0.1–0.25%, downward gaps only, last 90 days, Mondays only" should work as a single query.

---

## 1. Gap Fill

**Definition.** A gap is the difference between today's first traded price (the open of the 09:15 IST bar) and yesterday's last traded price (the close of the 15:30 IST bar). Gap size is `(today_open - prev_close) / prev_close * 100`, signed.

**Fill criteria.** A gap is "filled" if at any point during today's session, the price *touches* yesterday's close (i.e. for an up-gap, today's intraday low ≤ prev_close; for a down-gap, today's intraday high ≥ prev_close). Touch, not close-through.

**Buckets.** By absolute gap size: 0.0–0.1%, 0.1–0.25%, 0.25–0.5%, 0.5–1.0%, 1.0–2.0%, 2.0%+. Reported separately for up-gaps and down-gaps.

**Output columns.** bucket, direction, instances, fill_rate, fill_rate_ci_low, fill_rate_ci_high, avg_minutes_to_fill, median_minutes_to_fill, recent_30d_fill_rate.

**Edge cases.** If `|gap| < 1 tick`, exclude (not a real gap). If yesterday was a holiday-adjacent short session, flag but include.

---

## 2. Opening Range Breakout (ORB)

**Definition.** The opening range is the high and low of the first N minutes after market open. Default N = 15. The range is locked at 09:30 IST.

**Breakout criteria.** A breakout occurs when, after 09:30, price trades > opening_range_high (upside breakout) or < opening_range_low (downside breakout). Whichever side breaks first wins; if both break in the same minute, exclude the day.

**Continuation criteria.** Given a breakout direction, the day is a "continuation" if the close at 15:30 is in the breakout direction relative to the breakout level (above for upside, below for downside). A day with a breakout that reverses by EOD is a "false break."

**Output columns.** breakout_direction, instances, breakout_rate (% of days that have any breakout), continuation_rate (% of breakouts that continue), avg_continuation_size_pct, false_break_rate, recent_30d_continuation_rate.

**Filters.** Opening range window (5/15/30/60 min), day of week, expiry day flag.

---

## 3. Initial Balance Breakout (IB)

**Definition.** Same logic as ORB but the window is the first 60 minutes (09:15 to 10:15 IST). The "Initial Balance" is auction-market-theory terminology and is the standard prop-firm setup.

**Why separate from ORB.** Different statistical character. The 15-min ORB captures momentum; the 60-min IB captures a more developed range. Both belong in the report library because traders use them differently.

**Output columns.** Same as ORB.

---

## 4. Previous Day High / Low (PDH / PDL) Breaks

**Definition.** PDH = yesterday's session high. PDL = yesterday's session low. A "break" occurs when today's price trades above PDH or below PDL.

**Continuation criteria.** Same EOD-relative-to-break-level rule as ORB. Distinguish:
- Break of PDH that closes above PDH (continuation up)
- Break of PDH that closes below PDH (false break / reversal)
- Break of PDL that closes below PDL (continuation down)
- Break of PDL that closes above PDL (false break / reversal)

**Output columns.** level (PDH or PDL), break_rate, continuation_rate, false_break_rate, avg_continuation_pct, recent_30d.

---

## 5. Session Bias

**Definition.** Given today's open relative to yesterday's close, what's the probability today closes green (close > open) vs red (close < open)?

**Buckets.** By gap-at-open size (same buckets as Gap Fill) AND by gap direction. So six buckets × two directions = twelve rows per symbol.

**Output columns.** open_vs_prev_close_bucket, gap_direction, instances, p_green_close, p_red_close, avg_intraday_range_pct, recent_30d_p_green.

**Why this is interesting.** Conventional wisdom says big up-gaps fade. The report tells you whether that's actually true for NIFTY in the current regime — and the answer changes by year.

---

## 6. Engulfing Candle Reversals

**Definition.** A bullish engulfing candle on timeframe T is a candle whose body fully engulfs the prior candle's body, and which closes higher than it opens, after a downtrend (defined as: the prior 3 closes form a non-increasing sequence). Bearish engulfing is the mirror.

**Reversal criteria.** A bullish engulfing is a "successful reversal" if, within the next K candles on the same timeframe, price trades > engulfing candle's high. K default = 3.

**Timeframes.** 5m, 15m, 1h. Run separately, report separately.

**Output columns.** timeframe, type (bullish/bearish), instances, reversal_rate, avg_size_to_target, recent_30d_reversal_rate.

**Edge cases.** Engulfing candles in the first 15 minutes after open are excluded (opening-range noise). Engulfing candles after 15:00 IST are excluded (EOD noise).

---

## Probability stacking — joint reports

When the user stacks reports, the system finds all historical days where every selected report's signal fired in the same direction, and computes:

- Joint instance count
- Joint forward return distribution (next 30 / 60 / EOD minutes)
- Conditional probability vs each report's standalone probability (so the user can see whether stacking actually helps)

If joint instance count < 20, refuse to render and tell the user the sample is too small. Sample-size discipline is the whole point.
