-- edgeful-india DuckDB schema
-- Idempotent: safe to run multiple times.
-- Apply with: duckdb edgeful.duckdb < data/schema.sql

CREATE TABLE IF NOT EXISTS bars_1min (
    symbol      VARCHAR    NOT NULL,
    ts_ist      TIMESTAMP  NOT NULL,  -- IST, naive (we manage tz at the boundary)
    open        DOUBLE     NOT NULL,
    high        DOUBLE     NOT NULL,
    low         DOUBLE     NOT NULL,
    close       DOUBLE     NOT NULL,
    volume      BIGINT     NOT NULL,
    PRIMARY KEY (symbol, ts_ist)
);

CREATE INDEX IF NOT EXISTS idx_bars_1min_symbol_date
    ON bars_1min (symbol, CAST(ts_ist AS DATE));

-- Daily bars derived from 1-min, materialised for fast PDH/PDL lookups
CREATE TABLE IF NOT EXISTS bars_daily (
    symbol      VARCHAR    NOT NULL,
    trade_date  DATE       NOT NULL,
    open        DOUBLE     NOT NULL,
    high        DOUBLE     NOT NULL,
    low         DOUBLE     NOT NULL,
    close       DOUBLE     NOT NULL,
    volume      BIGINT     NOT NULL,
    bar_count   INTEGER    NOT NULL,  -- number of 1-min bars that fed this day
    PRIMARY KEY (symbol, trade_date)
);

-- Tracks which (symbol, month) combos have been backfilled, for idempotency
CREATE TABLE IF NOT EXISTS backfill_log (
    symbol        VARCHAR   NOT NULL,
    year_month    VARCHAR   NOT NULL,  -- 'YYYY-MM'
    completed_at  TIMESTAMP NOT NULL,
    row_count     INTEGER   NOT NULL,
    PRIMARY KEY (symbol, year_month)
);

CREATE TABLE IF NOT EXISTS nse_holidays (
    holiday_date DATE       NOT NULL PRIMARY KEY,
    name         VARCHAR    NOT NULL
);

CREATE TABLE IF NOT EXISTS data_quality (
    symbol         VARCHAR    NOT NULL,
    trade_date     DATE       NOT NULL,
    expected_bars  INTEGER    NOT NULL,  -- typically 375 for NSE: 09:15-15:30
    actual_bars    INTEGER    NOT NULL,
    gaps_json      VARCHAR,              -- JSON array of [start_ts, end_ts] pairs
    flagged        BOOLEAN    NOT NULL DEFAULT FALSE,
    PRIMARY KEY (symbol, trade_date)
);

-- Live signals from the screener (Phase 6)
CREATE TABLE IF NOT EXISTS live_signals (
    ts_ist       TIMESTAMP  NOT NULL,
    symbol       VARCHAR    NOT NULL,
    report_name  VARCHAR    NOT NULL,
    direction    VARCHAR,                  -- 'up' / 'down' / NULL for two-sided
    probability  DOUBLE     NOT NULL,
    sample_size  INTEGER    NOT NULL,
    payload_json VARCHAR,                  -- report-specific extras
    PRIMARY KEY (ts_ist, symbol, report_name)
);
