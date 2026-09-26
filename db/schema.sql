-- equitydash schema (CLAUDE.md section 6).
-- Long format: adding a new series never requires a schema migration.
-- Portable SQLite (phases 0-4) <-> PostgreSQL (phase 5): timestamps are ISO8601 UTC TEXT,
-- no SQLite-only types are used, and every statement is IF NOT EXISTS (idempotent apply).

PRAGMA foreign_keys = ON;

-- One row per (source, series_id, reference date, publication date) observation.
--
-- The primary key includes ts_release ON PURPOSE (section 6): one fiscal quarter can be
-- reported several times (original 10-Q, then a 10-K/A restatement). All versions are kept;
-- overwriting would destroy honest point-in-time backtests (sections 9.4, 9.6).
--
-- WHY ts_release IS `NOT NULL DEFAULT ''` AND NOT NULLABLE:
--   A NULL inside a composite PRIMARY KEY silently breaks idempotency. SQLite treats every
--   NULL as distinct in a unique index, so `ON CONFLICT DO UPDATE` never matches and three
--   identical re-ingests write three rows; PostgreSQL rejects a NULL PK column outright.
--   Both failures contradict the mandatory idempotency rule (section 6), so "publication
--   date unknown / not applicable" is encoded as the empty string (db.loader.TS_RELEASE_UNKNOWN).
--   Read it back with NULLIF(ts_release, '') when SQL-level NULL semantics are wanted.
CREATE TABLE IF NOT EXISTS observations (
    source      TEXT NOT NULL,              -- 'fred' | 'sec' | 'yfinance' | 'finra' | 'ibkr' | 'equitydash' (derived) | local fx
    series_id   TEXT NOT NULL,              -- 'CPIAUCSL' | '0000320193:Revenues' | 'SPY:close_raw'
    ts          TEXT NOT NULL,              -- ISO8601 UTC — REFERENCE date of the datum
    ts_release  TEXT NOT NULL DEFAULT '',   -- ISO8601 UTC — PUBLICATION date ('' = unknown)
    value       REAL,                       -- NULL is a legitimate "not reported" (section 12)
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (source, series_id, ts, ts_release)
);
CREATE INDEX IF NOT EXISTS idx_obs_series_ts ON observations(series_id, ts DESC);
-- Point-in-time reads filter by ts_release <= simulated date (section 9.4), never by ts.
CREATE INDEX IF NOT EXISTS idx_obs_series_release ON observations(series_id, ts_release);

-- Company registry. The CIK is the stable key; the ticker is a mutable attribute
-- that gets reassigned across companies (section 9.3: FB -> META, recycled tickers).
CREATE TABLE IF NOT EXISTS companies (
    cik             TEXT PRIMARY KEY,
    ticker          TEXT NOT NULL,          -- current ticker, mutable
    name            TEXT,
    sector          TEXT,                   -- GICS or equivalent
    thesis_category TEXT,                   -- the axis that actually matters (section 5.3)
    first_seen      TEXT,
    status          TEXT,                   -- 'active'|'delisted'|'acquired'|'merged'
    sic             TEXT,                   -- SEC industry code: NOT GICS, never in `sector`
    sic_description TEXT
);
CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies(ticker);

-- Reorganizations IBKR reported and the panel could not classify (section 9.2). Until
-- 2026-09-25 they only reached the log and Telegram; the portfolio page could not flag the
-- position. Account data: never in the public copy.
CREATE TABLE IF NOT EXISTS unmapped_actions (
    action_id   TEXT PRIMARY KEY,           -- 'ibkr:<transactionID>'
    ticker      TEXT,
    conid       TEXT,
    ex_date     TEXT,
    code        TEXT,                       -- the reorg code IBKR used
    description TEXT,                       -- IBKR's own text, verbatim
    source      TEXT NOT NULL,
    first_seen  TEXT,
    ingested_at TEXT NOT NULL
);

-- SEC filing history: source of ts_release and of the amendment flag (sections 4.4, 9.6).
CREATE TABLE IF NOT EXISTS filings (
    accession   TEXT PRIMARY KEY,
    cik         TEXT NOT NULL,
    form        TEXT NOT NULL,              -- '10-K' | '10-Q' | '8-K' | 'S-1' ...
    period_end  TEXT,
    filed_date  TEXT NOT NULL,
    is_amended  INTEGER DEFAULT 0,          -- 10-K/A, 10-Q/A -> possible restatement (9.6)
    url         TEXT,
    items       TEXT                        -- 8-K item codes, e.g. '2.02,9.01' (2026-09-25)
);
CREATE INDEX IF NOT EXISTS idx_filings_cik_filed ON filings(cik, filed_date DESC);

-- Corporate actions stored SEPARATELY from raw prices (section 9.1). The adjusted series
-- is rebuilt on demand for a given as-of date, applying only ex_date <= that date; storing
-- a pre-adjusted close would bake future information into every past point.
--
-- WHY cik IS NULLABLE HERE, unlike section 6's DDL: the level-1/2 market references are
-- ETFs (SPY, RSP, the sector SPDRs), which have no company CIK to resolve to, and section
-- 9.3 forbids guessing one. `trades` and `cash_transactions` already pair a nullable cik
-- with a NOT NULL ticker for the same reason; this table now matches its siblings.
CREATE TABLE IF NOT EXISTS corporate_actions (
    action_id   TEXT PRIMARY KEY,
    cik         TEXT,                       -- NULL for ETFs and unresolved tickers (9.3)
    ticker      TEXT NOT NULL,
    kind        TEXT NOT NULL,              -- 'split'|'dividend'|'spinoff'|'merger'|'delisting'
    ex_date     TEXT NOT NULL,
    ratio       REAL,                       -- splits
    amount      REAL,                       -- dividends, per share
    currency    TEXT,
    source      TEXT NOT NULL               -- which provider said so (section 9.8)
);
CREATE INDEX IF NOT EXISTS idx_corpact_ticker_ex ON corporate_actions(ticker, ex_date);

-- Dated-event calendar: macro releases, earnings dates, FOMC, catalysts.
CREATE TABLE IF NOT EXISTS events (
    event_id     TEXT PRIMARY KEY,
    category     TEXT,                      -- 'macro'|'earnings'|'fomc'|'lockup'|'catalyst'
    cik          TEXT,                      -- NULL for macro events
    ts           TEXT NOT NULL,
    is_estimated INTEGER DEFAULT 0,         -- earnings date confirmed vs. estimated
    label        TEXT,
    payload      TEXT                       -- JSON
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

-- Security master, from the Flex Query SecuritiesInfo section. One row per instrument the
-- account has ever touched.
--
-- WHY conid IS THE KEY (section 9.3): the ticker is not one. It gets reassigned between
-- companies and renamed under the same company (FB -> META), so a table keyed on it cannot
-- describe history. `conid` is IBKR's own permanent contract identifier, and the ISIN,
-- CUSIP and FIGI stored next to it are the industry-standard identifiers that survive a
-- rename as well. This is the only stable key the ACCOUNT side of the project has; the SEC
-- side keys on CIK, and the two meet here when the ticker resolves.
--
-- `issuer_country` earns its place on its own: it is the field that determines the SITUS of
-- an asset, which section 11 needs to show US-situs exposure as a question for an adviser.
-- It exists nowhere else in the statement. Nothing is computed from it here (section 11:
-- Claude asserts no tax treatment).
--
-- first_seen / last_seen make a rename visible instead of silently overwriting it: the row
-- keeps the current ticker, and the dates say when this mapping was observed.
CREATE TABLE IF NOT EXISTS securities (
    conid            TEXT PRIMARY KEY,      -- IBKR permanent contract id (section 9.3)
    ticker           TEXT NOT NULL,         -- current symbol, mutable
    cik              TEXT,                  -- NULL when unresolved; never guessed (9.3)
    name             TEXT,
    isin             TEXT,
    cusip            TEXT,
    figi             TEXT,
    asset_category   TEXT,                  -- 'STK' | 'ETF' | ...
    sub_category     TEXT,                  -- 'COMMON' | ...
    listing_exchange TEXT,
    issuer_country   TEXT,                  -- situs of the asset (section 11)
    currency         TEXT,
    multiplier       REAL,
    first_seen       TEXT,
    last_seen        TEXT,
    source           TEXT NOT NULL,
    ingested_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_securities_ticker ON securities(ticker);
CREATE INDEX IF NOT EXISTS idx_securities_cik ON securities(cik);

-- Index membership as intervals: who was in the universe, from when until when (section 9.5).
-- Exists for market breadth, which needs to know the members ON EACH DATE — today's list
-- applied to the past is survivorship bias, worst exactly in the crises breadth should see.
--
-- Intervals rather than daily snapshots: ~1.5k rows instead of ~1.4M, and a ticker that
-- leaves and later re-enters keeps both intervals (the source's README warns this happens).
-- end_date is EXCLUSIVE: the first date the ticker was no longer a member. NULL = current.
--
-- `first_seen` is when THIS project first learned of the interval, and is never overwritten.
-- The historical list is a reconstruction published long after the fact, so every row
-- backfilled from it says so; rows observed going forward carry the date they were seen.
CREATE TABLE IF NOT EXISTS universe_membership (
    universe    TEXT NOT NULL,              -- 'sp500'
    ticker      TEXT NOT NULL,              -- as the source spells it: BRK.B, not BRK-B
    start_date  TEXT NOT NULL,
    end_date    TEXT,                       -- exclusive; NULL while still a member
    source      TEXT NOT NULL,              -- provenance of the interval (section 9.8)
    first_seen  TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (universe, ticker, start_date)
);
CREATE INDEX IF NOT EXISTS idx_membership_dates ON universe_membership(universe, start_date, end_date);

-- Executed trades, read from the IBKR Activity Flex Query — never written by hand.
-- cik stays NULL with a warning when the ticker does not resolve; never guessed (9.3).
CREATE TABLE IF NOT EXISTS trades (
    trade_id    TEXT PRIMARY KEY,           -- IBKR tradeID (idempotency key)
    conid       TEXT,                       -- -> securities.conid, the stable key (9.3)
    cik         TEXT,
    ticker      TEXT NOT NULL,
    ts          TEXT NOT NULL,              -- ISO8601 UTC execution time
    side        TEXT NOT NULL,              -- 'buy' | 'sell'
    quantity    REAL NOT NULL,
    price       REAL NOT NULL,
    currency    TEXT NOT NULL,
    commission  REAL,
    fx_rate     REAL,                       -- to USD when applicable (section 9.9)
    ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_ticker_ts ON trades(ticker, ts);

-- Dividends, withholding, interest and fees, from the Flex Query CashTransactions section.
-- Withholding is the OBSERVED figure; never a assumed treaty rate (section 11).
CREATE TABLE IF NOT EXISTS cash_transactions (
    tx_id       TEXT PRIMARY KEY,
    conid       TEXT,                       -- -> securities.conid, the stable key (9.3)
    cik         TEXT,
    ticker      TEXT,
    ts          TEXT NOT NULL,
    kind        TEXT NOT NULL,              -- 'dividend'|'withholding_tax'|'interest'|'fee'
    amount      REAL NOT NULL,
    currency    TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cashtx_ticker_ts ON cash_transactions(ticker, ts);

-- Written thesis per company. invalidation is NOT NULL by design (sections 5.2, 6):
-- a thesis with no way to be proven wrong is not stored.
CREATE TABLE IF NOT EXISTS thesis_log (
    cik           TEXT PRIMARY KEY,
    thesis        TEXT NOT NULL,
    value_accrual TEXT NOT NULL,
    invalidation  TEXT NOT NULL,            -- *** required at schema level ***
    review_date   TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

-- Exit rules written BEFORE buying (section 2, level 4).
-- "trigger" is quoted throughout: it is a keyword in both engines.
CREATE TABLE IF NOT EXISTS exit_ladder (
    rule_id     TEXT PRIMARY KEY,
    cik         TEXT NOT NULL,
    kind        TEXT NOT NULL,              -- 'take_profit'|'thesis_invalidation'|'rebalance'
    "trigger"   TEXT NOT NULL,
    action      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

-- Fired-alert ledger; used to suppress duplicate alerts for the same event (section 8, phase 4).
CREATE TABLE IF NOT EXISTS alerts_log (
    alert_id    TEXT PRIMARY KEY,
    rule_id     TEXT NOT NULL,
    fired_at    TEXT NOT NULL,
    payload     TEXT
);
