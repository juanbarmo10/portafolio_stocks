"""Loader tests: idempotency and the point-in-time key (CLAUDE.md sections 6, 9.4, 9.6)."""

from __future__ import annotations

import pandas as pd
import pytest

from db import loader


@pytest.fixture()
def conn(tmp_path):
    """A fresh SQLite database with the schema applied."""
    connection = loader.init_db(tmp_path / "test.db")
    yield connection
    connection.close()


def _obs(**overrides) -> dict:
    row = {
        "source": "stooq",
        "series_id": "SPY:close_raw",
        "ts": "2026-01-02T00:00:00+00:00",
        "ts_release": "2026-01-02T00:00:00+00:00",
        "value": 512.34,
    }
    row.update(overrides)
    return row


def _count(conn, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_schema_apply_is_idempotent(conn, tmp_path):
    """Re-applying the DDL must not fail: run_ingest applies it on every run."""
    loader.apply_schema(conn)
    loader.apply_schema(conn)
    tables = {
        row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {
        "observations", "companies", "filings", "corporate_actions", "events",
        "trades", "cash_transactions", "thesis_log", "exit_ladder", "alerts_log",
    } <= tables


def test_observations_upsert_is_idempotent(conn):
    """Running the pipeline three times must leave exactly one row (section 6)."""
    df = pd.DataFrame([_obs()])
    for _ in range(3):
        loader.upsert_observations(conn, df)
    assert _count(conn, "observations") == 1


def test_observations_idempotent_when_release_date_unknown(conn):
    """The regression this schema exists to prevent.

    A NULL ts_release inside the composite primary key makes SQLite treat every
    re-ingest as a new row (NULLs are distinct in a unique index) and makes Postgres
    reject the insert. The loader normalizes it to TS_RELEASE_UNKNOWN instead.
    """
    df = pd.DataFrame([_obs(ts_release=None)])
    for _ in range(3):
        loader.upsert_observations(conn, df)
    assert _count(conn, "observations") == 1
    stored = conn.execute("SELECT ts_release FROM observations").fetchone()[0]
    assert stored == loader.TS_RELEASE_UNKNOWN


def test_restatement_keeps_every_version(conn):
    """Same fiscal period reported twice = two rows, not an overwrite (sections 9.4, 9.6).

    A 10-K/A can change a figure already published. Both vintages must survive, or
    point-in-time backtests silently use numbers nobody could have seen at the time.
    """
    original = _obs(
        source="sec", series_id="0000320193:Revenues", ts="2025-09-27",
        ts_release="2025-10-30", value=100.0,
    )
    restated = dict(original, ts_release="2026-02-15", value=97.5)
    loader.upsert_observations(conn, pd.DataFrame([original, restated]))

    rows = conn.execute(
        "SELECT ts_release, value FROM observations "
        "WHERE series_id = '0000320193:Revenues' ORDER BY ts_release"
    ).fetchall()
    assert rows == [("2025-10-30", 100.0), ("2026-02-15", 97.5)]


def test_corrected_value_updates_in_place(conn):
    """A re-fetch of the same key with a corrected value updates rather than duplicates."""
    loader.upsert_observations(conn, pd.DataFrame([_obs(value=512.34)]))
    loader.upsert_observations(conn, pd.DataFrame([_obs(value=512.99)]))
    assert _count(conn, "observations") == 1
    assert conn.execute("SELECT value FROM observations").fetchone()[0] == 512.99


def test_null_value_is_stored_as_null(conn):
    """A missing figure stays visibly NULL — never silently estimated (section 12)."""
    loader.upsert_observations(conn, pd.DataFrame([_obs(value=None)]))
    assert conn.execute("SELECT value FROM observations").fetchone()[0] is None


def test_missing_columns_fail_loudly(conn):
    """A parser that drops a contract column must raise, not write partial rows."""
    df = pd.DataFrame([{"source": "stooq", "series_id": "SPY:close_raw", "ts": "2026-01-02"}])
    with pytest.raises(ValueError, match="missing required columns"):
        loader.upsert_observations(conn, df)


def test_empty_frame_is_a_noop(conn):
    """Nothing to ingest is a valid outcome, not an error."""
    assert loader.upsert_observations(conn, pd.DataFrame(columns=loader.OBSERVATION_COLUMNS)) == 0
    assert _count(conn, "observations") == 0


def test_trades_and_cash_transactions_are_idempotent(conn):
    """Account data re-read from the Flex Query must not duplicate (sections 4.1, 6)."""
    trades = pd.DataFrame([{
        "trade_id": "IBKR-1", "cik": "0000320193", "ticker": "AAPL",
        "ts": "2026-01-05T14:31:00+00:00", "side": "buy", "quantity": 10.0,
        "price": 195.5, "currency": "USD", "commission": 1.0, "fx_rate": None,
    }])
    cash = pd.DataFrame([{
        "tx_id": "IBKR-DIV-1", "cik": "0000320193", "ticker": "AAPL",
        "ts": "2026-02-13T00:00:00+00:00", "kind": "withholding_tax",
        "amount": -0.37, "currency": "USD",
    }])
    for _ in range(2):
        loader.upsert_trades(conn, trades)
        loader.upsert_cash_transactions(conn, cash)
    assert _count(conn, "trades") == 1
    assert _count(conn, "cash_transactions") == 1
    # Unresolved CIK stays NULL rather than being guessed (section 9.3).
    orphan = trades.assign(trade_id="IBKR-2", cik=None, ticker="UNKNOWN")
    loader.upsert_trades(conn, orphan)
    assert conn.execute("SELECT cik FROM trades WHERE trade_id='IBKR-2'").fetchone()[0] is None


def test_thesis_without_invalidation_is_rejected(conn):
    """No thesis is stored without a way to be proven wrong (sections 5.2, 6)."""
    row = {
        "cik": "0000320193", "thesis": "t", "value_accrual": "v",
        "invalidation": "   ", "review_date": "2027-01-01", "created_at": "2026-08-08",
    }
    with pytest.raises(ValueError, match="no invalidation criterion"):
        loader.upsert_thesis_log(conn, [row])
    assert _count(conn, "thesis_log") == 0

    row["invalidation"] = "Gross margin below 38% for two consecutive quarters."
    assert loader.upsert_thesis_log(conn, [row]) == 1


def test_exit_ladder_reserved_word_column(conn):
    """`trigger` is a keyword in both engines; the loader quotes every identifier."""
    rows = [{
        "rule_id": "R1", "cik": "0000320193", "kind": "thesis_invalidation",
        "trigger": "diluted_shares_yoy > 0.05", "action": "review position",
        "created_at": "2026-08-08",
    }]
    for _ in range(2):
        loader.upsert_exit_ladder(conn, rows)
    assert _count(conn, "exit_ladder") == 1
    assert conn.execute('SELECT "trigger" FROM exit_ladder').fetchone()[0] == (
        "diluted_shares_yoy > 0.05"
    )


def test_reference_tables_are_idempotent(conn):
    """Companies, filings, corporate actions and events all re-run cleanly."""
    loader.upsert_companies(conn, [{
        "cik": "0000320193", "ticker": "AAPL", "name": "Apple Inc.",
        "sector": "Information Technology", "thesis_category": "hardware-ecosystem",
        "first_seen": "2026-08-08", "status": "active",
    }] * 1)
    loader.upsert_filings(conn, [{
        "accession": "0000320193-26-000001", "cik": "0000320193", "form": "10-K/A",
        "period_end": "2025-09-27", "filed_date": "2026-02-15", "is_amended": 1, "url": None,
    }])
    loader.upsert_corporate_actions(conn, [{
        "action_id": "AAPL-SPLIT-2020", "cik": "0000320193", "ticker": "AAPL",
        "kind": "split", "ex_date": "2020-08-31", "ratio": 4.0, "amount": None,
        "currency": "USD", "source": "yfinance",
    }])
    loader.upsert_events(conn, [{
        "event_id": "AAPL-EARNINGS-2026Q1", "category": "earnings", "cik": "0000320193",
        "ts": "2026-10-29", "is_estimated": 1, "label": "Q4 FY26 earnings", "payload": None,
    }])
    for table in ("companies", "filings", "corporate_actions", "events"):
        assert _count(conn, table) == 1

    # Re-running the same ingest updates in place.
    loader.upsert_companies(conn, [{
        "cik": "0000320193", "ticker": "AAPL", "name": "Apple Inc.", "sector": None,
        "thesis_category": None, "first_seen": "2026-08-08", "status": "active",
    }])
    assert _count(conn, "companies") == 1
