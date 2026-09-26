"""Idempotent database loader (CLAUDE.md sections 6, 10).

Source-agnostic: every ingest module produces a DataFrame with the columns
``[source, series_id, ts, ts_release, value]`` and this loader upserts it. Every write
uses ``INSERT ... ON CONFLICT DO UPDATE``, so the pipeline can be re-run any number of
times without duplicating rows — idempotency is mandatory (section 6), not best-effort.

Inputs (read):
    - db/schema.sql : DDL applied on connect (idempotent, IF NOT EXISTS).
    - pandas DataFrames / dict rows from ingest modules.

Outputs (write):
    - The database at Settings.db_path (or DATABASE_URL when set).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from core.logging_setup import get_logger
from db.database import add_missing_columns, open_connection

log = get_logger(__name__)

SCHEMA_PATH: Path = Path(__file__).resolve().parent / "schema.sql"

# Sentinel for "publication date unknown / not applicable".
#
# ts_release is part of the observations primary key (section 6) so that a restated
# quarter keeps every version. A NULL there would silently break idempotency: SQLite
# treats each NULL as distinct in a unique index (ON CONFLICT never matches, so
# re-ingesting duplicates), and PostgreSQL rejects NULL in a PK column. The empty
# string is a real, comparable value on both backends. Read back with
# NULLIF(ts_release, '') when SQL NULL semantics are wanted.
TS_RELEASE_UNKNOWN = ""

# The canonical column contract for observation DataFrames (section 10).
OBSERVATION_COLUMNS = ["source", "series_id", "ts", "ts_release", "value"]

# Per-table contract: (table, primary-key columns, all columns, numeric columns,
# whether the table carries an ingested_at stamp). Mirrors db/schema.sql.
COMPANY_COLUMNS = ["cik", "ticker", "name", "sector", "thesis_category", "first_seen", "status",
                   "sic", "sic_description"]
FILING_COLUMNS = ["accession", "cik", "form", "period_end", "filed_date", "is_amended", "url",
                  "items"]
CORPORATE_ACTION_COLUMNS = [
    "action_id", "cik", "ticker", "kind", "ex_date", "ratio", "amount", "currency", "source",
]
EVENT_COLUMNS = ["event_id", "category", "cik", "ts", "is_estimated", "label", "payload"]
TRADE_COLUMNS = [
    "trade_id", "conid", "cik", "ticker", "ts", "side", "quantity", "price", "currency",
    "commission", "fx_rate",
]
CASH_TRANSACTION_COLUMNS = [
    "tx_id", "conid", "cik", "ticker", "ts", "kind", "amount", "currency",
]
MEMBERSHIP_COLUMNS = [
    "universe", "ticker", "start_date", "end_date", "source", "first_seen",
]
UNMAPPED_ACTION_COLUMNS = ["action_id", "ticker", "conid", "ex_date", "code", "description",
                           "source", "first_seen"]
SECURITY_COLUMNS = [
    "conid", "ticker", "cik", "name", "isin", "cusip", "figi", "asset_category",
    "sub_category", "listing_exchange", "issuer_country", "currency", "multiplier",
    "first_seen", "last_seen", "source",
]
THESIS_LOG_COLUMNS = [
    "cik", "thesis", "value_accrual", "invalidation", "review_date", "created_at",
]
EXIT_LADDER_COLUMNS = ["rule_id", "cik", "kind", "trigger", "action", "created_at"]
ALERT_LOG_COLUMNS = ["alert_id", "rule_id", "fired_at", "payload"]


def _utc_now_iso() -> str:
    """Current UTC time as ISO8601 (section 10: timestamps are UTC ISO8601)."""
    return datetime.now(timezone.utc).isoformat()


def _q(identifier: str) -> str:
    """Double-quote an SQL identifier.

    Quoting is unconditional so column names that collide with reserved words
    (``trigger``, ``action``) never depend on a given engine's keyword list.
    """
    return '"' + identifier.replace('"', '""') + '"'


def _build_upsert_sql(
    table: str,
    pk_cols: Sequence[str],
    columns: Sequence[str],
    with_ingested: bool,
    immutable: Sequence[str] = (),
) -> str:
    """Build an idempotent ``INSERT ... ON CONFLICT DO UPDATE`` for a table.

    Every non-key column is refreshed from ``excluded``, so re-ingesting a corrected
    value updates in place instead of inserting a second row — except the ``immutable``
    ones, which keep the value from the first insert. That is how a "first seen" date
    stays the first one instead of becoming "last run".
    """
    all_cols = list(columns) + (["ingested_at"] if with_ingested else [])
    updatable = [c for c in all_cols if c not in pk_cols and c not in immutable]
    if not updatable:
        # A table whose columns are all key columns: a conflict means the row is
        # already identical, so DO NOTHING is the idempotent outcome.
        conflict = "DO NOTHING"
    else:
        assignments = ", ".join(f"{_q(c)} = excluded.{_q(c)}" for c in updatable)
        conflict = f"DO UPDATE SET {assignments}"
    return (
        f"INSERT INTO {table} ({', '.join(_q(c) for c in all_cols)}) "
        f"VALUES ({', '.join(':' + c for c in all_cols)}) "
        f"ON CONFLICT ({', '.join(_q(c) for c in pk_cols)}) {conflict}"
    )


def _to_records(
    rows: Iterable[dict[str, Any]],
    columns: Sequence[str],
    numeric: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Project dict rows onto ``columns``, turning NaN/NaT into None.

    Missing values stay None on purpose: a visible None always beats a silently
    invented number (section 12).
    """
    records: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {}
        for col in columns:
            value = row.get(col)
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                record[col] = None
            elif col in numeric:
                record[col] = float(value)
            else:
                record[col] = value
        records.append(record)
    return records


def _require_columns(df: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    """Fail loudly when a required column is absent (section 10: no silent failures)."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{what} DataFrame missing required columns: {missing}. Expected {list(columns)}."
        )


def _upsert(
    conn: sqlite3.Connection,
    table: str,
    pk_cols: Sequence[str],
    columns: Sequence[str],
    records: list[dict[str, Any]],
    with_ingested: bool,
    label: str,
    immutable: Sequence[str] = (),
) -> int:
    """Execute the upsert in a single transaction and report the row count."""
    if not records:
        log.info("No %s to upsert.", label)
        return 0
    if with_ingested:
        ingested_at = _utc_now_iso()
        for record in records:
            record["ingested_at"] = ingested_at
    sql = _build_upsert_sql(table, pk_cols, columns, with_ingested, immutable)
    with conn:  # transaction: commit on success, rollback on error
        conn.executemany(sql, records)
    log.info("Upserted %d %s.", len(records), label)
    return len(records)


# --- Connection lifecycle -----------------------------------------------------


def connect(db_path: Path):
    """Open the backend connection: PostgreSQL if DATABASE_URL is set, else SQLite.

    Args:
        db_path: SQLite destination file (used only in the SQLite fallback).

    Returns:
        An open connection with the sqlite3 interface. The caller closes it.
    """
    return open_connection(db_path)


# Columns added to tables that already existed in earlier databases. `CREATE TABLE IF NOT
# EXISTS` cannot reach those, so they are applied separately (see add_missing_columns).
# Only ever additive and nullable; existing rows fill in on the next idempotent ingest.
LATE_COLUMNS: dict[str, dict[str, str]] = {
    "trades": {"conid": "TEXT"},
    "cash_transactions": {"conid": "TEXT"},
    # 2026-09-25: the SEC industry code, for comparing a company with its peers.
    "companies": {"sic": "TEXT", "sic_description": "TEXT"},
    # 2026-09-25: the 8-K item codes, for the governance signals (4.02, 3.01, 4.01…).
    "filings": {"items": "TEXT"},
}


def apply_schema(conn: sqlite3.Connection) -> None:
    """Apply db/schema.sql to the connection, then any late column. Idempotent."""
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(ddl)
    conn.commit()
    add_missing_columns(conn, LATE_COLUMNS)
    log.debug("Schema applied from %s", SCHEMA_PATH)


def init_db(db_path: Path) -> sqlite3.Connection:
    """Connect and ensure the schema exists. Returns the open connection."""
    conn = connect(db_path)
    apply_schema(conn)
    return conn


# --- Observations -------------------------------------------------------------


def upsert_observations(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Upsert an observations DataFrame idempotently.

    Args:
        conn: Open connection with the schema applied.
        df: DataFrame with at least ``OBSERVATION_COLUMNS``. A missing ``ts_release``
            is stored as :data:`TS_RELEASE_UNKNOWN`, never as NULL (see the constant).

    Returns:
        Number of rows submitted. Re-running with the same rows does not duplicate.

    Raises:
        ValueError: If required columns are missing (fail loudly, section 10).
    """
    _require_columns(df, OBSERVATION_COLUMNS, "Observation")
    records = _to_records(
        df[OBSERVATION_COLUMNS].to_dict("records"), OBSERVATION_COLUMNS, numeric=("value",)
    )
    for record in records:
        if record["ts_release"] is None:
            record["ts_release"] = TS_RELEASE_UNKNOWN
    return _upsert(
        conn, "observations", ("source", "series_id", "ts", "ts_release"),
        OBSERVATION_COLUMNS, records, True, "observations",
    )


# --- Reference tables ---------------------------------------------------------


def upsert_companies(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    """Upsert the company registry, keyed by CIK (section 9.3: the ticker is not a key).

    The ticker and name update in place — they are the mutable attributes. ``first_seen``
    keeps the first run's date.
    """
    return _upsert(
        conn, "companies", ("cik",), COMPANY_COLUMNS,
        _to_records(rows, COMPANY_COLUMNS), False, "companies", immutable=("first_seen",),
    )


def upsert_filings(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    """Upsert SEC filings, keyed by accession number (sections 4.4, 9.6)."""
    return _upsert(
        conn, "filings", ("accession",), FILING_COLUMNS,
        _to_records(rows, FILING_COLUMNS), False, "filings",
    )


def upsert_corporate_actions(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    """Upsert corporate actions, kept separate from raw prices (section 9.1)."""
    return _upsert(
        conn, "corporate_actions", ("action_id",), CORPORATE_ACTION_COLUMNS,
        _to_records(rows, CORPORATE_ACTION_COLUMNS, numeric=("ratio", "amount")),
        False, "corporate actions",
    )


def upsert_events(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    """Upsert calendar events, keyed by event_id (macro releases, earnings dates)."""
    return _upsert(
        conn, "events", ("event_id",), EVENT_COLUMNS,
        _to_records(rows, EVENT_COLUMNS), False, "events",
    )


# --- Real account (IBKR Flex, read-only) --------------------------------------


def upsert_trades(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Upsert executed trades from the Flex Query, keyed by IBKR tradeID.

    Raises:
        ValueError: If required columns are missing (section 10).
    """
    _require_columns(df, TRADE_COLUMNS, "Trades")
    records = _to_records(
        df[TRADE_COLUMNS].to_dict("records"), TRADE_COLUMNS,
        numeric=("quantity", "price", "commission", "fx_rate"),
    )
    return _upsert(conn, "trades", ("trade_id",), TRADE_COLUMNS, records, True, "trades")


def upsert_cash_transactions(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Upsert dividends, withholding, interest and fees, keyed by tx_id.

    Withholding rows carry the figure IBKR actually reported; no rate is ever
    assumed on the user's behalf (section 11).

    Raises:
        ValueError: If required columns are missing (section 10).
    """
    _require_columns(df, CASH_TRANSACTION_COLUMNS, "Cash-transactions")
    records = _to_records(
        df[CASH_TRANSACTION_COLUMNS].to_dict("records"), CASH_TRANSACTION_COLUMNS,
        numeric=("amount",),
    )
    return _upsert(
        conn, "cash_transactions", ("tx_id",), CASH_TRANSACTION_COLUMNS, records,
        True, "cash transactions",
    )


def upsert_unmapped_actions(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    """Upsert the reorganizations IBKR reported and the panel could not classify (§9.2).

    ``first_seen`` keeps the first day: it says how long a position has been waiting for
    someone to look at it.
    """
    return _upsert(
        conn, "unmapped_actions", ("action_id",), UNMAPPED_ACTION_COLUMNS,
        _to_records(rows, UNMAPPED_ACTION_COLUMNS), True, "unmapped actions",
        immutable=("first_seen",),
    )


def upsert_securities(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    """Upsert the security master, keyed by IBKR's permanent ``conid`` (section 9.3).

    The ticker is a plain attribute here, not the key: it gets renamed under the same
    company and reassigned between companies, so a master keyed on it could not describe
    its own history. ``issuer_country`` is carried because it determines an asset's situs,
    which section 11 needs — and nothing is computed from it (Claude asserts no tax
    treatment).
    """
    return _upsert(
        conn, "securities", ("conid",), SECURITY_COLUMNS,
        _to_records(rows, SECURITY_COLUMNS, numeric=("multiplier",)), True, "securities",
    )


def upsert_universe_membership(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    """Upsert index-membership intervals (section 9.5).

    ``end_date`` updates in place when a member leaves — the interval is the same one, it
    just stopped. ``first_seen`` does not: it records when this project first learned of
    the interval, and overwriting it on every run would erase the only point-in-time fact
    there is about the list itself.
    """
    return _upsert(
        conn, "universe_membership", ("universe", "ticker", "start_date"),
        MEMBERSHIP_COLUMNS, _to_records(rows, MEMBERSHIP_COLUMNS), True,
        "membership intervals", immutable=("first_seen",),
    )


# --- Written discipline: theses and exit rules --------------------------------


def upsert_thesis_log(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    """Upsert thesis entries, keyed by CIK (sections 5.2, 6).

    The loader enforces the same rule as the schema's NOT NULL: an entry without a
    non-empty ``invalidation`` is rejected loudly, so the failure names the offending
    company instead of surfacing as an opaque constraint error.

    Raises:
        ValueError: If any row lacks an invalidation criterion.
    """
    for row in rows:
        if not str(row.get("invalidation") or "").strip():
            raise ValueError(
                f"thesis_log entry for CIK {row.get('cik')!r} has no invalidation "
                "criterion — a thesis is not stored without a way to be proven wrong "
                "(CLAUDE.md sections 5.2, 6)."
            )
    return _upsert(
        conn, "thesis_log", ("cik",), THESIS_LOG_COLUMNS,
        _to_records(rows, THESIS_LOG_COLUMNS), False, "thesis entries",
    )


def upsert_exit_ladder(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    """Upsert exit rules, keyed by rule_id — written before buying (section 2)."""
    return _upsert(
        conn, "exit_ladder", ("rule_id",), EXIT_LADDER_COLUMNS,
        _to_records(rows, EXIT_LADDER_COLUMNS), False, "exit-ladder rules",
        # When the rule was first written down is the point of the table: it proves the
        # exit was decided before the purchase (section 2, level 4). A re-sync must not
        # move it to "last run".
        immutable=("created_at",),
    )


def upsert_alerts(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    """Upsert fired-alert records, keyed by alert_id (dedup ledger, phase 4)."""
    return _upsert(
        conn, "alerts_log", ("alert_id",), ALERT_LOG_COLUMNS,
        _to_records(rows, ALERT_LOG_COLUMNS), False, "alerts",
    )
