"""Data access for the Streamlit pages (CLAUDE.md sections 3, 7).

The pages never open a connection themselves. Reads go through here so that caching,
path resolution and the "database not built yet" case are handled once.

Caching is keyed on the database file's modification time: the panel is **pull, not
push** (section 2), so it must not re-query on every widget interaction, but it also must
never show data older than the last ingest without saying so.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import load_settings
from db.database import open_connection, read_account_table, read_observations, read_table


def db_path() -> Path:
    """Absolute path of the configured database file."""
    return load_settings().db_path


def db_mtime() -> float:
    """Modification time of the database, or 0.0 when it does not exist yet."""
    path = db_path()
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def observations(source: str, _mtime: float) -> pd.DataFrame:
    """All observations from one source.

    Args:
        source: Source label ('fred', 'yfinance', ...).
        _mtime: Database mtime. Part of the cache key so a fresh ingest invalidates it;
            the leading underscore keeps Streamlit from hashing it as a data argument.

    Returns:
        Frame with ``[source, series_id, ts, ts_release, value]``; empty if the database
        has not been built yet.
    """
    path = db_path()
    if not path.exists():
        return pd.DataFrame(columns=["source", "series_id", "ts", "ts_release", "value"])
    conn = open_connection(path)
    try:
        return read_observations(conn, source=source)
    finally:
        conn.close()


def macro_observations() -> pd.DataFrame:
    """FRED observations, cache-invalidated by the database mtime."""
    return observations("fred", db_mtime())


def last_ingest() -> dt.datetime | None:
    """Timestamp of the last write to the database, as a local datetime."""
    mtime = db_mtime()
    return dt.datetime.fromtimestamp(mtime) if mtime else None


@st.cache_data(show_spinner=False)
def _account_table(table: str, _mtime: float) -> pd.DataFrame:
    """One IBKR account table, cache-invalidated by the database mtime."""
    path = db_path()
    if not path.exists():
        from db.database import ACCOUNT_TABLES  # noqa: PLC0415 — only for the empty case

        return pd.DataFrame(columns=ACCOUNT_TABLES[table])
    conn = open_connection(path)
    try:
        return read_account_table(conn, table)
    finally:
        conn.close()


def trades() -> pd.DataFrame:
    """Executed trades from the Flex Query. Never typed by hand (section 5.1)."""
    return _account_table("trades", db_mtime())


def cash_transactions() -> pd.DataFrame:
    """Dividends, withholding, interest, fees and deposits from the Flex Query."""
    return _account_table("cash_transactions", db_mtime())


@st.cache_data(show_spinner=False)
def _table(table: str, _mtime: float) -> pd.DataFrame:
    """Any allowlisted table, cache-invalidated by the database mtime."""
    path = db_path()
    if not path.exists():
        from db.database import READABLE_TABLES  # noqa: PLC0415 — only for the empty case

        return pd.DataFrame(columns=READABLE_TABLES[table][0])
    conn = open_connection(path)
    try:
        return read_table(conn, table)
    finally:
        conn.close()


def filings() -> pd.DataFrame:
    """Filing history from SEC submissions, including the amendment flag (section 9.6)."""
    return _table("filings", db_mtime())


def events() -> pd.DataFrame:
    """Calendar events — earnings dates, confirmed and estimated (RESEARCH.md 2.14)."""
    return _table("events", db_mtime())


def sec_observations() -> pd.DataFrame:
    """Audited XBRL facts for the tracked universe."""
    return observations("sec", db_mtime())


def account_observations() -> pd.DataFrame:
    """Positions and NAV series written by the IBKR ingester."""
    return observations("ibkr", db_mtime())


@st.cache_data(show_spinner=False)
def price_observations(tickers: tuple[str, ...], _mtime: float) -> pd.DataFrame:
    """Raw closes for the given tickers only.

    Reading every price series would pull ~200k rows to value a handful of positions, and
    the panel is meant to be cheap to open (section 2).
    """
    path = db_path()
    columns = ["source", "series_id", "ts", "ts_release", "value"]
    if not path.exists() or not tickers:
        return pd.DataFrame(columns=columns)
    conn = open_connection(path)
    try:
        return read_observations(conn, series_ids=[f"{t}:close_raw" for t in tickers])
    finally:
        conn.close()


def prices_for(tickers: list[str]) -> pd.DataFrame:
    """Raw closes for the held tickers, cache-invalidated by the database mtime."""
    return price_observations(tuple(sorted(tickers)), db_mtime())
