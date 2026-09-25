"""Data access for the Streamlit pages (CLAUDE.md sections 3, 7).

The pages never open a connection themselves. Reads go through here so that caching,
path resolution and the "database not built yet" case are handled once.

Caching is keyed on the database file's modification time: the panel is **pull, not
push** (section 2), so it must not re-query on every widget interaction, but it also must
never show data older than the last ingest without saying so.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import load_settings
from db.database import (
    database_url,
    open_connection,
    read_account_table,
    read_observations,
    read_table,
)


def db_path() -> Path:
    """Absolute path of the configured database file."""
    return load_settings().db_path


def is_postgres() -> bool:
    """Whether the panel reads PostgreSQL (``DATABASE_URL``) — the public deployment."""
    return database_url() is not None


def database_ready() -> bool:
    """Whether there is a database to read: the SQLite file, or a configured PostgreSQL."""
    return is_postgres() or db_path().exists()


def db_mtime() -> float:
    """The cache version: a fresh ingest must invalidate every cached read.

    SQLite: the file's modification time, or 0.0 when it does not exist yet. PostgreSQL has
    no file; the public copy is synced once a day, so the current hour is fresh enough and
    costs one round of queries per hour instead of one per click.
    """
    if is_postgres():
        return float(int(time.time() // 3600) * 3600)
    path = db_path()
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def observations(source: str, mtime: float) -> pd.DataFrame:
    """All observations from one source.

    Args:
        source: Source label ('fred', 'yfinance', ...).
        mtime: Database mtime. Part of the cache key, so a fresh ingest invalidates it.
            It must NOT carry a leading underscore: Streamlit leaves underscored arguments
            out of the key, which is exactly what they are for. Until 2026-09-24 it was
            ``mtime``, and a running panel never showed a new ingest until restarted.

    Returns:
        Frame with ``[source, series_id, ts, ts_release, value]``; empty if the database
        has not been built yet.
    """
    path = db_path()
    if not database_ready():
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
    if is_postgres():
        return _last_ingest_postgres(db_mtime())
    mtime = db_mtime()
    return dt.datetime.fromtimestamp(mtime) if mtime else None


@st.cache_data(show_spinner=False)
def _last_ingest_postgres(version: float) -> dt.datetime | None:
    """The newest ``ingested_at`` — in PostgreSQL there is no file date to read."""
    conn = open_connection(db_path())
    try:
        row = conn.execute("SELECT MAX(ingested_at) FROM observations").fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        return None
    return dt.datetime.fromisoformat(str(row[0])).astimezone()


@st.cache_data(show_spinner=False)
def _account_table(table: str, mtime: float) -> pd.DataFrame:
    """One IBKR account table, cache-invalidated by the database mtime."""
    path = db_path()
    if not database_ready():
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


def unmapped_actions() -> pd.DataFrame:
    """Reorganizations IBKR reported and the panel could not classify (section 9.2)."""
    return _account_table("unmapped_actions", db_mtime())


@st.cache_data(show_spinner=False)
def _table(table: str, mtime: float) -> pd.DataFrame:
    """Any allowlisted table, cache-invalidated by the database mtime."""
    path = db_path()
    if not database_ready():
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


def corporate_actions() -> pd.DataFrame:
    """Splits, dividends and reorgs from **both** providers, each tagged with its source.

    Both are returned on purpose: ``transform.corporate_actions`` needs the two witnesses
    present to notice that they disagree (section 9.2).
    """
    return _table("corporate_actions", db_mtime())


def securities() -> pd.DataFrame:
    """Security master from the Flex Query, keyed on IBKR's permanent ``conid``.

    The only stable key the account side has (section 9.3), and the only place the
    issuer country — an asset's situs, which section 11 needs — is recorded.
    """
    return _table("securities", db_mtime())


def companies() -> pd.DataFrame:
    """The company registry (CIK ↔ current ticker)."""
    return _table("companies", db_mtime())


def universe_membership() -> pd.DataFrame:
    """S&P 500 membership intervals, for breadth (section 9.5)."""
    return _table("universe_membership", db_mtime())


def fred_observations() -> pd.DataFrame:
    """Every FRED series — the regime light reads several beyond the level-1 table."""
    return observations("fred", db_mtime())


def sec_observations() -> pd.DataFrame:
    """Audited XBRL facts for the tracked universe."""
    return observations("sec", db_mtime())


def account_observations() -> pd.DataFrame:
    """Positions and NAV series written by the IBKR ingester."""
    return observations("ibkr", db_mtime())


@st.cache_data(show_spinner=False)
def price_observations(tickers: tuple[str, ...], mtime: float) -> pd.DataFrame:
    """Raw closes for the given tickers only.

    Reading every price series would pull ~200k rows to value a handful of positions, and
    the panel is meant to be cheap to open (section 2).
    """
    path = db_path()
    columns = ["source", "series_id", "ts", "ts_release", "value"]
    if not database_ready() or not tickers:
        return pd.DataFrame(columns=columns)
    conn = open_connection(path)
    try:
        return read_observations(conn, series_ids=[f"{t}:close_raw" for t in tickers])
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def recent_closes(tickers: tuple[str, ...], since: str, mtime: float) -> pd.DataFrame:
    """Raw closes from ``since`` on — the last price of ~500 members without their history."""
    columns = ["source", "series_id", "ts", "ts_release", "value"]
    if not database_ready() or not tickers:
        return pd.DataFrame(columns=columns)
    conn = open_connection(db_path())
    try:
        return read_observations(conn, series_ids=[f"{t}:close_raw" for t in tickers],
                                 since=since)
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def source_last_ingest(source: str, mtime: float) -> str | None:
    """Date (ISO) of the last ingest of one source; ``None`` if it never ran."""
    if not database_ready():
        return None
    conn = open_connection(db_path())
    try:
        row = conn.execute("SELECT MAX(ingested_at) FROM observations WHERE source = ?",
                           (source,)).fetchone()
    finally:
        conn.close()
    return str(row[0])[:10] if row and row[0] else None


def prices_for(tickers: list[str]) -> pd.DataFrame:
    """Raw closes for the held tickers, cache-invalidated by the database mtime."""
    return price_observations(tuple(sorted(tickers)), db_mtime())


@st.cache_data(show_spinner="Calculando el semáforo…")
def regime_view(mtime: float) -> dict | None:
    """The regime light and its history, cached until the database changes.

    Shared by the market page and the landing page, so both read the same verdict and it
    is computed once per data version (``transform.regime.build`` is not cheap).
    """
    from transform import regime as rg  # noqa: PLC0415 — keeps page imports light

    level2 = load_settings().raw["panel"]["level2"]
    built = rg.build(fred_observations(), prices_for(rg.regime_tickers(level2)),
                     corporate_actions(), level2)
    if built is None:
        return None
    return {
        "reading": rg.reading_at(built.calendar[-1], built.rules, built.components,
                                 built.votes, built.frame),
        "frame": built.frame[["verdict", "available"]],
        "evaluation": rg.evaluate(built.frame, built.benchmark),
        "closes": built.closes, "splits": built.splits,
    }


@st.cache_data(show_spinner="Calculando el cribado…")
def screen_view(mtime: float) -> pd.DataFrame:
    """The quarterly screen table (``transform.screen``): SEC frames, the registry's names and
    the last closes. Shared by the screen page and the company page's peer comparison."""
    from ingest.screen import SOURCE  # noqa: PLC0415 — the label lives with its ingester
    from transform import screen as sc  # noqa: PLC0415

    frames = observations(SOURCE, mtime)
    if frames.empty:
        return pd.DataFrame()
    registry = companies()
    names = {str(r["cik"]): (str(r["ticker"]), r.get("name"))
             for r in registry.to_dict("records")}
    # Class shares: a dot in the membership list (BRK.B), a dash at the price source.
    tickers = tuple(sorted({t for t, _ in names.values()}
                           | {t.replace(".", "-") for t, _ in names.values()}))
    since = (dt.date.today() - dt.timedelta(days=20)).isoformat()
    closes = recent_closes(tickers, since, mtime)
    return sc.screen_table(frames, closes, corporate_actions(), names)


@st.cache_data(show_spinner="Calculando la valoración…")
def valuation_view(cik: str, ticker: str, as_of_iso: str, years: int, mtime: float):
    """Today's valuation and its monthly history, point-in-time (``transform.valuation``).
    Shared by the company page and the deployment page's valuation band."""
    from transform import valuation as val  # noqa: PLC0415

    obs = sec_observations()
    prices = prices_for([ticker])
    actions = corporate_actions()
    now = val.assess(obs, prices, actions, cik, ticker, as_of_iso)
    end = pd.Timestamp(as_of_iso)
    dates = [*pd.date_range(end - pd.DateOffset(years=years), end, freq="ME"), end]
    return now, val.history(obs, prices, actions, cik, ticker, dates)
