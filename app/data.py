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
from db.database import open_connection, read_observations


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
