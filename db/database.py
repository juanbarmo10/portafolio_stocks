"""Backend selection: SQLite (local, phases 0-4) or PostgreSQL (cloud, phase 5).

CLAUDE.md sections 3 and 8: the adapter exists from the start so the long-format schema
and the ``ON CONFLICT`` upserts stay portable, rather than being retrofitted later.

The rest of the codebase talks to a connection with the sqlite3 interface it already
uses (``execute`` / ``executemany`` / ``executescript`` / ``commit`` / ``close`` and
``with conn:``). This module returns either:

- a **native sqlite3 connection** (default; the SQLite path is completely unchanged), or
- a thin **Postgres facade** mimicking that interface, when ``DATABASE_URL`` is set.

Only the Postgres facade translates SQL (``?`` -> ``%s``, ``:name`` -> ``%(name)s``) and
applies the schema statement-by-statement, skipping SQLite-only ``PRAGMA``.

Postgres needs the optional extra: ``pip install -e ".[postgres]"``.
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

from core.logging_setup import get_logger

log = get_logger(__name__)

# Named placeholders are :word. The negative lookbehind keeps '::' casts and the ':' in
# an ISO8601 literal (e.g. '2026-01-02T00:00:00Z') from being mistaken for a parameter.
_NAMED = re.compile(r"(?<![:\w]):(\w+)")


def _translate(sql: str, named: bool) -> str:
    """Translate sqlite-style placeholders to psycopg style for Postgres."""
    return _NAMED.sub(r"%(\1)s", sql) if named else sql.replace("?", "%s")


class _PgCursor:
    """Wrap a psycopg cursor so row access matches sqlite3 (tuple rows)."""

    def __init__(self, cursor: Any) -> None:
        self._cur = cursor

    def fetchone(self) -> Any:
        return self._cur.fetchone()

    def fetchall(self) -> list[Any]:
        return self._cur.fetchall()

    def __iter__(self):
        # sqlite3 cursors are iterable row-by-row; mirror that so
        # `for row in conn.execute(...)` behaves identically on both backends.
        return iter(self._cur)


class _PgConnection:
    """Minimal sqlite3-like facade over a psycopg connection."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def execute(self, sql: str, params: Sequence[Any] | dict | None = None) -> _PgCursor:
        cur = self._conn.cursor()
        named = isinstance(params, dict)
        cur.execute(_translate(sql, named), params if params is not None else None)
        return _PgCursor(cur)

    def executemany(self, sql: str, seq: Sequence[Any]) -> None:
        seq = list(seq)
        named = bool(seq) and isinstance(seq[0], dict)
        cur = self._conn.cursor()
        cur.executemany(_translate(sql, named), seq)

    def executescript(self, script: str) -> None:
        # Postgres has no executescript: run each statement. Strip line comments (-- to
        # end of line) FIRST — a ';' inside a comment would otherwise split a statement in
        # two. Then skip SQLite-only PRAGMA. (schema.sql has no '--' inside string
        # literals, so removing comments this way is safe.)
        no_comments = "\n".join(line.split("--", 1)[0] for line in script.splitlines())
        for raw in no_comments.split(";"):
            code = raw.strip()
            if not code or code.upper().startswith("PRAGMA"):
                continue
            self._conn.cursor().execute(code)

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "_PgConnection":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()


def database_url() -> str | None:
    """Return DATABASE_URL if set (Postgres), else None (use SQLite)."""
    url = os.getenv("DATABASE_URL")
    return url.strip() if url else None


def open_connection(sqlite_path: Path) -> sqlite3.Connection | _PgConnection:
    """Open the backend connection: Postgres if DATABASE_URL is set, else SQLite.

    Args:
        sqlite_path: Path used when falling back to SQLite (parent dirs are created).

    Returns:
        An open connection exposing the sqlite3 interface.

    Raises:
        RuntimeError: If DATABASE_URL is set but psycopg is not installed.
    """
    url = database_url()
    if url:
        try:
            import psycopg  # noqa: PLC0415 — optional dependency, only for Postgres
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "DATABASE_URL is set but psycopg is not installed. "
                'Install the extra: pip install -e ".[postgres]".'
            ) from exc
        log.info("Using PostgreSQL backend (DATABASE_URL).")
        return _PgConnection(psycopg.connect(url))

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def read_observations(
    conn: Any,
    series_ids: Sequence[str] | None = None,
    source: str | None = None,
) -> "pd.DataFrame":
    """Read observations into a DataFrame, optionally filtered by series or source.

    Every version of a restated fact is returned — selecting the one vigent at a given
    date is the caller's job and belongs in ``transform`` (section 9.6). Filtering here
    would bake one answer into the data-access layer.

    Args:
        conn: Open connection from :func:`open_connection`.
        series_ids: Restrict to these series. ``None`` means every series.
        source: Restrict to one source label ('fred', 'yfinance', ...).

    Returns:
        Frame with columns ``[source, series_id, ts, ts_release, value]``. Empty (but
        correctly shaped) when nothing matches, so callers never special-case it.
    """
    import pandas as pd  # noqa: PLC0415 — keeps the adapter importable without pandas

    columns = ["source", "series_id", "ts", "ts_release", "value"]
    sql = f"SELECT {', '.join(columns)} FROM observations"
    clauses: list[str] = []
    params: list[Any] = []
    if series_ids:
        clauses.append(f"series_id IN ({', '.join('?' for _ in series_ids)})")
        params.extend(series_ids)
    if source:
        clauses.append("source = ?")
        params.append(source)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([dict(zip(columns, r)) for r in rows])


# Tables readable row-by-row, with their columns and the column to order by. Declared here
# rather than imported from db.loader because the dependency runs the other way (loader
# imports this module); a test asserts the lists stay identical, so the duplication cannot
# drift in silence. The allowlist is also what keeps a table name out of string-built SQL.
READABLE_TABLES: dict[str, tuple[list[str], str]] = {
    "trades": ([
        "trade_id", "cik", "ticker", "ts", "side", "quantity", "price", "currency",
        "commission", "fx_rate",
    ], "ts"),
    "cash_transactions": (
        ["tx_id", "cik", "ticker", "ts", "kind", "amount", "currency"], "ts",
    ),
    "filings": (
        ["accession", "cik", "form", "period_end", "filed_date", "is_amended", "url"],
        "filed_date",
    ),
    "events": (
        ["event_id", "category", "cik", "ts", "is_estimated", "label", "payload"], "ts",
    ),
    "corporate_actions": (
        ["action_id", "cik", "ticker", "kind", "ex_date", "ratio", "amount", "currency",
         "source"],
        "ex_date",
    ),
}

# Kept for the account tables specifically, which is what most callers want.
ACCOUNT_TABLES: dict[str, list[str]] = {
    name: columns for name, (columns, _order) in READABLE_TABLES.items()
    if name in ("trades", "cash_transactions")
}


def read_table(conn: Any, table: str) -> "pd.DataFrame":
    """Read one allowlisted table into a DataFrame, ordered by its time column.

    Args:
        conn: Open connection from :func:`open_connection`.
        table: One of :data:`READABLE_TABLES`.

    Returns:
        Frame with that table's columns. Empty (but correctly shaped) when it has no rows,
        so callers never special-case it.

    Raises:
        ValueError: On any other name.
    """
    import pandas as pd  # noqa: PLC0415 — keeps the adapter importable without pandas

    if table not in READABLE_TABLES:
        raise ValueError(f"Unknown table {table!r}. Known: {sorted(READABLE_TABLES)}.")
    columns, order_by = READABLE_TABLES[table]
    rows = conn.execute(
        f"SELECT {', '.join(columns)} FROM {table} ORDER BY {order_by}"
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([dict(zip(columns, r)) for r in rows])


def read_account_table(conn: Any, table: str) -> "pd.DataFrame":
    """Read one of the IBKR account tables. Thin alias over :func:`read_table`."""
    if table not in ACCOUNT_TABLES:
        raise ValueError(f"Unknown account table {table!r}. Known: {sorted(ACCOUNT_TABLES)}.")
    return read_table(conn, table)
