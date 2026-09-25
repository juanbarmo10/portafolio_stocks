"""Backend-adapter tests (CLAUDE.md sections 3, 8 phase 5).

The SQLite<->Postgres adapter exists from day one so the schema and the upserts stay
portable. These tests exercise the translation layer without needing a Postgres server.
"""

from __future__ import annotations

import sqlite3

from db import database, loader


def test_sqlite_is_the_default_backend(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert database.database_url() is None
    conn = database.open_connection(tmp_path / "nested" / "test.db")
    assert isinstance(conn, sqlite3.Connection)
    conn.close()
    assert (tmp_path / "nested" / "test.db").exists()


def test_database_url_selects_postgres(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "  postgresql://user@host/db  ")
    assert database.database_url() == "postgresql://user@host/db"


def test_positional_placeholder_translation():
    assert database._translate("SELECT * FROM t WHERE a = ? AND b = ?", named=False) == (
        "SELECT * FROM t WHERE a = %s AND b = %s"
    )


def test_named_placeholder_translation():
    sql = "INSERT INTO observations VALUES (:source, :series_id, :ts)"
    assert database._translate(sql, named=True) == (
        "INSERT INTO observations VALUES (%(source)s, %(series_id)s, %(ts)s)"
    )


def test_iso8601_literals_are_not_mistaken_for_placeholders():
    """A ':' inside a timestamp literal is not a parameter.

    Timestamps are ISO8601 UTC everywhere (section 10), so this case is guaranteed to
    show up the moment a query filters on a literal date.
    """
    sql = "SELECT * FROM observations WHERE ts > '2026-01-02T00:00:00+00:00' AND source = :source"
    translated = database._translate(sql, named=True)
    assert "'2026-01-02T00:00:00+00:00'" in translated
    assert translated.endswith("source = %(source)s")


def test_upsert_sql_refreshes_every_non_key_column():
    """Correcting a value must update in place; that is what makes re-runs safe."""
    sql = loader._build_upsert_sql(
        "observations", ("source", "series_id", "ts", "ts_release"),
        loader.OBSERVATION_COLUMNS, with_ingested=True,
    )
    assert 'ON CONFLICT ("source", "series_id", "ts", "ts_release")' in sql
    assert '"value" = excluded."value"' in sql
    assert '"ingested_at" = excluded."ingested_at"' in sql
    # Key columns are never in the SET clause.
    assert '"ts" = excluded."ts"' not in sql


def test_schema_has_no_sqlite_only_types():
    """Portability guard: the DDL must survive being applied to Postgres (section 3)."""
    ddl = loader.SCHEMA_PATH.read_text(encoding="utf-8").upper()
    for sqlite_only in ("AUTOINCREMENT", "WITHOUT ROWID", " BLOB", "DATETIME("):
        assert sqlite_only not in ddl, f"schema.sql uses SQLite-only construct: {sqlite_only}"


def test_postgres_executescript_splits_statements_and_skips_pragma():
    """The Postgres facade applies the schema statement by statement."""
    executed: list[str] = []

    class _FakeCursor:
        def execute(self, sql, params=None):
            executed.append(sql.strip())

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

    facade = database._PgConnection(_FakeConn())
    facade.executescript(loader.SCHEMA_PATH.read_text(encoding="utf-8"))

    assert executed, "no statements were executed"
    assert not any(s.upper().startswith("PRAGMA") for s in executed)
    assert any(s.startswith("CREATE TABLE IF NOT EXISTS observations") for s in executed)
    # A ';' inside a comment must not have split a statement in two.
    assert all(s.upper().startswith(("CREATE TABLE", "CREATE INDEX")) for s in executed)


def test_account_table_columns_match_the_loader():
    """The reader declares its own columns because the import runs the other way.

    db.loader imports db.database, so the reader cannot import the loader's constants back.
    The duplication is deliberate and this test is the reason it cannot drift in silence.
    """
    from db import loader
    from db.database import ACCOUNT_TABLES

    assert ACCOUNT_TABLES["trades"] == loader.TRADE_COLUMNS
    assert ACCOUNT_TABLES["cash_transactions"] == loader.CASH_TRANSACTION_COLUMNS


def test_every_readable_table_matches_the_loader_column_list():
    """The two lists are declared separately (the dependency runs one way) so they are
    compared here — otherwise they drift in silence and a read returns the wrong shape."""
    from db.database import READABLE_TABLES

    expected = {
        "trades": loader.TRADE_COLUMNS,
        "cash_transactions": loader.CASH_TRANSACTION_COLUMNS,
        "filings": loader.FILING_COLUMNS,
        "events": loader.EVENT_COLUMNS,
        "corporate_actions": loader.CORPORATE_ACTION_COLUMNS,
        "securities": loader.SECURITY_COLUMNS,
        "universe_membership": loader.MEMBERSHIP_COLUMNS,
        "companies": loader.COMPANY_COLUMNS,
        "exit_ladder": loader.EXIT_LADDER_COLUMNS,
        "alerts_log": loader.ALERT_LOG_COLUMNS,
    }
    assert set(READABLE_TABLES) == set(expected), "a readable table has no column check"
    for table, columns in expected.items():
        assert READABLE_TABLES[table][0] == columns, table


# --- Additive migration for tables that already existed ----------------------


def test_a_column_added_later_reaches_a_database_that_predates_it(tmp_path):
    """``CREATE TABLE IF NOT EXISTS`` does nothing to a table that already exists.

    The long-format ``observations`` table exists so new *series* never need a migration
    (section 6), but the account tables are relational and ``conid`` had to be added to
    them. Without this step a database created last week would silently keep the old
    shape and every write would fail on the missing column.
    """
    import sqlite3

    from db.database import add_missing_columns, table_columns

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE trades (trade_id TEXT PRIMARY KEY, ticker TEXT NOT NULL)"
        )
        conn.execute("INSERT INTO trades VALUES ('T1', 'AAPL')")
        conn.commit()

        assert "conid" not in table_columns(conn, "trades")
        added = add_missing_columns(conn, {"trades": {"conid": "TEXT"}})

        assert added == ["trades.conid"]
        assert "conid" in table_columns(conn, "trades")
        # The existing row survives, with the new column empty until the next ingest
        # upserts over it — which is safe precisely because every ingest is idempotent.
        assert conn.execute("SELECT trade_id, conid FROM trades").fetchone() == ("T1", None)
    finally:
        conn.close()


def test_applying_the_migration_twice_changes_nothing(tmp_path):
    """Idempotent, like everything else that touches the database (section 6)."""
    import sqlite3

    from db.database import add_missing_columns

    conn = sqlite3.connect(tmp_path / "twice.db")
    try:
        conn.execute("CREATE TABLE trades (trade_id TEXT PRIMARY KEY)")
        conn.commit()
        assert add_missing_columns(conn, {"trades": {"conid": "TEXT"}}) == ["trades.conid"]
        assert add_missing_columns(conn, {"trades": {"conid": "TEXT"}}) == []
    finally:
        conn.close()


def test_a_table_that_does_not_exist_yet_is_left_to_the_schema(tmp_path):
    """schema.sql has just created it with the column already in place."""
    import sqlite3

    from db.database import add_missing_columns

    conn = sqlite3.connect(tmp_path / "absent.db")
    try:
        assert add_missing_columns(conn, {"nope": {"conid": "TEXT"}}) == []
    finally:
        conn.close()


def test_init_db_on_a_legacy_database_makes_it_writable_again(tmp_path):
    """End to end: an old database is opened, migrated, and accepts the new rows."""
    import sqlite3

    import pandas as pd

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    try:
        # The trades table exactly as it was before conid existed.
        conn.execute(
            """CREATE TABLE trades (
                trade_id TEXT PRIMARY KEY, cik TEXT, ticker TEXT NOT NULL, ts TEXT NOT NULL,
                side TEXT NOT NULL, quantity REAL NOT NULL, price REAL NOT NULL,
                currency TEXT NOT NULL, commission REAL, fx_rate REAL, ingested_at TEXT NOT NULL
            )"""
        )
        conn.commit()
    finally:
        conn.close()

    conn = loader.init_db(db)
    try:
        written = loader.upsert_trades(conn, pd.DataFrame([{
            "trade_id": "T1", "conid": "272093", "cik": None, "ticker": "MSFT",
            "ts": "2026-01-05T14:31:00+00:00", "side": "buy", "quantity": 1.0,
            "price": 400.0, "currency": "USD", "commission": -0.35, "fx_rate": 1.0,
        }]))
        assert written == 1
        assert conn.execute("SELECT conid FROM trades").fetchone()[0] == "272093"
    finally:
        conn.close()
