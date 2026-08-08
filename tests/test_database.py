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
