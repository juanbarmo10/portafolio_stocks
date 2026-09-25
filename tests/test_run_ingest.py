"""Entry-point tests — the phase-0 acceptance criterion (CLAUDE.md section 8)."""

from __future__ import annotations

import pandas as pd
import pytest

import run_ingest
from core import config
from ingest.base import Ingester


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """Point load_settings at a throwaway database so tests never touch the real one."""
    db_path = tmp_path / "test.db"
    real_loader = config.load_settings

    config.load_settings.cache_clear()
    settings = real_loader()
    monkeypatch.setattr(
        run_ingest, "load_settings", lambda: type(settings)(
            raw=settings.raw, db_path=db_path, log_level="WARNING",
            secrets={}, public_mode=settings.public_mode,
        )
    )
    yield db_path
    config.load_settings.cache_clear()


def test_dry_run_creates_the_database_and_succeeds(isolated_db):
    """`python run_ingest.py --dry-run` creates the DB and does not fail (section 8)."""
    assert run_ingest.main(["--dry-run"]) == 0
    assert isolated_db.exists()


def test_dry_run_is_repeatable(isolated_db):
    """Re-running applies the schema again without error and adds no rows."""
    assert run_ingest.main(["--dry-run"]) == 0
    assert run_ingest.main(["--dry-run"]) == 0
    from db import loader
    conn = loader.connect(isolated_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
    finally:
        conn.close()


def test_unknown_ingester_name_fails_loudly():
    """A typo must not silently degrade into "ran nothing" (section 10)."""
    with pytest.raises(SystemExit, match="Unknown ingester"):
        run_ingest.select_ingesters(["freddd"])


def test_fred_is_registered():
    assert "fred" in run_ingest.INGESTERS
    assert set(run_ingest.select_ingesters(["fred"])) == {"fred"}


def _stub(
    df=None,
    error: Exception | None = None,
    tables: dict | None = None,
    partial: list[str] | None = None,
):
    """A registry entry: always available, fetch() returns df or raises.

    Subclasses Ingester so the stub inherits the real fetch_tables() default; a bare
    duck-typed object would have hidden a missing hook rather than exercising it.
    """
    class _Stub(Ingester):
        source = "stub"

        def fetch(self):
            if error is not None:
                raise error
            return df

        def fetch_tables(self):
            return tables or {}

        def partial_failures(self):
            return partial or []

    return (lambda _s: True, lambda _s: _Stub())


def test_select_ingesters_defaults_to_all(monkeypatch):
    monkeypatch.setattr(run_ingest, "INGESTERS", {"a": _stub(), "b": _stub()})
    assert set(run_ingest.select_ingesters(None)) == {"a", "b"}
    assert set(run_ingest.select_ingesters(["a"])) == {"a"}


def test_ingester_without_prerequisites_is_skipped_not_failed(isolated_db, monkeypatch):
    """A missing optional key must not block the sources that are configured."""
    good = pd.DataFrame([{
        "source": "good", "series_id": "SPY:close_raw", "ts": "2026-01-02",
        "ts_release": "2026-01-02", "value": 500.0,
    }])
    unconfigured = (lambda _s: False, lambda _s: None)
    monkeypatch.setattr(
        run_ingest, "INGESTERS", {"good": _stub(good), "unconfigured": unconfigured}
    )
    assert run_ingest.main([]) == 0

    from db import loader
    conn = loader.connect(isolated_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
    finally:
        conn.close()


def test_side_tables_are_routed_to_their_loader(isolated_db, monkeypatch):
    """An ingester's non-observation rows reach their own table (e.g. corporate actions)."""
    df = pd.DataFrame([{
        "source": "yfinance", "series_id": "AAPL:close_raw", "ts": "2026-01-02",
        "ts_release": "2026-01-02", "value": 499.23,
    }])
    actions = [{
        "action_id": "AAPL:split:2020-08-31", "cik": None, "ticker": "AAPL",
        "kind": "split", "ex_date": "2020-08-31T00:00:00+00:00", "ratio": 4.0,
        "amount": None, "currency": None, "source": "yfinance",
    }]
    monkeypatch.setattr(
        run_ingest, "INGESTERS", {"prices": _stub(df, tables={"corporate_actions": actions})}
    )
    assert run_ingest.main([]) == 0

    from db import loader
    conn = loader.connect(isolated_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM corporate_actions").fetchone()[0] == 1
        assert conn.execute("SELECT ratio FROM corporate_actions").fetchone()[0] == 4.0
    finally:
        conn.close()


def test_rows_for_an_unknown_table_fail_loudly(isolated_db, monkeypatch):
    """Routing rows to a table with no loader must raise, never drop them silently."""
    df = pd.DataFrame(columns=["source", "series_id", "ts", "ts_release", "value"])
    monkeypatch.setattr(
        run_ingest, "INGESTERS", {"x": _stub(df, tables={"not_a_table": [{"a": 1}]})}
    )
    assert run_ingest.main([]) == 1


def test_a_failing_source_does_not_abort_the_run_but_sets_exit_code(isolated_db, monkeypatch):
    """One dead source must not lose the data from the healthy ones, yet the run
    reports failure so the orchestrator notices (section 8, phase 4)."""
    good = pd.DataFrame([{
        "source": "good", "series_id": "SPY:close_raw", "ts": "2026-01-02",
        "ts_release": "2026-01-02", "value": 500.0,
    }])
    monkeypatch.setattr(run_ingest, "INGESTERS", {
        "good": _stub(good),
        "broken": _stub(error=RuntimeError("upstream structure changed")),
    })
    assert run_ingest.main([]) == 1

    from db import loader
    conn = loader.connect(isolated_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
    finally:
        conn.close()


def test_partial_failure_loads_the_good_data_and_still_fails_the_run(isolated_db, monkeypatch):
    """A source that lost one series keeps the other ten, but the run reports failure.

    A partial run that exits 0 is indistinguishable from a clean one, which is how a
    silently shrinking macro table survives for months.
    """
    good = pd.DataFrame([{
        "source": "fred", "series_id": "CPIAUCSL", "ts": "2026-01-01",
        "ts_release": "2026-02-11", "value": 317.6,
    }])
    monkeypatch.setattr(
        run_ingest, "INGESTERS", {"fred": _stub(good, partial=["T10Y2Y"])}
    )
    assert run_ingest.main([]) == 1

    from db import loader
    conn = loader.connect(isolated_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
    finally:
        conn.close()


def test_a_failed_source_is_recorded_as_an_alert(isolated_db, monkeypatch):
    """Phase 4: "any ingester failing" alerts. Without a bot it is recorded as undelivered,
    so it still goes out on the first run that has one — and the exit code is unchanged."""
    monkeypatch.setattr(run_ingest, "INGESTERS", {
        "ibkr": _stub(error=RuntimeError("Flex token expired")),
    })
    assert run_ingest.main([]) == 1

    import json

    from db import loader
    conn = loader.connect(isolated_db)
    try:
        rows = conn.execute("SELECT alert_id, payload FROM alerts_log").fetchall()
    finally:
        conn.close()
    assert len(rows) == 1 and rows[0][0].startswith("ingest_failure:ibkr:")
    assert json.loads(rows[0][1])["delivered"] is False


def test_a_broken_alert_path_does_not_mask_the_ingest_result(isolated_db, monkeypatch):
    import alerts.rules

    def explode(*a, **k):
        raise RuntimeError("telegram down")

    monkeypatch.setattr(alerts.rules, "dispatch", explode)
    monkeypatch.setattr(run_ingest, "INGESTERS", {"x": _stub(error=RuntimeError("boom"))})
    assert run_ingest.main([]) == 1
