"""Entry-point tests — the phase-0 acceptance criterion (CLAUDE.md section 8)."""

from __future__ import annotations

import pandas as pd
import pytest

import run_ingest
from core import config


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


def _stub(df=None, error: Exception | None = None):
    """A registry entry: always available, fetch() returns df or raises."""
    class _Stub:
        def fetch(self):
            if error is not None:
                raise error
            return df
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
