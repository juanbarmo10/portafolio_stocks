"""The public copy (section 8 phase 5): only what the public panel shows, never the account.

The local database is seeded with private rows of every kind, so the "nothing private
leaves" checks are not vacuous. The target is a SQLite file: the sync writes through the
same loader to either backend.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from core.config import Settings, load_settings
from db import loader
from db import public_sync as ps
from db.database import connect_url, read_observations, read_table
from transform.breadth import BreadthReading, readings_from_observations, readings_to_observations

HIMS, TMUS = "0001773751", "0001283699"


def settings_with_watchlist() -> Settings:
    base = load_settings()
    raw = json.loads(json.dumps(base.raw, default=str))
    raw["universe"]["watchlist"] = [{"ticker": "HIMS", "cik": HIMS}]
    return Settings(raw=raw, db_path=base.db_path, log_level="INFO", secrets={},
                    public_mode=False)


def obs(source, series_id, ts="2026-09-20", release=None):
    return {"source": source, "series_id": series_id, "ts": ts,
            "ts_release": release or ts, "value": 1.0}


@pytest.fixture()
def local(tmp_path):
    conn = loader.init_db(tmp_path / "local.db")
    loader.upsert_observations(conn, pd.DataFrame([
        obs("fred", "VIXCLS"), obs("sec", f"{HIMS}:revenue:q"), obs("sec", f"{TMUS}:revenue:q"),
        obs("yfinance", "SPY:close_raw"), obs("yfinance", "HIMS:close_raw"),
        obs("yfinance", "TMUS:close_raw"), obs("yfinance", "AAPL:close_raw"),
        obs("ibkr", "TMUS:position_qty"), obs("ibkr", "NAV:total"),
        obs("finra", "HIMS:short_interest"), obs("banrep", "TRM:X"),
        obs("fred", "OLD", ts="2010-01-01", release="2010-01-05"),
        obs("sec", f"{HIMS}:revenue:q", ts="2020-03-31", release="2026-09-18"),
    ]))
    loader.upsert_companies(conn, [
        {"cik": HIMS, "ticker": "HIMS", "name": "Hims", "sector": None, "thesis_category": None,
         "first_seen": "2026-09-24", "status": "active"},
        {"cik": TMUS, "ticker": "TMUS", "name": "T-Mobile", "sector": None,
         "thesis_category": None, "first_seen": "2026-09-24", "status": "active"},
    ])
    loader.upsert_filings(conn, [
        {"accession": "a", "cik": HIMS, "form": "10-Q", "period_end": None,
         "filed_date": "2026-08-10", "is_amended": 0, "url": None},
        {"accession": "b", "cik": TMUS, "form": "10-Q/A", "period_end": None,
         "filed_date": "2020-08-10", "is_amended": 1, "url": None},
    ])
    loader.upsert_events(conn, [
        {"event_id": "m", "category": "macro", "cik": None, "ts": "2026-09-30",
         "is_estimated": 0, "label": "PCE", "payload": "{}"},
        {"event_id": "h", "category": "earnings", "cik": HIMS, "ts": "2026-11-02",
         "is_estimated": 1, "label": "HIMS", "payload": "{}"},
        {"event_id": "t", "category": "earnings", "cik": TMUS, "ts": "2026-10-22",
         "is_estimated": 1, "label": "TMUS", "payload": "{}"},
    ])
    loader.upsert_corporate_actions(conn, [
        {"action_id": "s", "cik": None, "ticker": "SPY", "kind": "dividend",
         "ex_date": "2026-06-20", "ratio": None, "amount": 1.8, "currency": "USD",
         "source": "yfinance"},
        {"action_id": "t", "cik": None, "ticker": "TMUS", "kind": "dividend",
         "ex_date": "2026-06-01", "ratio": None, "amount": 1.0, "currency": "USD",
         "source": "yfinance"},
    ])
    yield conn
    conn.close()


def test_only_public_rows_are_selected(local):
    settings = settings_with_watchlist()
    sel = ps.select(local, settings)
    series = set(sel.observations["series_id"])
    assert {"VIXCLS", f"{HIMS}:revenue:q", "SPY:close_raw", "HIMS:close_raw"} <= series
    assert not {"TMUS:position_qty", "NAV:total", "HIMS:short_interest", "TRM:X"} & series
    assert "TMUS:close_raw" not in series, "a held-only price would reveal the holding"
    assert f"{TMUS}:revenue:q" not in series and "AAPL:close_raw" not in series
    assert [r["cik"] for r in sel.tables["companies"]] == [HIMS]
    assert [r["accession"] for r in sel.tables["filings"]] == ["a"]
    assert {r["event_id"] for r in sel.tables["events"]} == {"m", "h"}
    assert [r["ticker"] for r in sel.tables["corporate_actions"]] == ["SPY"]


def test_a_numeric_null_stays_null_not_nan(local):
    row = ps.select(local, settings_with_watchlist()).tables["corporate_actions"][0]
    assert row["ratio"] is None


def test_the_incremental_window_keeps_a_recent_restatement_of_an_old_period(local):
    """A 2020 quarter re-filed last week has an old ts and a new ts_release."""
    sel = ps.select(local, settings_with_watchlist(), since="2026-09-10")
    kept = set(zip(sel.observations["series_id"], sel.observations["ts"].str[:10]))
    assert (f"{HIMS}:revenue:q", "2020-03-31") in kept
    assert ("OLD", "2010-01-01") not in kept


def test_the_guard_refuses_what_the_filter_should_never_let_through(local):
    settings = settings_with_watchlist()
    sel = ps.select(local, settings)
    sel.observations = pd.concat([sel.observations, pd.DataFrame([obs("ibkr", "NAV:total")])])
    with pytest.raises(RuntimeError, match="private sources"):
        ps.assert_public(sel, settings, held={"TMUS"})
    sel = ps.select(local, settings)
    sel.tables["companies"].append({"cik": TMUS, "ticker": "TMUS"})
    with pytest.raises(RuntimeError, match="non-researched"):
        ps.assert_public(sel, settings, held={"TMUS"})


def test_the_copy_is_written_and_idempotent(local, tmp_path):
    settings = settings_with_watchlist()
    sel = ps.select(local, settings)
    ps.assert_public(sel, settings, held={"TMUS"})
    target = connect_url(f"sqlite:///{tmp_path / 'public.db'}")
    try:
        ps.write(target, sel)
        ps.write(target, sel)
        copied = read_observations(target)
        assert len(copied) == len(sel.observations)
        assert set(copied["source"]) == {"fred", "sec", "yfinance"}
        assert read_table(target, "trades").empty and read_table(target, "securities").empty
    finally:
        target.close()


def test_breadth_readings_survive_the_trip():
    """The public copy carries the computed series; the page must read back the same
    readings, coverage and holes included, and a missing breadth stays missing."""
    readings = [
        BreadthReading("2026-09-21", 503, 500, 480, 250, 250 / 480, 480 / 503, True),
        BreadthReading("2026-09-22", 503, 70, 60, 30, None, 60 / 503, False, source_hole=True),
    ]
    back = readings_from_observations(readings_to_observations(readings))
    assert back[0] == readings[0]
    assert back[1].breadth is None and back[1].source_hole and not back[1].meets_threshold
