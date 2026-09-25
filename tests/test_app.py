"""Smoke tests for the Streamlit pages (CLAUDE.md sections 3, 7).

These do not check layout — they check that a page renders without raising, both against
an empty database and against a populated one. The empty path is the likeliest regression
because it is the one nobody looks at after the first week, and it is also the one a new
clone of the repo hits first.

Skipped when the ``app`` extra is not installed; CI installs ``[dev,ibkr]`` only.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

pytest.importorskip("streamlit", reason="the 'app' extra is not installed")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import data as app_data  # noqa: E402
from db import loader  # noqa: E402

# AppTest resolves a relative script path against *this* file, not the repo root.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = str(REPO_ROOT / "app" / "main.py")
PORTFOLIO = str(REPO_ROOT / "app" / "pages" / "portfolio.py")


@pytest.fixture()
def app_db(tmp_path, monkeypatch):
    """Point the pages at a throwaway database and drop Streamlit's data cache."""
    db_path = tmp_path / "app.db"
    monkeypatch.setattr(app_data, "db_path", lambda: db_path)
    st.cache_data.clear()
    yield db_path
    st.cache_data.clear()


def _run() -> AppTest:
    at = AppTest.from_file(MAIN, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_today_page_survives_an_empty_database(app_db):
    """No data must produce an actionable warning, never a traceback or an empty chart."""
    at = _run()
    assert at.title[0].value == "🏠 Hoy"
    assert any("run_ingest" in w.value for w in at.warning), "the fix is not spelled out"


def test_today_page_renders_the_level_1_table(app_db):
    """With macro rows present the page shows the reading table, point-in-time."""
    conn = loader.init_db(app_db)
    try:
        loader.upsert_observations(conn, pd.DataFrame([
            {"source": "fred", "series_id": "T10Y2Y", "ts": "2026-08-27",
             "ts_release": "2026-08-28", "value": 0.39},
            {"source": "fred", "series_id": "CPIAUCSL", "ts": "2026-07-01",
             "ts_release": "2026-08-12", "value": 332.813},
        ]))
    finally:
        conn.close()

    at = _run()
    table = next(d.value for d in at.dataframe if "Serie" in d.value.columns)
    assert "Curva 10a − 2a" in set(table["Serie"])
    # Spanish notation, like the rest of the panel (it read 0.390 until 2026-09-25).
    assert table.loc[table["Serie"] == "Curva 10a − 2a", "Valor"].iloc[0] == "0,390"


def test_a_series_with_no_data_is_a_visible_hole(app_db):
    """A configured series with nothing published keeps its row, empty (section 12)."""
    conn = loader.init_db(app_db)
    try:
        loader.upsert_observations(conn, pd.DataFrame([
            {"source": "fred", "series_id": "T10Y2Y", "ts": "2026-08-27",
             "ts_release": "2026-08-28", "value": 0.39},
        ]))
    finally:
        conn.close()

    at = _run()
    table = next(d.value for d in at.dataframe if "Serie" in d.value.columns)
    missing = table[table["Serie"] == "VIX"]
    assert len(missing) == 1, "the series vanished instead of showing as a hole"
    assert missing["Valor"].iloc[0] == "—", "a hole is shown as a hole, never as a zero"


def test_portfolio_page_says_what_is_missing_instead_of_showing_nothing(app_db):
    """Without Flex credentials the page must explain the unblocking steps."""
    at = AppTest.from_file(MAIN, default_timeout=60)
    at.run()
    at.switch_page(PORTFOLIO)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("IBKR" in w.value for w in at.warning)


def test_charts_never_put_two_scales_on_one_axis(app_db):
    """Every chart carries a single y encoding — a dual axis is the classic viz lie.

    Also fixes the palette contract: colour is assigned by series identity from a fixed
    order, so adding or filtering a series never repaints the survivors.
    """
    import json

    conn = loader.init_db(app_db)
    try:
        rows = []
        for series_id, value in (("BAA10Y", 1.6), ("BAMLH0A0HYM2", 2.63), ("T10Y2Y", 0.39)):
            for day in ("2026-08-26", "2026-08-27"):
                rows.append({"source": "fred", "series_id": series_id, "ts": day,
                             "ts_release": day, "value": value})
        loader.upsert_observations(conn, pd.DataFrame(rows))
    finally:
        conn.close()

    at = _run()
    charts = at.get("vega_lite_chart")
    assert len(charts) == 2, "a configured chart did not render"

    for chart in charts:
        spec = json.loads(chart.proto.spec)
        for layer in spec.get("layer", [spec]):
            y = layer.get("encoding", {}).get("y", {})
            assert not isinstance(y, list), "two y encodings means a dual axis"

    credit = json.loads(charts[0].proto.spec)["layer"][0]["encoding"]["color"]["scale"]
    assert credit["range"] == ["#2a78d6", "#eb6834"], "the palette slots drifted"
    assert credit["domain"][0] == "Prima crédito Baa − 10a", "series order is not fixed"


# --- Section 9.2 reaches the screen -------------------------------------------


def test_a_contradicted_corporate_action_reaches_the_portfolio_page(app_db):
    """The flag is only worth computing if the operator sees it before trusting a number.

    A spin-off IBKR calls a spin-off and yfinance calls a split, on a ticker that is held:
    the quantity and the cost base of that position are in doubt until a human looks.
    """
    conn = loader.init_db(app_db)
    try:
        loader.upsert_observations(conn, pd.DataFrame([
            {"source": "ibkr", "series_id": "XLF:position_qty", "ts": "2026-09-15",
             "ts_release": "2026-09-15", "value": 4.0},
            {"source": "ibkr", "series_id": "XLF:position_cost_basis", "ts": "2026-09-15",
             "ts_release": "2026-09-15", "value": 160.0},
            {"source": "yfinance", "series_id": "XLF:close_raw", "ts": "2026-09-15",
             "ts_release": "2026-09-15", "value": 45.0},
            {"source": "ibkr", "series_id": "NAV:stock", "ts": "2026-09-15",
             "ts_release": "2026-09-15", "value": 180.0},
        ]))
        loader.upsert_corporate_actions(conn, [
            {"action_id": "ibkr:1", "cik": None, "ticker": "XLF", "kind": "spinoff",
             "ex_date": "2016-09-19", "ratio": None, "amount": None, "currency": "USD",
             "source": "ibkr"},
            {"action_id": "yf:1", "cik": None, "ticker": "XLF", "kind": "split",
             "ex_date": "2016-09-19", "ratio": 1.231, "amount": None, "currency": None,
             "source": "yfinance"},
        ])
    finally:
        conn.close()

    at = AppTest.from_file(MAIN, default_timeout=60).run()
    at.switch_page(PORTFOLIO)
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    errors = " ".join(element.value for element in at.get("error"))
    assert "XLF — revisar" in errors
    assert "Manda IBKR" in errors, "the panel must say which source wins"
    assert "no se corrige solo" in errors, "it must say it does not fix it by itself"


def test_a_portfolio_with_no_contradiction_raises_nothing(app_db):
    """Guards the guard: a flag that always fires is a flag nobody reads."""
    conn = loader.init_db(app_db)
    try:
        loader.upsert_observations(conn, pd.DataFrame([
            {"source": "ibkr", "series_id": "AAPL:position_qty", "ts": "2026-09-15",
             "ts_release": "2026-09-15", "value": 4.0},
            {"source": "ibkr", "series_id": "AAPL:position_cost_basis", "ts": "2026-09-15",
             "ts_release": "2026-09-15", "value": 160.0},
            {"source": "yfinance", "series_id": "AAPL:close_raw", "ts": "2026-09-15",
             "ts_release": "2026-09-15", "value": 45.0},
            {"source": "ibkr", "series_id": "NAV:stock", "ts": "2026-09-15",
             "ts_release": "2026-09-15", "value": 180.0},
        ]))
        loader.upsert_corporate_actions(conn, [
            {"action_id": "ibkr:1", "cik": None, "ticker": "AAPL", "kind": "split",
             "ex_date": "2020-08-31", "ratio": None, "amount": None, "currency": "USD",
             "source": "ibkr"},
            {"action_id": "yf:1", "cik": None, "ticker": "AAPL", "kind": "split",
             "ex_date": "2020-08-31", "ratio": 4.0, "amount": None, "currency": None,
             "source": "yfinance"},
        ])
    finally:
        conn.close()

    at = AppTest.from_file(MAIN, default_timeout=60).run()
    at.switch_page(PORTFOLIO)
    at.run()

    errors = " ".join(element.value for element in at.get("error"))
    assert "revisar" not in errors


# --- The cache must notice a new ingest -------------------------------------------


def test_a_new_ingest_reaches_a_running_panel(app_db):
    """The cache key has to include the database mtime.

    It used to be passed as ``_mtime``, and Streamlit leaves underscored arguments out of
    the key — that is their purpose. A running panel kept serving the first read until it
    was restarted, and every test missed it because each one clears the cache first.
    """
    conn = loader.init_db(app_db)
    try:
        loader.upsert_observations(conn, pd.DataFrame([{
            "source": "fred", "series_id": "VIXCLS", "ts": "2026-09-01",
            "ts_release": "2026-09-01", "value": 15.0,
        }]))
    finally:
        conn.close()
    first = app_data.observations("fred", 1.0)

    conn = loader.init_db(app_db)
    try:
        loader.upsert_observations(conn, pd.DataFrame([{
            "source": "fred", "series_id": "VIXCLS", "ts": "2026-09-02",
            "ts_release": "2026-09-02", "value": 16.0,
        }]))
    finally:
        conn.close()
    second = app_data.observations("fred", 2.0)

    assert len(first) == 1
    assert len(second) == 2, "the cache served the old read after a new ingest"


# --- The landing page summarizes the rest of the checklist (2026-09-25) -----------------


def _seed_upcoming(db_path):
    """A macro release in three days, and results of a held company with no card."""
    import json

    today = pd.Timestamp.now(tz="UTC").normalize()
    conn = loader.init_db(db_path)
    try:
        loader.upsert_observations(conn, pd.DataFrame([
            {"source": "fred", "series_id": "T10Y2Y", "ts": "2026-08-27",
             "ts_release": "2026-08-28", "value": 0.39},
            {"source": "ibkr", "series_id": "ZZHELD:position_qty",
             "ts": today.date().isoformat(), "ts_release": today.date().isoformat(),
             "value": 1.0},
        ]))
        loader.upsert_companies(conn, [{"cik": "0000000042", "ticker": "ZZHELD", "name": "Z",
                                        "sector": None, "thesis_category": None,
                                        "first_seen": "2026-01-01", "status": "active"}])
        loader.upsert_events(conn, [
            {"event_id": "fred:10:x", "category": "macro", "cik": None,
             "ts": (today + pd.Timedelta(days=3, hours=12)).isoformat(), "is_estimated": 0,
             "label": "CPI (inflación)",
             "payload": json.dumps({"calendar_as_of": today.date().isoformat(), "release_id": 10})},
            {"event_id": "0000000042:earnings:next", "category": "earnings",
             "cik": "0000000042", "ts": (today + pd.Timedelta(days=5)).date().isoformat(),
             "is_estimated": 1, "label": "ZZHELD", "payload": "{}"},
        ])
    finally:
        conn.close()


def test_the_landing_page_says_what_is_coming(app_db):
    """It read "sin construir" for three built levels until 2026-09-25."""
    _seed_upcoming(app_db)
    at = _run()
    text = " ".join(m.value for m in at.markdown)
    assert "sin construir" not in text.lower()
    upcoming = next(pd.DataFrame(d.value) for d in at.dataframe
                    if "Qué" in pd.DataFrame(d.value).columns)
    assert "CPI (inflación)" in set(upcoming["Qué"])
    assert "Resultados de ZZHELD" in set(upcoming["Qué"]), "a held company's results"


def test_publicly_the_landing_page_names_no_position(app_db, monkeypatch):
    """The earnings of a company held without a card would name the position."""
    from core import config

    _seed_upcoming(app_db)
    monkeypatch.setenv("PUBLIC_MODE", "1")
    config.load_settings.cache_clear()
    at = _run()
    upcoming = next(pd.DataFrame(d.value) for d in at.dataframe
                    if "Qué" in pd.DataFrame(d.value).columns)
    assert "CPI (inflación)" in set(upcoming["Qué"])
    assert not any("ZZHELD" in str(v) for v in upcoming.to_numpy().ravel())


def test_the_landing_page_leads_with_what_can_change_todays_decision(app_db):
    """§15.2: the four readings that matter as tiles, the eleven series one click away."""
    _seed_upcoming(app_db)
    at = _run()
    headers = [h.value for h in at.subheader]
    assert headers.index("Lo que viene — 14 días") < headers.index("1 · ¿Hay apetito por riesgo?")
    assert any(m.label == "Curva 10a − 2a" for m in at.metric)
    assert at.expander, "the full macro table is folded, not gone"


def test_an_unclassified_reorganization_reaches_the_portfolio_page(app_db):
    """Until 2026-09-25 an IBKR reorg the panel could not classify only reached the log and
    Telegram; the position looked normal on the page."""
    conn = loader.init_db(app_db)
    try:
        loader.upsert_observations(conn, pd.DataFrame([
            {"source": "ibkr", "series_id": "XLF:position_qty", "ts": "2026-09-15",
             "ts_release": "2026-09-15", "value": 4.0},
            {"source": "ibkr", "series_id": "XLF:position_cost_basis", "ts": "2026-09-15",
             "ts_release": "2026-09-15", "value": 160.0},
            {"source": "yfinance", "series_id": "XLF:close_raw", "ts": "2026-09-15",
             "ts_release": "2026-09-15", "value": 45.0},
            {"source": "ibkr", "series_id": "NAV:stock", "ts": "2026-09-15",
             "ts_release": "2026-09-15", "value": 180.0},
        ]))
        loader.upsert_unmapped_actions(conn, [
            {"action_id": "ibkr:9", "ticker": "XLF", "conid": "1", "ex_date": "2026-09-01",
             "code": "STOCKDIV", "description": "XLF STOCK DIVIDEND", "source": "ibkr",
             "first_seen": "2026-09-02"}])
    finally:
        conn.close()
    at = AppTest.from_file(MAIN, default_timeout=60).run()
    at.switch_page(PORTFOLIO)
    at.run()
    errors = " ".join(element.value for element in at.get("error"))
    assert "XLF — revisar" in errors and "STOCKDIV" in errors
