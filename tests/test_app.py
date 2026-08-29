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
    table = at.dataframe[0].value
    assert "Curva 10a − 2a" in set(table["Serie"])
    assert float(table.loc[table["Serie"] == "Curva 10a − 2a", "Valor"].iloc[0]) == 0.39


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
    table = at.dataframe[0].value
    missing = table[table["Serie"] == "VIX"]
    assert len(missing) == 1, "the series vanished instead of showing as a hole"
    assert pd.isna(missing["Valor"].iloc[0])


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
