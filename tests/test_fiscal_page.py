"""The fiscal page (section 11): local only, ESTIMADO, and "no calculable" without the
owner's parameters. Seeded with a synthetic currency — the real one is local config."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

pytest.importorskip("streamlit", reason="the 'app' extra is not installed")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import data as app_data  # noqa: E402
from core import config  # noqa: E402
from db import loader  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = str(REPO_ROOT / "app" / "main.py")
FISCAL = str(REPO_ROOT / "app" / "pages" / "fiscal.py")

LOCAL_YAML = """
fiscal:
  local_fx:
    url: http://127.0.0.1:1/rates
    currency: XYZ
    series_id: "FX:XYZ"
    source: test_fx
"""


@pytest.fixture()
def fiscal_db(tmp_path, monkeypatch):
    db_path = tmp_path / "fiscal.db"
    monkeypatch.setattr(app_data, "db_path", lambda: db_path)
    st.cache_data.clear()
    yield db_path
    st.cache_data.clear()


def with_local_config(tmp_path, monkeypatch):
    path = tmp_path / "local.yaml"
    path.write_text(LOCAL_YAML, encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", path)
    config.load_settings.cache_clear()


def seed(db_path):
    obs = [
        ("ibkr", "NAV:total", "2025-03-03", 110.0),
        ("ibkr", "ZZZ:position_qty", "2025-03-03", 1.0),
        ("ibkr", "ZZZ:position_cost_basis", "2025-03-03", 101.0),
        ("yfinance", "ZZZ:close_raw", "2025-03-03", 110.0),
        ("test_fx", "FX:XYZ", "2025-01-02", 10.0),
        ("test_fx", "FX:XYZ", "2025-03-03", 8.0),
    ]
    conn = loader.init_db(db_path)
    try:
        loader.upsert_observations(conn, pd.DataFrame([
            {"source": s, "series_id": i, "ts": t, "ts_release": t, "value": v}
            for s, i, t, v in obs
        ]))
        loader.upsert_trades(conn, pd.DataFrame([{
            "trade_id": "1", "conid": None, "cik": None, "ticker": "ZZZ",
            "ts": "2025-01-02T15:00:00+00:00", "side": "buy", "quantity": 1.0,
            "price": 100.0, "currency": "USD", "commission": -1.0, "fx_rate": 1.0,
        }]))
    finally:
        conn.close()


def render():
    app = AppTest.from_file(MAIN, default_timeout=120).run()
    app.switch_page(FISCAL)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def test_without_the_local_rate_the_page_says_how_to_configure_it(fiscal_db):
    text = " ".join(i.value for i in render().info)
    assert "fiscal.local_fx" in text


def test_the_page_converts_and_splits_and_says_estimado(fiscal_db, tmp_path, monkeypatch):
    with_local_config(tmp_path, monkeypatch)
    seed(fiscal_db)
    app = render()
    assert "ESTIMADO" in app.warning[0].value
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Costo (congelado)"] == "1.010,00 XYZ", "101 USD at the purchase-day 10"
    assert metrics["Valor hoy"] == "880,00 XYZ", "110 USD at today's 8"
    # +9 USD in dollars, a loss in the local currency: the case the layer exists for.
    assert metrics["Resultado"] == "-130,00 XYZ"


def test_no_holding_period_means_no_calculable(fiscal_db, tmp_path, monkeypatch):
    with_local_config(tmp_path, monkeypatch)
    seed(fiscal_db)
    table = pd.DataFrame(render().dataframe[0].value)
    assert set(table["Cumple el periodo"]) == {"no calculable"}
