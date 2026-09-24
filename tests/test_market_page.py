"""The market page: the regime light with every vote behind it (sections 2, 8 phase 3).

Rendered end to end against a synthetic database built so the verdict is known in
advance — every component risk-on, or every component risk-off — because what is tested is
the page's wiring and honesty, not the market.

Skipped when the ``app`` extra is not installed.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit", reason="the 'app' extra is not installed")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import data as app_data  # noqa: E402
from app.format import PUBLIC_PAGES  # noqa: E402
from db import loader  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = str(REPO_ROOT / "app" / "main.py")
MARKET = str(REPO_ROOT / "app" / "pages" / "market.py")

DAYS = pd.bdate_range("2024-01-01", periods=320)
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


def seed(db_path, *, healthy: bool) -> None:
    """Every component pointing the same way: all risk-on, or all risk-off."""
    up = np.linspace(100, 150, len(DAYS))
    down = np.linspace(150, 100, len(DAYS))
    good, bad = (up, down) if healthy else (down, up)

    def daily(series_id, values):
        return [{"source": "fred", "series_id": series_id, "ts": d.date().isoformat(),
                 "ts_release": d.date().isoformat(), "value": float(v)}
                for d, v in zip(DAYS, values)]

    level = -0.5 if healthy else 0.5
    rows = [
        *daily("NFCI", np.full(len(DAYS), level)),                  # tight when positive
        *daily("T10Y2Y", np.full(len(DAYS), -level)),               # inverted when negative
        *daily("VIXCLS", np.full(len(DAYS), 15.0 if healthy else 30.0)),
        *daily("VXVCLS", np.full(len(DAYS), 18.0 if healthy else 25.0)),
        *daily("BAA10Y", bad / 50),                                  # widening is risk-off
        *daily("DTWEXBGS", bad),                                     # strengthening is off
        *daily("WALCL", good * 1e4),                                 # contracting is off
        *daily("WTREGEN", np.full(len(DAYS), 1e5)),
        *daily("WLRRAL", np.full(len(DAYS), 1e5)),
    ]
    for ticker in SECTORS:
        rows += [{"source": "yfinance", "series_id": f"{ticker}:close_raw",
                  "ts": d.date().isoformat(), "ts_release": d.date().isoformat(),
                  "value": float(v)} for d, v in zip(DAYS, good)]
    # RSP outrunning SPY is a broadening market; the reverse is narrowing.
    rsp = good * (np.linspace(1.0, 1.2, len(DAYS)) if healthy else np.linspace(1.2, 1.0, len(DAYS)))
    for ticker, values in (("RSP", rsp), ("SPY", good)):
        rows += [{"source": "yfinance", "series_id": f"{ticker}:close_raw",
                  "ts": d.date().isoformat(), "ts_release": d.date().isoformat(),
                  "value": float(v)} for d, v in zip(DAYS, values)]

    conn = loader.init_db(db_path)
    try:
        loader.upsert_observations(conn, pd.DataFrame(rows))
    finally:
        conn.close()


@pytest.fixture()
def market_db(tmp_path, monkeypatch):
    db_path = tmp_path / "market.db"
    monkeypatch.setattr(app_data, "db_path", lambda: db_path)
    st.cache_data.clear()
    yield db_path
    st.cache_data.clear()


def render() -> AppTest:
    app = AppTest.from_file(MAIN, default_timeout=120).run()
    app.switch_page(MARKET)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def verdict(app: AppTest) -> str:
    return next(m.value for m in app.markdown if m.value.startswith("###"))


# --- The verdict ------------------------------------------------------------------------


def test_every_component_healthy_is_green(market_db):
    seed(market_db, healthy=True)
    assert "Risk-on" in verdict(render())


def test_every_component_stressed_is_red_and_says_it_blocks(market_db):
    seed(market_db, healthy=False)
    app = render()
    assert "Risk-off" in verdict(app)
    assert "no se compra" in verdict(app)


def test_every_vote_is_on_the_page(market_db):
    """A verdict nobody can take apart is a verdict nobody can check."""
    seed(market_db, healthy=True)
    table = pd.DataFrame(render().dataframe[0].value)

    assert len(table) == 8
    assert {"Componente", "Valor", "Comparado con", "Voto", "Dato del"} <= set(table.columns)
    assert not table["Voto"].str.contains("abstiene").any()


def test_the_page_says_green_is_not_a_buy_signal(market_db):
    seed(market_db, healthy=True)
    captions = " ".join(c.value for c in render().caption)
    assert "Bloquea, no dispara" in captions
    assert "no retocadas" in captions, "the rules must say they were fixed in advance"


def test_numbers_are_in_spanish_notation(market_db):
    """No bare f-string: the panel reads 14,21 everywhere, never 14.21 next to 24,8 %."""
    seed(market_db, healthy=True)
    app = render()
    vix = next(m.value for m in app.metric if m.label.startswith("VIX (30"))
    assert vix == "15,00"


# --- Empty and private ------------------------------------------------------------------


def test_an_empty_database_explains_what_to_run(market_db):
    text = " ".join(i.value for i in render().info)
    assert "run_ingest.py" in text


def test_the_page_is_private_until_the_user_decides():
    """Section 13, the user's decision of 2026-09-23: built pages start private."""
    assert "Mercado" not in PUBLIC_PAGES
