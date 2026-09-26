"""The decision journal (CLAUDE.md §15.1.10), on synthetic trades with known answers."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from db import loader
from transform import journal as jr


def trade(tid, ts, ticker, side, qty, price, commission=-0.35):
    return {"trade_id": tid, "conid": "1", "cik": None, "ticker": ticker, "ts": ts,
            "side": side, "quantity": qty, "price": price, "currency": "USD",
            "commission": commission, "fx_rate": 1.0}


TRADES = pd.DataFrame([
    trade("1", "2026-02-12T15:00:00+00:00", "AAA", "buy", 0.5, 100.0),
    trade("2", "2026-02-12T15:01:00+00:00", "AAA", "buy", 1.5, 104.0),   # same decision
    trade("3", "2026-03-01T15:00:00+00:00", "AAA", "sell", 1.0, 120.0),
])


def test_executions_of_one_day_ticker_and_side_are_one_decision():
    d = jr.decisions(TRADES).set_index(["date", "side"])
    buy = d.loc[("2026-02-12", "buy")]
    assert buy["quantity"] == 2.0 and buy["executions"] == 2
    assert buy["price"] == pytest.approx((50 + 156) / 2), "value-weighted average"
    assert buy["commission"] == pytest.approx(-0.70)


def test_an_entry_finds_its_decision_within_the_tolerance_and_a_plan_stays_visible():
    entries = [{"date": "2026-02-11", "ticker": "aaa", "why": "barata"},       # the eve
               {"date": "2026-05-01", "ticker": "BBB", "why": "plan"}]         # not done
    table = jr.match_entries(jr.decisions(TRADES), entries, tolerance_days=3)
    buy = table[(table["side"] == "buy")].iloc[0]
    assert buy["entry"]["why"] == "barata"
    sell = table[table["side"] == "sell"].iloc[0]
    assert sell["entry"] is None, "a decision without a written reason is flagged"
    plan = table[table["side"] == "sin ejecutar"].iloc[0]
    assert plan["ticker"] == "BBB"


def test_the_outcome_is_measured_from_the_decision_day():
    series = pd.Series([100.0, 110.0, 121.0],
                       index=pd.to_datetime(["2026-02-11", "2026-02-12", "2026-03-01"]))
    assert jr.since(series, "2026-02-12", "2026-03-01") == pytest.approx(0.10)
    frame = pd.DataFrame({"verdict": ["risk_on", "risk_off"]},
                         index=pd.to_datetime(["2026-01-01", "2026-02-20"]))
    assert jr.verdict_on(frame, "2026-02-12") == "risk_on", "the light in force that day"


def test_a_review_comes_due_on_its_own_date():
    assert jr.review_due({"review_after": "2026-08-12"}, "2026-08-12")
    assert not jr.review_due({"review_after": "2026-08-12"}, "2026-08-11")
    assert not jr.review_due(None, "2030-01-01")


# --- The page -----------------------------------------------------------------------------

st = pytest.importorskip("streamlit", reason="the 'app' extra is not installed")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import data as app_data  # noqa: E402
from app.format import PUBLIC_PAGES  # noqa: E402
from core import config  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = str(REPO_ROOT / "app" / "main.py")
PAGE = str(REPO_ROOT / "app" / "pages" / "journal.py")


def test_the_page_flags_a_decision_without_a_reason_and_hands_the_template(tmp_path,
                                                                            monkeypatch):
    local = tmp_path / "settings.local.yaml"
    local.write_text('journal:\n  - {date: 2026-02-12, ticker: AAA, why: "barata", '
                     'review_after: 2026-03-01}\n', encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", local)
    config.load_settings.cache_clear()
    db_path = tmp_path / "journal.db"
    conn = loader.init_db(db_path)
    loader.upsert_trades(conn, TRADES)
    conn.close()
    monkeypatch.setattr(app_data, "db_path", lambda: db_path)
    st.cache_data.clear()
    app = AppTest.from_file(MAIN, default_timeout=120).run()
    app.switch_page(PAGE)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    labels = [e.label for e in app.expander]
    assert any("Venta AAA" in label and "sin razón escrita" in label for label in labels)
    assert any("Compra AAA" in label and "toca revisar" in label for label in labels)
    assert any("ticker: AAA" in c.value for c in app.code), "the template, prefilled"
    mirror = next(pd.DataFrame(d.value) for d in app.dataframe
                  if "Lo que hizo la cuenta" in pd.DataFrame(d.value).columns)
    rows = dict(zip(mirror["Qué"], mirror["Lo que hizo la cuenta"]))
    assert rows["Qué se vende"].startswith("1 ventas con ganancia y 0 con pérdida")
    assert rows["Razones escritas"] == "1 de 2 decisiones"
    st.cache_data.clear()
    config.load_settings.cache_clear()


def test_the_journal_is_never_public():
    assert "Diario" not in PUBLIC_PAGES
