"""Sell discipline and "reinforce before opening" (CLAUDE.md §15.5, point 1).

What is defended: a sale is "backed" only by something the user wrote before (a crossed
invalidation rule, a triggered exit rule), never by the price; unknown is never "intact";
and adding to a position follows the thesis metric the user chose, point-in-time.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from core.config import load_settings
from ingest.sec_xbrl import extract_metric
from transform import behavior as bh
from transform import discipline as dc
from transform import thesis as th

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "sec_companyfacts_msft.json"
CIK = "0000789019"
# The MSFT 10-K for FY2025 lands on 2025-07-30: 100 days earlier the TTM was smaller.
AS_OF = "2025-08-15"


@pytest.fixture(scope="module")
def observations() -> pd.DataFrame:
    cfg = load_settings().source("sec")
    facts = json.loads(FIXTURE.read_text(encoding="utf-8"))["facts"]
    records = []
    for metric, spec in cfg["concepts"].items():
        rows, _ = extract_metric(facts, metric, spec, CIK, cfg["duration_windows"])
        records.extend(rows)
    return pd.DataFrame(records)


def card(**overrides) -> dict:
    base = {"ticker": "MSFT", "cik": CIK, "thesis": "t", "value_accrual": "v",
            "key_metric": "k", "invalidation": "Ingresos TTM por debajo de 1.",
            "review_date": "2027-06-30"}
    base.update(overrides)
    return base


VALUED = pd.DataFrame({"ticker": ["MSFT"], "price": [500.0], "market_value": [1000.0],
                       "unrealized_return": [0.25], "weight": [0.5]})
LOTS = pd.DataFrame({"ticker": ["MSFT", "MSFT"],
                     "ts": ["2025-06-01T15:00:00+00:00", "2025-08-01T15:00:00+00:00"]})


# --- Before selling -------------------------------------------------------------------


def test_without_a_card_there_is_no_written_criterion_either_way():
    check = dc.before_selling("MSFT", None, {}, VALUED, LOTS, AS_OF)
    assert check.backing == dc.NO_CARD
    assert check.held_days == 75 and check.newest_lot_days == 14, "oldest and newest lot"
    assert check.before_horizon is True
    assert check.unrealized_return == pytest.approx(0.25)


def test_a_rise_in_price_alone_never_backs_a_sale(observations):
    """The pattern the account showed: selling a winner because it rose. Nothing written
    is met, so the sale is unbacked — however good the open result looks."""
    fundamentals = th.evaluable_metrics(observations, CIK, AS_OF)
    check = dc.before_selling(
        "MSFT", card(invalidation_rule={"metric": "revenue_ttm", "operator": "<",
                                        "threshold": 1}),
        fundamentals, VALUED, LOTS, AS_OF)
    assert check.invalidation_breached is False
    assert check.backing == dc.UNBACKED
    assert any("Nada de lo que escribiste" in r for r in check.reasons)


def test_a_crossed_invalidation_rule_backs_the_sale(observations):
    fundamentals = th.evaluable_metrics(observations, CIK, AS_OF)
    check = dc.before_selling(
        "MSFT", card(invalidation_rule={"metric": "revenue_ttm", "operator": ">",
                                        "threshold": 1}),
        fundamentals, VALUED, LOTS, AS_OF)
    assert check.invalidation_breached is True and check.backing == dc.BACKED


def test_exit_rules_use_the_position_metrics_and_prose_is_left_to_the_user():
    ladder = [
        {"rule_id": "tp", "kind": "rebalance", "trigger": "pesa más del 40 %",
         "action": "vender hasta el 30 %", "rule": {"metric": "weight", "operator": ">",
                                                    "threshold": 0.4}},
        {"rule_id": "words", "kind": "take_profit", "trigger": "cuando lo decida",
         "action": "vender la mitad"},
        {"rule_id": "odd", "kind": "take_profit", "trigger": "x", "action": "y",
         "rule": {"metric": "no_such_metric", "operator": ">", "threshold": 1}},
    ]
    check = dc.before_selling("MSFT", card(exit_ladder=ladder), {}, VALUED, LOTS, AS_OF)
    assert [r["rule_id"] for r in check.exit_triggered] == ["tp"]
    assert [r["rule_id"] for r in check.exit_prose] == ["words"]
    assert check.exit_unevaluable == ["odd"], "an unevaluable rule is said, not skipped"
    assert check.backing == dc.BACKED


def test_an_unevaluable_rule_is_not_read_as_intact():
    """NU files IFRS: no SEC figures, so its rule cannot be evaluated — and says so."""
    check = dc.before_selling(
        "NU", card(ticker="NU", invalidation_rule={"metric": "revenue_ttm", "operator": "<",
                                                    "threshold": 1}),
        {}, VALUED, LOTS, AS_OF)
    assert check.invalidation_breached is None
    assert any("no se puede evaluar" in r for r in check.reasons)


# --- Reinforce before opening -----------------------------------------------------------


def reinforce(observations, rule, **kw):
    return dc.reinforce(["MSFT"], {"MSFT": card(invalidation_rule=rule)}, {"MSFT": CIK},
                        observations, VALUED, AS_OF, **kw).iloc[0]


def test_a_metric_moving_away_from_its_threshold_is_improving(observations):
    row = reinforce(observations, {"metric": "revenue_ttm", "operator": "<", "threshold": 1})
    assert row["value_now"] > row["value_before"], "the 10-K of 2025-07-30 is in between"
    assert row["status"] == "improving"


def test_the_same_move_towards_a_ceiling_is_worsening(observations):
    row = reinforce(observations, {"metric": "revenue_ttm", "operator": ">",
                                   "threshold": 1e15})
    assert row["status"] == "worsening"
    assert row["headroom"] > 0, "still on the safe side of the ceiling"


def test_without_a_new_filing_in_between_the_trend_is_flat(observations):
    row = reinforce(observations, {"metric": "revenue_ttm", "operator": "<", "threshold": 1},
                    lookback_days=5)
    assert row["status"] == "flat"


def test_the_comparison_is_point_in_time(observations):
    """Moving ``as_of`` to the day before the 10-K removes it from "now": the TTM of
    2025-07-29 is not the one of 2025-08-15."""
    early = dc.reinforce(["MSFT"], {"MSFT": card(invalidation_rule={
        "metric": "revenue_ttm", "operator": "<", "threshold": 1})}, {"MSFT": CIK},
        observations, VALUED, "2025-07-29").iloc[0]
    late = reinforce(observations, {"metric": "revenue_ttm", "operator": "<", "threshold": 1})
    assert early["value_now"] < late["value_now"]


def test_no_room_under_the_written_cap_puts_it_at_the_limit(observations):
    row = reinforce(observations, {"metric": "revenue_ttm", "operator": "<", "threshold": 1},
                    nav_total=2000.0, max_position=0.4)
    assert row["weight"] == pytest.approx(0.5) and row["room"] == pytest.approx(-0.1)
    assert row["status"] == "at_limit"


def test_the_order_puts_breached_theses_last_and_unknown_is_never_first(observations):
    cards = {"AAA": None, "MSFT": card(invalidation_rule={"metric": "revenue_ttm",
                                                           "operator": ">", "threshold": 1}),
             "NU": card(ticker="NU", cik="0001691493")}
    table = dc.reinforce(["AAA", "MSFT", "NU"], cards, {"MSFT": CIK, "NU": "0001691493"},
                         observations, VALUED, AS_OF)
    assert list(table["ticker"]) == ["NU", "AAA", "MSFT"]
    assert list(table["status"]) == ["unknown", "no_card", "breached"]


# --- The sale history behind the warning ----------------------------------------------


def trade(tid, day, ticker, side, qty, price):
    return {"trade_id": tid, "conid": "1", "cik": None, "ticker": ticker,
            "ts": f"{day}T15:00:00+00:00", "side": side, "quantity": qty, "price": price,
            "currency": "USD", "commission": 0.0, "fx_rate": 1.0}


def test_each_sale_says_what_the_stock_did_afterwards_and_the_totals_match_the_mirror():
    trades = pd.DataFrame([
        trade("1", "2025-10-01", "WIN", "buy", 2.0, 100.0),
        trade("2", "2025-10-21", "WIN", "sell", 1.0, 110.0),
        trade("3", "2025-11-20", "WIN", "sell", 1.0, 120.0),
    ])
    end = "2026-09-25"
    win = pd.Series([110.0, 120.0, 150.0], index=pd.to_datetime(["2025-10-21", "2025-11-20",
                                                                 end]))
    spy = pd.Series([100.0, 100.0, 110.0], index=win.index)
    sales = bh.sales_after(trades, {"WIN": win, "SPY": spy}, end)
    assert list(sales["day"].dt.date.astype(str)) == ["2025-11-20", "2025-10-21"]
    first = sales.iloc[1]
    assert first["holding_days"] == pytest.approx(20)
    assert first["realized_return"] == pytest.approx(0.10)
    assert first["after"] == pytest.approx(150 / 110 - 1)
    assert first["after_usd"] == pytest.approx(110 * (150 / 110 - 1))
    nav = pd.DataFrame({"ts": ["2025-10-01", end], "value": [200.0, 200.0]})
    m = bh.mirror(trades, nav, nav.assign(value=0.0), pd.DataFrame(),
                  {"WIN": win, "SPY": spy}, end)
    assert m.after_sale == pytest.approx(sales["after_usd"].sum())


def test_a_sale_without_a_price_series_is_unknown_not_zero():
    trades = pd.DataFrame([trade("1", "2025-10-01", "X", "buy", 1.0, 10.0),
                           trade("2", "2025-10-05", "X", "sell", 1.0, 11.0)])
    sales = bh.sales_after(trades, {}, "2026-01-01")
    assert sales["after"].isna().all() and sales["after_usd"].isna().all()


# --- The page -----------------------------------------------------------------------------

st = pytest.importorskip("streamlit", reason="the 'app' extra is not installed")


def test_the_deploy_page_puts_the_sale_history_in_front_of_an_unbacked_sale(tmp_path,
                                                                           monkeypatch):
    """AAA was half-sold at +10 % and kept rising: selling the rest with nothing written
    must say so, next to the reinforce table."""
    from streamlit.testing.v1 import AppTest

    from app import data as app_data
    from core import config
    from db import loader

    local = tmp_path / "settings.local.yaml"
    local.write_text("universe:\n  tracked: []\n", encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", local)
    config.load_settings.cache_clear()
    db_path = tmp_path / "deploy.db"
    conn = loader.init_db(db_path)
    loader.upsert_trades(conn, pd.DataFrame([
        trade("1", "2026-01-05", "AAA", "buy", 2.0, 100.0),
        trade("2", "2026-02-05", "AAA", "sell", 1.0, 110.0),
    ]))
    days = pd.bdate_range("2026-01-01", "2026-09-25")
    obs = [{"source": "yfinance", "series_id": f"{t}:close_raw", "ts": f"{d.date()}T00:00:00Z",
            "ts_release": "", "value": base + i * step, "ingested_at": "2026-09-25"}
           for t, base, step in (("AAA", 100.0, 0.5), ("SPY", 500.0, 0.1))
           for i, d in enumerate(days)]
    stamp = "2026-09-25T00:00:00+00:00"
    obs += [{"source": "ibkr", "series_id": sid, "ts": stamp, "ts_release": "", "value": v,
             "ingested_at": stamp}
            for sid, v in (("AAA:position_qty", 1.0), ("AAA:position_cost_basis", 100.0),
                           ("NAV:total", 400.0), ("NAV:cash", 180.0))]
    loader.upsert_observations(conn, pd.DataFrame(obs))
    conn.close()
    monkeypatch.setattr(app_data, "db_path", lambda: db_path)
    st.cache_data.clear()
    main = str(pathlib.Path(__file__).resolve().parents[1] / "app" / "main.py")
    page = str(pathlib.Path(__file__).resolve().parents[1] / "app" / "pages" / "deploy.py")
    app = AppTest.from_file(main, default_timeout=120).run()
    app.switch_page(page)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    reinforce_table = next(m.value for m in app.markdown if "Métrica de tu regla" in m.value)
    assert "AAA" in reinforce_table and "Sin ficha de tesis" in reinforce_table
    warning = " ".join(w.value for w in app.warning)
    assert "Sin ficha no hay criterio de venta" in warning
    assert "**1 de 1 siguieron subiendo**" in warning
    history = next(pd.DataFrame(d.value) for d in app.dataframe
                   if "Lo que hizo después" in pd.DataFrame(d.value).columns)
    assert list(history["Empresa"]) == ["AAA"]
    st.cache_data.clear()
    config.load_settings.cache_clear()
