"""Core and satellite, and the idle-cash rule (CLAUDE.md §15.5, point 4), with known answers.

What is defended: nothing is assumed without a written policy; contributions, never sales,
move the account towards its target (and the amount needed is exact); the cash spell counts
only the unbroken run up to today; and a share written as a percentage is refused.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alerts import rules
from core.config import validate_policy
from transform import allocation as al

VALUED = pd.DataFrame({"ticker": ["QQQM", "HIMS", "DUOL", "NU"],
                       "market_value": [200.0, 300.0, 250.0, 150.0]})
POLICY = al.Policy.from_config({"core": {"tickers": ["qqqm"], "target_share": 0.5,
                                         "band": 0.05},
                                "max_satellites": 3, "cash_max": 0.05,
                                "cash_grace_days": 30})


def test_without_a_written_policy_nothing_is_assumed():
    split = al.allocate(VALUED, 1000.0, 100.0, al.Policy.from_config({}))
    assert split.destination is None and split.to_target is None
    assert any("Sin política escrita" in n for n in split.notes)


def test_below_target_the_tranche_goes_to_the_core_and_the_amount_needed_is_exact():
    split = al.allocate(VALUED, 1000.0, 100.0, POLICY)
    assert split.core_share == pytest.approx(0.2)
    assert split.satellite_share == pytest.approx(0.7)
    assert split.cash_share == pytest.approx(0.1)
    assert split.destination == "core"
    # Contributing `to_target`, all to the core, lands exactly on the target.
    assert (200 + split.to_target) / (1000 + split.to_target) == pytest.approx(0.5)


def test_within_the_band_the_tranche_may_go_to_the_satellite():
    valued = VALUED.assign(market_value=[460.0, 300.0, 150.0, 0.0])
    split = al.allocate(valued, 1000.0, 90.0, POLICY)
    assert split.core_share == pytest.approx(0.46) and split.destination == "satellite"
    assert split.satellites == ["DUOL", "HIMS"], "a position worth nothing is not held"


def test_the_satellite_limit_is_said():
    split = al.allocate(VALUED, 1000.0, 100.0, POLICY)
    assert any("máximo es 3" in n for n in split.notes)


def daily(values, start="2026-08-01"):
    days = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({"ts": [d.date().isoformat() for d in days], "value": values})


def test_the_cash_spell_is_the_unbroken_run_up_to_today():
    nav = daily([1000.0] * 50)
    cash = daily([100.0] * 10 + [10.0] * 5 + [300.0] * 35)   # above, below, above again
    days, since = al.cash_spell(nav, cash, 0.05)
    assert since == "2026-08-16" and days == 34


def test_no_spell_when_cash_is_within_the_limit_or_there_is_no_limit():
    nav = daily([1000.0] * 5)
    assert al.cash_spell(nav, daily([10.0] * 5), 0.05) == (0, None)
    assert al.cash_spell(nav, daily([900.0] * 5), None) == (None, None)


def snapshot(policy, cash_values):
    return rules.Snapshot(now=pd.Timestamp("2026-09-28", tz="UTC"),
                          nav=daily([1000.0] * len(cash_values)), nav_cash=daily(cash_values),
                          policy=policy)


def test_the_alert_fires_after_the_grace_once_per_spell_and_is_silent_without_a_policy():
    policy = {"cash_max": 0.05, "cash_grace_days": 30}
    fired = rules.cash_above_limit(snapshot(policy, [300.0] * 40), {})
    assert len(fired) == 1 and fired[0].key == "cash_above_limit:2026-08-01"
    assert rules.cash_above_limit(snapshot(policy, [300.0] * 20), {}) == [], "within grace"
    assert rules.cash_above_limit(snapshot({}, [300.0] * 40), {}) == []


@pytest.mark.parametrize("portfolio, fragment", [
    ({"core": {"tickers": ["QQQM"], "target_share": 50}}, "fraction"),
    ({"cash_max": 5}, "fraction"),
    ({"max_satellites": 5.5}, "whole number"),
    ({"core": {"tickers": "QQQM"}}, "list of tickers"),
    ({"core": {"target_share": 0.5}}, "without portfolio.core.tickers"),
])
def test_a_policy_that_would_invert_the_advice_is_refused(portfolio, fragment):
    with pytest.raises(ValueError, match=fragment):
        validate_policy(portfolio)


def test_an_empty_or_valid_policy_passes():
    validate_policy(None)
    validate_policy({"core": {"tickers": ["QQQM"], "target_share": 0.5, "band": 0.05},
                     "max_satellites": 6, "cash_max": 0.05, "cash_grace_days": 45})


# --- The page ---------------------------------------------------------------------------------

st = pytest.importorskip("streamlit", reason="the 'app' extra is not installed")


def test_the_deploy_page_sends_the_tranche_to_the_core_when_it_is_below_target(tmp_path,
                                                                               monkeypatch):
    import pathlib

    from streamlit.testing.v1 import AppTest

    from app import data as app_data
    from core import config
    from db import loader

    local = tmp_path / "settings.local.yaml"
    local.write_text("portfolio:\n  core: {tickers: [QQQM], target_share: 0.5}\n",
                     encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", local)
    config.load_settings.cache_clear()
    db_path = tmp_path / "deploy.db"
    conn = loader.init_db(db_path)
    stamp = "2026-09-25T00:00:00+00:00"
    loader.upsert_observations(conn, pd.DataFrame([
        {"source": "ibkr", "series_id": s, "ts": stamp, "ts_release": "", "value": v,
         "ingested_at": stamp}
        for s, v in (("QQQM:position_qty", 1.0), ("QQQM:position_cost_basis", 200.0),
                     ("AAA:position_qty", 4.0), ("AAA:position_cost_basis", 700.0),
                     ("NAV:total", 1000.0), ("NAV:cash", 0.0))]
        + [{"source": "yfinance", "series_id": f"{t}:close_raw", "ts": stamp,
            "ts_release": "", "value": p, "ingested_at": stamp}
           for t, p in (("QQQM", 200.0), ("AAA", 200.0))]))
    conn.close()
    monkeypatch.setattr(app_data, "db_path", lambda: db_path)
    st.cache_data.clear()
    root = pathlib.Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app" / "main.py"), default_timeout=120).run()
    app.switch_page(str(root / "app" / "pages" / "deploy.py"))
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    text = " ".join(m.value for m in app.markdown) + " ".join(i.value for i in app.info)
    assert "20 %" in text and "objetivo del **50 %**" in text
    assert "Este tramo va al núcleo" in text and "600,00 USD" in text, \
        "(0,5 × 1000 − 200) / (1 − 0,5) = 600"
    st.cache_data.clear()
    config.load_settings.cache_clear()
