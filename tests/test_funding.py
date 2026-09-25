"""What it costs to bring pesos to a position (📋 Desplegar capital, transform/funding.py).

Synthetic deposits whose cost is known in advance: a fixed fee plus a spread, so the fit
has one right answer. No fee of any real intermediary is assumed anywhere.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from db import loader
from transform import funding as fx

TRM = pd.DataFrame({"series_id": "TRM:COP_USD",
                    "ts": ["2026-01-01", "2026-02-01", "2026-03-01"],
                    "value": [4000.0, 4000.0, 4000.0]})


def deposits(*pairs) -> pd.DataFrame:
    return pd.DataFrame([{"ts": f"{d}T00:00:00+00:00", "amount": usd, "kind": "deposit"}
                         for d, usd in pairs])


def paid_for(usd: float, *, fixed_cop: float, spread: float) -> float:
    """Pesos paid for ``usd`` arriving: ``usd × TRM`` grossed up by the spread, plus a fee."""
    return (usd * 4000.0 + fixed_cop) / (1 - spread)


def test_the_end_to_end_cost_is_pesos_paid_minus_dollars_at_the_trm():
    costs = fx.deposit_costs(deposits(("2026-02-02", 100.0)),
                             [{"date": "2026-02-02", "cop_paid": 420_000}], TRM)
    row = costs.iloc[0]
    assert row["trm"] == 4000.0 and row["trm_date"] == "2026-02-01", "last TRM on or before"
    assert row["cost_cop"] == pytest.approx(20_000)
    assert row["cost_pct"] == pytest.approx(20_000 / 420_000)


def test_a_deposit_without_its_pesos_has_no_cost_rather_than_a_guess():
    costs = fx.deposit_costs(deposits(("2026-02-02", 100.0)), [], TRM)
    assert costs["cost_pct"].isna().all() and costs["cop_paid"].isna().all()


def test_the_purchase_day_sets_the_trm_when_it_is_given():
    trm = pd.DataFrame({"ts": ["2026-01-30", "2026-02-02"], "value": [3900.0, 4100.0]})
    costs = fx.deposit_costs(deposits(("2026-02-02", 100.0)),
                             [{"date": "2026-02-02", "cop_paid": 400_000,
                               "bought_on": "2026-01-30"}], trm)
    assert costs.iloc[0]["trm"] == 3900.0


def test_the_fit_separates_a_fixed_fee_from_the_spread():
    """Smaller deposits pay a larger share: that difference is the fixed part."""
    sizes = [("2026-02-02", 100.0), ("2026-02-10", 300.0), ("2026-02-20", 800.0)]
    paid = [{"date": d, "cop_paid": paid_for(usd, fixed_cop=20_000, spread=0.01)}
            for d, usd in sizes]
    costs = fx.deposit_costs(deposits(*sizes), paid, TRM)
    assert costs["cost_pct"].is_monotonic_decreasing, "the fixed part weighs on small ones"
    split = fx.split_costs(costs)
    assert split.deposits == 3
    assert split.proportional == pytest.approx(0.01, abs=1e-9)
    assert split.fixed_cop == pytest.approx(20_000, rel=1e-6)


def test_no_fit_on_too_few_deposits_or_one_size():
    one_size = [("2026-02-02", 100.0), ("2026-02-10", 100.0), ("2026-02-20", 100.0)]
    paid = [{"date": d, "cop_paid": 420_000} for d, _ in one_size]
    assert fx.split_costs(fx.deposit_costs(deposits(*one_size), paid, TRM)) is None
    two = one_size[:2]
    assert fx.split_costs(fx.deposit_costs(deposits(*two), paid, TRM)) is None


def test_the_fee_per_order_is_the_accounts_own_median():
    trades = pd.DataFrame({"commission": [-0.35, -0.35, -0.36, 0.0]})
    assert fx.order_fee(trades) == pytest.approx(0.35)
    assert fx.order_fee(pd.DataFrame()) is None


def test_batching_divides_the_fixed_part_and_leaves_the_spread():
    table = fx.batching(100.0, every_months=[1, 3], fixed_usd=3.0, proportional=0.01,
                        fee_per_order=0.35)
    monthly, quarterly = table.iloc[0], table.iloc[1]
    assert monthly["fixed"] == pytest.approx(0.03) and quarterly["fixed"] == pytest.approx(0.01)
    assert monthly["proportional"] == quarterly["proportional"] == 0.01
    assert quarterly["total"] == pytest.approx(0.01 + 0.01 + 0.35 / 300)
    assert monthly["fixed_and_commission_usd_year"] == pytest.approx(3.35 * 12)
    assert quarterly["fixed_and_commission_usd_year"] == pytest.approx(3.35 * 4)


def test_an_unknown_fee_leaves_the_total_unknown():
    table = fx.batching(100.0, every_months=[1], fixed_usd=None, proportional=None,
                        fee_per_order=0.35)
    assert table.iloc[0]["total"] is None
    assert table.iloc[0]["commission"] == pytest.approx(0.0035), "what is known still shows"


# --- The page -----------------------------------------------------------------------------

st = pytest.importorskip("streamlit", reason="the 'app' extra is not installed")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import data as app_data  # noqa: E402
from app.format import PUBLIC_PAGES  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = str(REPO_ROOT / "app" / "main.py")
PAGE = str(REPO_ROOT / "app" / "pages" / "deploy.py")


@pytest.fixture()
def deploy_db(tmp_path, monkeypatch):
    db_path = tmp_path / "deploy.db"
    monkeypatch.setattr(app_data, "db_path", lambda: db_path)
    st.cache_data.clear()
    conn = loader.init_db(db_path)
    loader.upsert_cash_transactions(conn, pd.DataFrame([{
        "tx_id": "d1", "conid": None, "cik": None, "ticker": None,
        "ts": "2026-02-02T00:00:00+00:00", "kind": "deposit", "amount": 100.0,
        "currency": "USD"}]))
    loader.upsert_trades(conn, pd.DataFrame([{
        "trade_id": "t1", "conid": "1", "cik": None, "ticker": "AAA",
        "ts": "2026-02-03T15:00:00+00:00", "side": "BUY", "quantity": 1.0, "price": 90.0,
        "currency": "USD", "commission": -0.35, "fx_rate": 1.0}]))
    conn.close()
    yield db_path
    st.cache_data.clear()


def test_the_page_asks_for_the_pesos_of_each_deposit_by_its_date(deploy_db):
    app = AppTest.from_file(MAIN, default_timeout=120).run()
    app.switch_page(PAGE)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert "2026-02-02" in app.code[0].value
    assert any("0,35 USD" in m.value for m in app.markdown), "the account's own fee"


def test_the_deploy_page_is_never_public():
    """It shows the account's cash and deposits."""
    assert "Desplegar capital" not in PUBLIC_PAGES


def test_the_deploy_page_checks_the_pick_before_the_price(deploy_db, monkeypatch, tmp_path):
    """No card and no exit rule are said before any multiple: §2, level 4."""
    from core import config

    local = tmp_path / "settings.local.yaml"
    local.write_text('universe:\n  watchlist:\n    - ticker: AAA\n      cik: "0000000001"\n',
                     encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", local)
    config.load_settings.cache_clear()
    app = AppTest.from_file(MAIN, default_timeout=120).run()
    app.switch_page(PAGE)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    checks = next(m.value for m in app.markdown if "Comprobación" in m.value)
    assert "sin ficha" in checks and "sin escribir" in checks
    assert any("Sin banda de valoración para AAA" in c.value for c in app.caption)
    config.load_settings.cache_clear()
