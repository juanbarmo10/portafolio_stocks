"""Local-currency tax layer (section 11): conversions, the asset/FX split, and no invented
parameter anywhere. Synthetic rates on purpose — the real source and currency live only in
the owner's local config."""

from __future__ import annotations

import pandas as pd
import pytest

from core.config import load_settings
from fiscal import estimates as est
from fiscal import lots as fl
from ingest.local_fx import LocalFxIngester, parse_rates

# A rate valid over a weekend arrives once, starting on the Friday.
FX = pd.Series([4000.0, 4100.0, 4200.0],
               index=pd.to_datetime(["2025-01-03", "2025-01-06", "2025-06-02"]))


def test_the_rate_of_a_date_is_the_latest_valid_on_it():
    assert fl.fx_asof(FX, "2025-01-05T15:00:00+00:00") == 4000.0, "Sunday uses Friday's"
    assert fl.fx_asof(FX, "2025-01-06") == 4100.0
    assert fl.fx_asof(FX, "2025-01-02") is None, "before the history: no borrowed rate"


def test_a_disposal_splits_into_asset_and_currency_exactly():
    disposals = pd.DataFrame([{
        "ticker": "X", "ts": "2025-06-02", "acquired_ts": "2025-01-06", "holding_days": 147,
        "quantity": 1.0, "proceeds": 90.0, "cost": 100.0, "realized_pnl": -11.0,
        "buy_commission": -0.5, "sell_commission": -0.5,
    }])
    row = fl.local_disposals(disposals, FX).iloc[0]
    assert row["cost_usd"] == 100.5 and row["proceeds_usd"] == 89.5
    assert row["result_usd"] == pytest.approx(-11.0), "same result as IBKR's"
    assert row["cost_local"] == pytest.approx(100.5 * 4100)
    assert row["proceeds_local"] == pytest.approx(89.5 * 4200)
    assert row["asset_part_local"] + row["fx_part_local"] == pytest.approx(row["result_local"])


def test_a_dollar_loss_can_be_a_local_gain():
    """The reason the layer exists (section 9.9): the currency moved more than the asset."""
    disposals = pd.DataFrame([{
        "ticker": "X", "ts": "2025-06-02", "acquired_ts": "2025-01-03", "holding_days": 150,
        "quantity": 1.0, "proceeds": 98.0, "cost": 100.0, "realized_pnl": -2.0,
        "buy_commission": 0.0, "sell_commission": 0.0,
    }])
    row = fl.local_disposals(disposals, FX).iloc[0]
    assert row["result_usd"] < 0 < row["result_local"]


def test_an_open_lot_keeps_its_frozen_cost_and_the_split_reconciles():
    open_lots = pd.DataFrame([{"ticker": "X", "ts": "2025-01-06", "quantity": 2.0,
                               "price": 50.0, "commission": -1.0, "cost": 101.0}])
    lots = fl.local_lots(open_lots, FX)
    assert lots.iloc[0]["cost_local"] == pytest.approx(101.0 * 4100)
    valued = fl.valued_lots(lots, {"X": 60.0}, fx_now=4200.0)
    a = est.attribution(valued)
    assert a.result_usd == pytest.approx(19.0)
    assert a.asset_part_local + a.fx_part_local == pytest.approx(a.result_local)
    assert a.fx_cost_average == pytest.approx(4100.0)


def test_a_lot_without_a_rate_leaves_the_total_unknown_not_partial():
    open_lots = pd.DataFrame([
        {"ticker": "X", "ts": "2025-01-06", "quantity": 1.0, "price": 1, "commission": 0, "cost": 1.0},
        {"ticker": "Y", "ts": "2024-12-01", "quantity": 1.0, "price": 1, "commission": 0, "cost": 1.0},
    ])
    valued = fl.valued_lots(fl.local_lots(open_lots, FX), {"X": 1.0, "Y": 1.0}, 4200.0)
    assert est.attribution(valued) is None


def test_no_holding_period_is_assumed():
    """The threshold is the owner's to write, after asking an accountant (section 11)."""
    lots = pd.DataFrame({"ts": ["2025-01-06"]})
    out = est.holding_threshold(lots, "2025-06-01", None)
    assert out["threshold_date"].iloc[0] is None and out["days_to_threshold"].iloc[0] is None
    with_param = est.holding_threshold(lots, "2025-06-01", 12)
    assert with_param["threshold_date"].iloc[0] == "2026-01-06"
    assert est.classify_disposals(pd.DataFrame({"ts": ["x"], "acquired_ts": ["y"]}), None)[
        "held_past_threshold"].iloc[0] is None


def test_the_shipped_config_carries_no_tax_parameter_or_source():
    """Tax parameters and the rate's source reveal the jurisdiction: local config only."""
    fiscal = load_settings().raw.get("fiscal") or {}
    assert not fiscal.get("local_fx")
    assert fiscal.get("holding_period_months") is None


def test_without_the_local_config_the_rate_is_not_fetched():
    assert LocalFxIngester.is_available(load_settings()) is False


def test_dividends_use_the_observed_withholding_at_the_payment_rate():
    cash = pd.DataFrame([
        {"ts": "2025-01-06T12:00:00+00:00", "ticker": "X", "kind": "dividend", "amount": 10.0},
        {"ts": "2025-01-06T12:00:00+00:00", "ticker": "X", "kind": "withholding_tax", "amount": -3.0},
        {"ts": "2025-01-06T12:00:00+00:00", "ticker": None, "kind": "deposit", "amount": 500.0},
    ])
    row = est.dividends_local(cash, FX).iloc[0]
    assert (row["gross_usd"], row["withholding_usd"], row["net_usd"]) == (10.0, -3.0, 7.0)
    assert row["net_local"] == pytest.approx(7.0 * 4100)


def test_us_situs_counts_only_known_us_issuers_and_says_what_it_could_not_place():
    valued = pd.DataFrame([{"ticker": "A", "market_value": 100.0},
                           {"ticker": "B", "market_value": 50.0},
                           {"ticker": "C", "market_value": 30.0}])
    securities = pd.DataFrame([{"ticker": "A", "issuer_country": "US"},
                               {"ticker": "B", "issuer_country": "IE"}])
    out = est.us_situs_value(valued, securities)
    assert out == {"us_usd": 100.0, "unknown_usd": 30.0, "unknown": ["C"]}


def test_a_changed_endpoint_fails_loudly():
    with pytest.raises(ValueError):
        parse_rates({"error": "x"}, date_field="d", value_field="v", source="s", series_id="i")
    with pytest.raises(KeyError):
        parse_rates([{"other": 1}], date_field="d", value_field="v", source="s", series_id="i")
    frame = parse_rates([{"d": "2025-01-03T00:00:00.000", "v": "4000.5"}], date_field="d",
                        value_field="v", source="s", series_id="i")
    assert frame.iloc[0]["ts"] == "2025-01-03T00:00:00+00:00" and frame.iloc[0]["value"] == 4000.5
