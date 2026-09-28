"""Peers chosen by the user (CLAUDE.md §15.5, point 11), with known answers.

What is defended: a peer outside the growth screen (DiDi: 20-F, yuan, annual) is read in
its own currency, where every ratio cancels it; a fiscal year is labelled as such; the
median is the peers' without the company; a peer with no figures stays visible as missing;
and the peers' configuration fails loudly on a shape that would be misread.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.config import load_settings, validate_peers
from ingest import peers as ip
from transform import growth_screen as gs
from transform import peers as pr

CIK = "0001764757"


def fact(val, end, filed):
    """One fiscal-year fact of a 20-F."""
    start = (pd.Timestamp(end) - pd.Timedelta(days=364)).date().isoformat()
    return {"val": val, "start": start, "end": end, "filed": filed, "form": "20-F", "fp": "FY"}


FACTS = {"us-gaap": {
    "Revenues": {"units": {"CNY": [fact(192e9, "2023-12-31", "2024-04-15"),
                                   fact(206e9, "2024-12-31", "2025-04-18"),
                                   fact(227e9, "2025-12-31", "2026-04-13")],
                           "USD": [fact(31e9, "2025-12-31", "2026-04-13")]}},
    "GrossProfit": {"units": {"CNY": [fact(45.4e9, "2025-12-31", "2026-04-13")]}},
    "OperatingIncomeLoss": {"units": {"CNY": [fact(2.0e9, "2024-12-31", "2025-04-18"),
                                              fact(6.54e9, "2025-12-31", "2026-04-13")]}},
    "NetCashProvidedByUsedInOperatingActivities": {"units": {"CNY": [
        fact(11.35e9, "2025-12-31", "2026-04-13")]}},
    "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"CNY": [
        fact(2.27e9, "2025-12-31", "2026-04-13")]}},
}}


@pytest.fixture(scope="module")
def cfg():
    return load_settings().source("sec")


def test_the_reporting_currency_is_the_one_most_revenue_is_filed_in():
    assert ip.reporting_currency(FACTS, ["Revenues"]) == "CNY"


def test_annual_rows_keep_fiscal_years_in_the_reporting_currency(cfg):
    rows = ip.annual_rows(FACTS, CIK, cfg["concepts"], cfg["duration_windows"])
    revenue = sorted((r["ts"], r["value"]) for r in rows
                     if r["series_id"] == f"{CIK}:revenue:fy")
    assert revenue[-1] == ("2025-12-31", 227e9), "yuan, not the one USD convenience fact"
    assert all(r["source"] == ip.SOURCE for r in rows)
    assert {r["ts_release"] for r in rows if r["ts"] == "2025-12-31"} == {"2026-04-13"}


def test_the_annual_table_reads_the_latest_year_against_the_one_before(cfg):
    obs = pd.DataFrame(ip.annual_rows(FACTS, CIK, cfg["concepts"], cfg["duration_windows"]))
    table = pr.annual_table(obs, {CIK: ("DIDIY", "DiDi")}, "2026-09-28").set_index("ticker")
    d = table.loc["DIDIY"]
    assert d["quarter"] == "FY2025"
    assert d["revenue_growth"] == pytest.approx(227 / 206 - 1)
    assert d["gross_margin"] == pytest.approx(45.4 / 227)
    assert d["incremental_margin"] == pytest.approx((6.54 - 2.0) / (227 - 206))
    assert d["fcf_margin"] == pytest.approx((11.35 - 2.27) / 227)
    assert pd.isna(d["market_cap"]), "no price for an annual-only peer: unknown"


def test_point_in_time_hides_a_20f_not_yet_filed(cfg):
    obs = pd.DataFrame(ip.annual_rows(FACTS, CIK, cfg["concepts"], cfg["duration_windows"]))
    d = pr.annual_table(obs, {CIK: ("DIDIY", "DiDi")}, "2026-04-12").iloc[0]
    assert d["quarter"] == "FY2024", "the FY2025 20-F lands on 2026-04-13"


def growth_row(ticker, **values):
    return {**dict.fromkeys(gs.COLUMNS), "ticker": ticker, "quarter": "CY2026Q2", **values}


def test_the_peer_table_keeps_the_order_the_median_of_peers_and_says_what_is_missing():
    growth = pd.DataFrame([growth_row("UBER", revenue_growth=0.12),
                           growth_row("LYFT", revenue_growth=0.16),
                           growth_row("DASH", revenue_growth=0.36)])
    annual = pd.DataFrame([growth_row("DIDIY", quarter="FY2025", revenue_growth=0.10)])
    rows, median = pr.peer_table(growth, annual, "UBER", ["LYFT", "DASH", "DIDIY", "WAYMO"])
    assert list(rows["ticker"]) == ["UBER", "LYFT", "DASH", "DIDIY", "WAYMO"]
    assert list(rows["found"]) == [True, True, True, True, False]
    assert median["revenue_growth"] == pytest.approx(0.16), "LYFT, DASH, DIDIY — not UBER"
    assert rows.set_index("ticker").loc["DIDIY", "quarter"] == "FY2025"


def test_the_quarter_wins_over_the_year_for_the_same_company():
    growth = pd.DataFrame([growth_row("AAA", revenue_growth=0.3)])
    annual = pd.DataFrame([growth_row("AAA", quarter="FY2025", revenue_growth=0.1)])
    rows, _ = pr.peer_table(growth, annual, "AAA", [])
    assert rows.iloc[0]["quarter"] == "CY2026Q2"


@pytest.mark.parametrize("raw", [
    {"UBER": "LYFT"},                  # a string would be read one letter per peer
    {"UBER": ["UBER", "LYFT"]},        # itself
    ["UBER"],
])
def test_a_peers_block_with_the_wrong_shape_is_refused(raw):
    with pytest.raises(ValueError):
        validate_peers(raw)


def test_a_valid_peers_block_passes():
    validate_peers({"UBER": ["LYFT", "DASH"]})
    validate_peers(None)


def test_a_banks_free_cash_flow_is_not_shown():
    """SoFi's operating cash flow carries its loans: −643 % of revenue as "FCF margin"."""
    growth = pd.DataFrame([growth_row("NU"), growth_row("SOFI", fcf_margin=-6.4,
                                                        rule_of_40=-6.4, financial=True),
                           growth_row("MELI", fcf_margin=0.37, financial=False)])
    rows, median = pr.peer_table(growth, pd.DataFrame(), "NU", ["SOFI", "MELI"])
    sofi = rows.set_index("ticker").loc["SOFI"]
    assert pd.isna(sofi["fcf_margin"]) and pd.isna(sofi["rule_of_40"])
    assert median["fcf_margin"] == pytest.approx(0.37)
