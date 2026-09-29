"""A sector group's financial health, US GAAP or IFRS (CLAUDE.md §15.5, point 14).

What is defended, each a trap measured on the silver miners' real filings (2026-09-29):
a figure missing in the latest fiscal year is unknown, never another year's; a current debt
of 0 does not mean no debt; a tagged total beats long-term + current; a weighted share count
in another scale is flagged instead of read as dilution; an acquisition is not dilution; and
a company is read in the taxonomy of its most recent annual report.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ingest import sector_groups as sg
from transform import sector_health as sh

CIK = "0000000001"
FY, PREV = "2025-12-31", "2024-12-31"


def obs(metric, value, ts=FY, filed="2026-03-01"):
    return {"source": sg.SOURCE, "series_id": f"{CIK}:{metric}", "ts": ts, "ts_release": filed,
            "value": value}


def base(**over):
    rows = {
        "revenue:fy": [(FY, 1000.0), (PREV, 800.0)],
        "operating_income:fy": [(FY, 300.0)], "depreciation:fy": [(FY, 100.0)],
        "operating_cash_flow:fy": [(FY, 350.0)], "capex:fy": [(FY, 150.0)],
        "cash": [(FY, 500.0)], "long_term_debt": [(FY, 700.0)], "debt_current": [(FY, 50.0)],
        "equity": [(FY, 2000.0)], "basic_shares:fy": [(FY, 105.0), (PREV, 100.0)],
        "shares_outstanding": [("2026-02-15", 106.0)], "ifrs": [(FY, 1.0)],
    }
    rows.update(over)
    return pd.DataFrame([obs(k, v, ts) for k, pairs in rows.items() if pairs
                         for ts, v in pairs])


def table(frame, price=20.0):
    return sh.health_table(frame, ["AAA"], {CIK: ("AAA", "Alpha")}, {"AAA": price},
                           "2026-09-29").iloc[0]


def test_the_healthy_case_reads_every_figure():
    row = table(base())
    assert row["basis"] == "IFRS" and row["fy_end"] == FY
    assert row["revenue_growth"] == pytest.approx(0.25)
    assert row["fcf"] == pytest.approx(200.0) and row["fcf_margin"] == pytest.approx(0.2)
    assert row["debt"] == pytest.approx(750.0) and row["net_cash"] == pytest.approx(-250.0)
    assert row["net_debt_to_ebitda"] == pytest.approx(250.0 / 400.0)
    assert row["dilution"] == pytest.approx(0.05)
    assert row["market_cap"] == pytest.approx(20.0 * 106.0)
    assert row["ev_to_ebitda"] == pytest.approx((2120.0 + 250.0) / 400.0)


def test_a_zero_current_debt_is_not_no_debt():
    row = table(base(long_term_debt=[]))
    assert row["debt"] is None and row["net_cash"] is None and row["ev_to_ebitda"] is None
    assert "deuda a largo sin etiquetar" in row["notes"]


def test_a_tagged_total_wins_over_the_parts():
    row = table(base(debt_total=[(FY, 900.0)]))
    assert row["debt"] == pytest.approx(900.0)


def test_a_figure_only_in_an_older_year_is_unknown_not_carried_forward():
    row = table(base(capex=[], **{"capex:fy": [(PREV, 120.0)]}, long_term_debt=[(PREV, 10.0)]))
    assert row["fcf"] is None, "the 2024 capex must not stand in for 2025"
    assert row["debt"] is None, "the 2024 debt must not stand in for 2025"


def test_a_share_count_in_another_scale_is_flagged_not_dilution():
    row = table(base(**{"basic_shares:fy": [(FY, 105_000.0), (PREV, 100.0)]}))
    assert row["dilution"] is None and "otra escala" in row["notes"]


def test_an_acquisition_is_not_dilution():
    row = table(base(**{"basic_shares:fy": [(FY, 160.0), (PREV, 100.0)]},
                     shares_outstanding=[("2026-02-15", 161.0)]))
    assert row["dilution"] is None and "fusión o compra" in row["notes"]


def test_net_cash_has_no_net_debt_ratio_and_an_absent_company_keeps_its_row():
    row = table(base(cash=[(FY, 900.0)]))
    assert row["net_cash"] == pytest.approx(150.0) and row["net_debt_to_ebitda"] is None
    empty = sh.health_table(base(), ["AAA", "ZZZ"], {CIK: ("AAA", "Alpha")}, {},
                            "2026-09-29")
    assert empty.iloc[1]["notes"] == "sin cifras anuales en la SEC"


def fact(val, end, start=None, form="40-F", filed="2026-03-01"):
    out = {"val": val, "end": end, "filed": filed, "form": form, "fp": "FY", "fy": 2025}
    if start:
        out["start"] = start
    return out


def test_the_newest_annual_report_decides_the_taxonomy_and_rows_are_annual_only():
    facts = {
        "us-gaap": {"Revenues": {"units": {"USD": [fact(10.0, "2019-12-31", "2019-01-01",
                                                        form="10-K")]}}},
        "ifrs-full": {
            "Revenue": {"units": {"USD": [
                fact(900.0, FY, "2025-01-01"),
                {**fact(250.0, "2025-06-30", "2025-04-01"), "fp": "Q2"}]}},
            "CashAndCashEquivalents": {"units": {"USD": [fact(55.0, FY)]}},
        },
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"val": 300.0, "end": "2026-02-20", "filed": "2026-03-01"}]}}},
    }
    concepts = {"revenue": {"unit": "USD", "tags": ["Revenues"]}}
    ifrs = {"revenue": {"unit": "USD", "tags": ["Revenue"]},
            "cash": {"unit": "USD", "tags": ["CashAndCashEquivalents"]}}
    windows = {"quarter": [80, 100], "year": [350, 380]}
    assert sg.choose_taxonomy(facts, concepts, ifrs) == sg.IFRS
    rows = sg.company_rows(facts, CIK, concepts, ifrs, windows)
    ids = {r["series_id"] for r in rows}
    assert f"{CIK}:revenue:fy" in ids and f"{CIK}:revenue:q" not in ids, "no quarters"
    assert f"{CIK}:cash" in ids and f"{CIK}:shares_outstanding" in ids
    flag = [r for r in rows if r["series_id"] == f"{CIK}:ifrs"]
    assert flag and flag[0]["value"] == 1.0 and flag[0]["ts_release"] == "2026-03-01"


def test_a_group_written_as_a_string_is_rejected(tmp_path, monkeypatch):
    from core import config
    from ingest.prices import PricesIngester

    local = tmp_path / "settings.local.yaml"
    local.write_text('universe:\n  sector_groups:\n    "Plata": HL\n', encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", local)
    config.load_settings.cache_clear()
    with pytest.raises(ValueError, match="sector_groups"):
        config.load_settings()
    local.write_text('universe:\n  sector_groups:\n    "Plata": [hl, CDE]\n', encoding="utf-8")
    config.load_settings.cache_clear()
    settings = config.load_settings()
    assert settings.sector_groups == {"Plata": ["HL", "CDE"]}
    assert {"HL", "CDE"} <= set(PricesIngester.tickers_for(settings)), "their prices are fetched"
    config.load_settings.cache_clear()
