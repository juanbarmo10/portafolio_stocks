"""Insider buying and selling as context (user request, 2026-09-29), with known answers.

What is defended: only open-market purchases and sales by directors and officers count; a
sale is plan, discretionary or unknown as the filing says; sales are drawn below zero; and a
transaction is invisible before its Form 4 was filed (section 9.4).
"""

from __future__ import annotations

import pandas as pd
import pytest

from ingest import insider_activity as ing
from transform import insider_activity as ia


def form4(*transactions, officer="1", director="0", title="Chief Financial Officer",
          plan="1"):
    blocks = "".join(
        f"<nonDerivativeTransaction><transactionDate><value>{d}</value></transactionDate>"
        f"<transactionCoding><transactionCode>{c}</transactionCode></transactionCoding>"
        f"<transactionAmounts><transactionShares><value>{n}</value></transactionShares>"
        f"<transactionPricePerShare><value>{p}</value></transactionPricePerShare>"
        f"</transactionAmounts></nonDerivativeTransaction>"
        for c, d, n, p in transactions)
    plan_tag = "" if plan is None else f"<aff10b5One>{plan}</aff10b5One>"
    return (f"<ownershipDocument>{plan_tag}<reportingOwner><reportingOwnerId><rptOwnerCik>"
            f"0000000099</rptOwnerCik></reportingOwnerId><reportingOwnerRelationship>"
            f"<isDirector>{director}</isDirector><isOfficer>{officer}</isOfficer>"
            f"<officerTitle>{title}</officerTitle></reportingOwnerRelationship>"
            f"</reportingOwner><nonDerivativeTable>{blocks}</nonDerivativeTable>"
            "</ownershipDocument>")


def test_only_purchases_and_sales_count_and_a_sale_carries_its_plan_flag():
    xml = form4(("M", "2026-09-22", 14000, 5.01), ("S", "2026-09-22", 29277, 29.95),
                ("F", "2026-09-22", 100, 30.0), ("P", "2026-09-23", 1000, 30.0))
    got = ing.parse_form4(xml)
    assert [(t["kind"], t["plan"]) for t in got] == [("sell", "plan"), ("buy", "none")]
    assert got[0]["value"] == pytest.approx(29277 * 29.95)


def test_a_fund_that_is_only_a_ten_percent_owner_is_left_out():
    assert ing.parse_form4(form4(("P", "2026-01-02", 1000, 10.0), officer="0",
                                 director="0")) == []


def test_before_the_box_existed_the_plan_is_unknown_and_a_zero_price_is_skipped():
    got = ing.parse_form4(form4(("S", "2022-05-02", 100, 10.0), ("S", "2022-05-02", 50, 0),
                                plan=None))
    assert [(t["plan"], t["shares"]) for t in got] == [("unknown", 100.0)]


def test_daily_sums_and_distinct_owners_by_filing_date():
    filings = [{"filed": "2026-09-24", "owner": "A", "transactions": ing.parse_form4(
                   form4(("S", "2026-09-22", 100, 10.0), ("S", "2026-09-22", 50, 10.0)))},
               {"filed": "2026-09-24", "owner": "B", "transactions": ing.parse_form4(
                   form4(("S", "2026-09-22", 10, 10.0)))}]
    rows = {r["series_id"]: r["value"] for r in ing.observation_rows("0000000001", filings)}
    assert rows["0000000001:insider:sell:plan"] == pytest.approx(1600.0)
    assert rows["0000000001:insider:sell:plan:owners"] == 2.0


def obs(series, day, filed, value):
    return {"source": ing.SOURCE, "series_id": f"0000000001:insider:{series}", "ts": day,
            "ts_release": filed, "value": value}


OBS = pd.DataFrame([obs("buy", "2026-03-10", "2026-03-11", 500_000.0),
                    obs("buy:owners", "2026-03-10", "2026-03-11", 2.0),
                    obs("sell:plan", "2026-06-10", "2026-06-12", 3_000_000.0),
                    obs("sell:discretionary", "2026-09-20", "2026-09-24", 800_000.0)])


def test_sales_go_below_zero_and_the_year_adds_up():
    a = ia.assess(OBS, "0000000001", "2026-09-29")
    by_kind = a.monthly.set_index("kind")["value"]
    assert by_kind["buy"] == 500_000 and by_kind["sell:plan"] == -3_000_000
    assert a.buy_12m == 500_000 and a.sell_12m == 3_800_000
    assert a.discretionary_12m == 800_000 and a.buyers_12m == 2


def test_a_sale_is_invisible_before_its_form4_is_filed():
    a = ia.assess(OBS, "0000000001", "2026-09-23")
    assert "sell:discretionary" not in set(a.monthly["kind"]), "filed on the 24th"
    assert ia.assess(OBS, "0000000009", "2026-09-29") is None
