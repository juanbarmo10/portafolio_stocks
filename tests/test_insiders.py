"""The insider-purchase study (CLAUDE.md §15.1.6), on synthetic filings with known answers."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from ingest import insiders as ingest_insiders
from validation import insiders as ins

CIK = "0000000001"


def buy(accession, day, owner, *, title="", value=50_000.0, symbol="AAA"):
    return {"accession": accession, "filing_date": day, "trans_date": day, "issuer_cik": CIK,
            "symbol": symbol, "owner_cik": owner, "is_director": True, "is_officer": False,
            "title": title, "shares": value / 10, "price": 10.0, "value": value}


def test_three_distinct_buyers_in_a_month_is_a_cluster_dated_by_the_third_filing():
    purchases = pd.DataFrame([buy("a", "2024-01-02", "o1"), buy("b", "2024-01-10", "o2"),
                              buy("c", "2024-01-25", "o3")])
    events = ins.cluster_events(purchases, min_insiders=3, window_days=30, cooldown_days=180)
    assert list(events["date"]) == [pd.Timestamp("2024-01-25")], "public on the third filing"


def test_a_joint_filing_is_one_buyer_not_three():
    """A fund and two partners reporting the same purchase in one filing (NKTX 2024-03-29)."""
    purchases = pd.DataFrame([buy("a", "2024-03-29", o) for o in ("o1", "o2", "o3")])
    assert ins.cluster_events(purchases, min_insiders=3, window_days=30,
                              cooldown_days=180).empty


def test_buyers_further_apart_than_the_window_are_not_a_cluster():
    purchases = pd.DataFrame([buy("a", "2024-01-02", "o1"), buy("b", "2024-02-10", "o2"),
                              buy("c", "2024-03-20", "o3")])
    assert ins.cluster_events(purchases, min_insiders=3, window_days=30,
                              cooldown_days=180).empty


def test_the_cooldown_keeps_one_event_per_company_per_half_year():
    rows = [buy(f"a{i}", day, f"o{i}") for i, day in enumerate(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-03-01", "2024-03-02",
         "2024-03-03", "2024-09-01", "2024-09-02", "2024-09-03"])]
    events = ins.cluster_events(pd.DataFrame(rows), min_insiders=3, window_days=30,
                                cooldown_days=180)
    # March (4 buyers) is inside the cooldown; in September the third buyer files on the 3rd.
    assert list(events["date"].dt.date.astype(str)) == ["2024-01-04", "2024-09-03"]


def test_an_executive_purchase_needs_the_title_and_the_size():
    pattern = "(chief executive|\\bceo\\b|chief financial|\\bcfo\\b)"
    purchases = pd.DataFrame([
        buy("a", "2024-01-02", "o1", title="President and CEO", value=150_000),
        buy("b", "2024-06-02", "o1", title="CEO", value=20_000),                 # too small
        buy("c", "2024-01-02", "o2", title="Director of Procurement", value=900_000,
            symbol="BBB") | {"issuer_cik": "0000000002"},                         # not an exec
    ])
    events = ins.executive_events(purchases, title_pattern=pattern, min_value_usd=100_000,
                                  cooldown_days=180)
    assert list(events["symbol"]) == ["AAA"]


def test_excess_return_counts_only_while_a_member():
    days = pd.bdate_range("2024-01-01", periods=300)
    prices = pd.DataFrame({"AAA": [100.0 * 1.001 ** i for i in range(300)]}, index=days)
    spy = pd.Series(100.0, index=days)
    member = pd.DataFrame({"AAA": [i >= 50 for i in range(300)]}, index=days)
    points = pd.DataFrame({"symbol": ["AAA", "AAA"],
                           "date": [days[10], days[100]]})
    out = ins.excess_returns(prices, spy, member, points, [30])
    assert out.loc[0, "excess_30"] is None, "not in the index yet"
    assert out.loc[1, "excess_30"] > 0


def test_the_baseline_stays_away_from_the_companys_own_events():
    days = pd.bdate_range("2020-01-01", "2024-12-31")
    member = pd.DataFrame({"AAA": True}, index=days)
    events = pd.DataFrame({"symbol": ["AAA"], "date": [pd.Timestamp("2022-06-01")]})
    base = ins.baseline_points(events, member, step_days=30, away_days=180)
    gaps = (base["date"] - pd.Timestamp("2022-06-01")).dt.days.abs()
    assert len(base) > 20 and (gaps > 180).all()


def test_the_decision_rule_needs_significance_a_positive_edge_and_both_halves():
    good = ins.Result("cluster", 90, 50, 500, 0.03, 0.0, 0.02, 0.6, 0.5, 0.001, 0.01, True,
                      (0.02, 0.04))
    unstable = ins.Result("executive", 180, 50, 500, 0.03, 0.0, 0.02, 0.6, 0.5, 0.001, 0.01,
                          True, (0.05, -0.01))
    cfg = {"decision_horizons": [90, 180]}
    assert ins.decision([good, unstable], cfg) == {"cluster": True, "executive": False}


def test_the_quarters_run_to_the_last_complete_one():
    qs = ingest_insiders.quarters("2025q3", dt.date(2026, 5, 10))
    assert qs == ["2025q3", "2025q4", "2026q1"]


def test_purchases_keep_no_personal_data():
    """The data sets carry names and street addresses; none of it survives."""
    sub = pd.DataFrame({"ACCESSION_NUMBER": ["x"], "FILING_DATE": ["02-JAN-2024"],
                        "ISSUERCIK": ["1"], "ISSUERTRADINGSYMBOL": ["aaa"],
                        "DOCUMENT_TYPE": ["4"]})
    owners = pd.DataFrame({"ACCESSION_NUMBER": ["x"], "RPTOWNERCIK": ["9"],
                           "RPTOWNERNAME": ["Jane Doe"], "RPTOWNER_STREET1": ["1 Main St"],
                           "RPTOWNER_RELATIONSHIP": ["Director"], "RPTOWNER_TITLE": [None]})
    trans = pd.DataFrame({"ACCESSION_NUMBER": ["x"], "TRANS_CODE": ["P"],
                          "TRANS_ACQUIRED_DISP_CD": ["A"], "TRANS_DATE": ["01-JAN-2024"],
                          "TRANS_SHARES": ["10"], "TRANS_PRICEPERSHARE": ["5"]})
    out = ingest_insiders.purchases(sub, owners, trans)
    assert list(out.columns) == ingest_insiders.COLUMNS
    assert "Jane Doe" not in out.to_string() and "Main" not in out.to_string()
    assert out.loc[0, "value"] == pytest.approx(50.0)
    assert out.loc[0, "symbol"] == "AAA" and out.loc[0, "filing_date"] == "2024-01-02"
