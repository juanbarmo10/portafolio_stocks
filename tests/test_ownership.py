"""Who owns and who trades the stock (user request 2026-09-29), with known answers.

What is defended: institutional shares are each manager's latest version filed by the date
(point-in-time, a restatement counts from its filing); a quarter still being filed is marked
incomplete rather than read as selling; movers compare the last two complete quarters; and
retail participation is a share of the week's own volume.
"""

from __future__ import annotations

import pandas as pd
import pytest

from transform import ownership as own

CIK = "0001773751"


def obs(filer, quarter, filed, value):
    return {"source": "sec_13f", "series_id": f"{CIK}:inst:{filer}", "ts": quarter,
            "ts_release": filed, "value": value}


OBS = pd.DataFrame([
    obs("A", "2026-03-31", "2026-05-10", 100.0), obs("B", "2026-03-31", "2026-05-12", 50.0),
    obs("A", "2026-06-30", "2026-08-10", 160.0), obs("B", "2026-06-30", "2026-08-12", 0.0),
    obs("C", "2026-06-30", "2026-08-13", 30.0),
    obs("A", "2026-06-30", "2026-09-20", 120.0),          # A restates Q2 later
    obs("A", "2026-09-30", "2026-10-15", 10.0),           # Q3 still being filed
])


def test_totals_holders_share_and_change_point_in_time():
    outstanding = pd.Series([1000.0], index=[pd.Timestamp("2026-01-01")])
    view = own.institutional(OBS, CIK, "2026-09-10", outstanding, {"A": "Fondo A"})
    q2 = view.quarters.set_index("quarter").loc["2026-06-30"]
    assert q2["shares"] == 190.0, "A 160 (restatement not filed yet) + C 30; B exited"
    assert q2["holders"] == 2 and q2["share_of_company"] == pytest.approx(0.19)
    assert q2["change"] == pytest.approx(190.0 - 150.0)
    later = own.institutional(OBS, CIK, "2026-09-25", outstanding)
    assert later.quarters.set_index("quarter").loc["2026-06-30", "shares"] == 150.0


def test_a_quarter_still_being_filed_is_marked_and_movers_use_complete_ones():
    view = own.institutional(OBS, CIK, "2026-10-20", None, {"A": "Fondo A"})
    last = view.quarters.iloc[-1]
    assert last["quarter"] == "2026-09-30" and not last["complete"]
    assert view.latest == "2026-06-30" and any("todavía" in n for n in view.notes)
    movers = view.movers.set_index("filer_cik")
    assert movers.loc["B", "change"] == -50.0 and movers.loc["C", "change"] == 30.0
    assert movers.loc["A", "name"] == "Fondo A"


def test_retail_participation_is_a_share_of_that_weeks_volume():
    finra = pd.DataFrame([
        {"source": "finra_otc", "series_id": "HIMS:otc_nonats:w", "ts": "2026-08-24",
         "ts_release": "2026-09-08", "value": 400.0},
        {"source": "finra_otc", "series_id": "HIMS:ats:w", "ts": "2026-08-24",
         "ts_release": "2026-09-08", "value": 100.0}])
    days = pd.bdate_range("2026-08-24", periods=10)
    volume = pd.Series(100.0, index=days)                  # 500 in the week of the 24th
    out = own.retail_participation(finra, volume, "HIMS", "2026-09-29").iloc[0]
    assert out["otc_share"] == pytest.approx(0.8) and out["ats_share"] == pytest.approx(0.2)
    assert own.retail_participation(finra, volume, "HIMS", "2026-09-01").empty, \
        "published on the 8th"


def test_an_exit_and_an_entry_of_the_same_manager_are_flagged_not_merged():
    movers = pd.DataFrame({"filer_cik": ["1", "2", "3"],
                           "name": ["Pershing Square Capital Management, L.P.",
                                    "PERSHING SQUARE INC.", "Norges Bank"],
                           "before": [29.96, 0.0, 0.0], "after": [0.0, 34.33, 26.7],
                           "change": [-29.96, 34.33, 26.7]})
    assert list(own.entity_changes(movers)) == [True, True, False]
