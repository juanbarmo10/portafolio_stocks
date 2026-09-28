"""Business segments (CLAUDE.md §15.5, point 15), on a synthetic XBRL instance and rows
with known answers.

What is defended: only a segment's own figure is read (not an elimination nor a country
inside it, which would double count); the fourth quarter is derived from the year, dated by
the 10-K; a change of profit measure is named and never spliced; a restated comparative is
counted; and a filing is invisible before it is filed (section 9.4).
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.config import load_settings
from ingest import segments as ing
from transform import segments as sg

XML = """<?xml version="1.0"?>
<xbrl xmlns="http://www.xbrl.org/2003/instance">
<context id="mob_q"><entity><segment>
  <xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">x:MobilityMember</xbrldi:explicitMember>
  <xbrldi:explicitMember dimension="srt:ConsolidationItemsAxis">us-gaap:OperatingSegmentsMember</xbrldi:explicitMember>
</segment></entity><period><startDate>2026-04-01</startDate><endDate>2026-06-30</endDate></period></context>
<context id="del_q"><entity><segment>
  <xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">x:DeliverySegmentMember</xbrldi:explicitMember>
</segment></entity><period><startDate>2026-04-01</startDate><endDate>2026-06-30</endDate></period></context>
<context id="elim"><entity><segment>
  <xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">x:MobilityMember</xbrldi:explicitMember>
  <xbrldi:explicitMember dimension="srt:ConsolidationItemsAxis">us-gaap:IntersegmentEliminationMember</xbrldi:explicitMember>
</segment></entity><period><startDate>2026-04-01</startDate><endDate>2026-06-30</endDate></period></context>
<context id="country"><entity><segment>
  <xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">x:MobilityMember</xbrldi:explicitMember>
  <xbrldi:explicitMember dimension="srt:StatementGeographicalAxis">country:US</xbrldi:explicitMember>
</segment></entity><period><startDate>2026-04-01</startDate><endDate>2026-06-30</endDate></period></context>
<context id="total"><entity></entity><period><startDate>2026-04-01</startDate><endDate>2026-06-30</endDate></period></context>
<us-gaap:Revenues contextRef="elim" unitRef="usd" decimals="-6">-10000000</us-gaap:Revenues>
<us-gaap:Revenues contextRef="country" unitRef="usd" decimals="-6">3000000000</us-gaap:Revenues>
<us-gaap:Revenues contextRef="mob_q" unitRef="usd" decimals="-6">7363000000</us-gaap:Revenues>
<us-gaap:Revenues contextRef="del_q" unitRef="usd" decimals="-6">5245000000</us-gaap:Revenues>
<us-gaap:Revenues contextRef="total" unitRef="usd" decimals="-6">14191000000</us-gaap:Revenues>
<us-gaap:OperatingIncomeLoss contextRef="mob_q" unitRef="usd" decimals="-6">2215000000</us-gaap:OperatingIncomeLoss>
<us-gaap:Goodwill contextRef="mob_q" unitRef="usd" decimals="-6">1</us-gaap:Goodwill>
</xbrl>"""


def test_only_a_segments_own_figure_is_read():
    facts = ing.segment_facts(XML, {"Revenues", "OperatingIncomeLoss"})
    got = {(ing.member_label(f["member"]), f["concept"]): f["value"] for f in facts}
    assert got == {("Mobility", "Revenues"): 7363e6, ("Delivery", "Revenues"): 5245e6,
                   ("Mobility", "OperatingIncomeLoss"): 2215e6}, (
        "no elimination, no country inside the segment, no company total, no Goodwill — "
        "and they come first in the document, so first-seen cannot hide a wrong filter")


def test_rows_are_classified_by_duration_and_dated_by_the_filing():
    cfg = load_settings().source("sec")
    rows = ing.observation_rows("0000000001", ing.segment_facts(XML, {"Revenues"}),
                                "2026-08-05", cfg["duration_windows"])
    assert {r["series_id"] for r in rows} == {"0000000001:Mobility:Revenues:q",
                                              "0000000001:Delivery:Revenues:q"}
    assert {r["ts_release"] for r in rows} == {"2026-08-05"}


# --- Transform -------------------------------------------------------------------------------

CIK = "0000000001"


def row(member, concept, label, end, value, filed):
    return {"source": ing.SOURCE, "series_id": f"{CIK}:{member}:{concept}:{label}",
            "ts": end, "ts_release": filed, "value": value}


OBS = pd.DataFrame([
    # 2025 by quarter (10-Qs) and as a year (10-K, 2026-02-13), with the old measure.
    row("Mobility", "Revenues", "q", "2025-06-30", 7288.0, "2025-08-06"),
    row("Mobility", "Revenues", "ytd3", "2025-09-30", 21466.0, "2025-11-04"),
    row("Mobility", "Revenues", "fy", "2025-12-31", 29670.0, "2026-02-13"),
    row("Mobility", "Revenues", "q", "2026-06-30", 7363.0, "2026-08-05"),
    row("Delivery", "Revenues", "q", "2025-06-30", 4102.0, "2025-08-06"),
    row("Delivery", "Revenues", "q", "2026-06-30", 5245.0, "2026-08-05"),
    row("Mobility", "OldEbitda", "q", "2025-06-30", 2000.0, "2025-08-06"),
    # The new measure, with the 2025 comparative restated to it by the 2026 filing.
    row("Mobility", "OperatingIncomeLoss", "q", "2026-06-30", 2215.0, "2026-08-05"),
    row("Mobility", "OperatingIncomeLoss", "q", "2025-06-30", 1729.0, "2026-08-05"),
    # A revenue comparative the 2026 filing changed.
    row("Delivery", "Revenues", "q", "2025-06-30", 4150.0, "2026-08-05"),
])
REVENUE = ["Revenues"]
PROFIT = ["OperatingIncomeLoss", "OldEbitda"]


@pytest.fixture(scope="module")
def view():
    return sg.assess(OBS, CIK, "2026-09-28", REVENUE, PROFIT)


def test_growth_share_margin_and_the_measure_of_the_latest_quarter(view):
    table = view.table.set_index("member")
    assert view.quarter == "2026-06-30" and view.members == ["Mobility", "Delivery"]
    assert table.loc["Mobility", "revenue_growth"] == pytest.approx(7363 / 7288 - 1)
    assert table.loc["Delivery", "revenue_growth"] == pytest.approx(5245 / 4150 - 1), \
        "the restated comparative, the latest version"
    assert table.loc["Mobility", "share"] == pytest.approx(7363 / (7363 + 5245))
    assert table.loc["Mobility", "margin"] == pytest.approx(2215 / 7363)
    assert table.loc["Mobility", "profit_growth"] == pytest.approx(2215 / 1729 - 1), \
        "within the new measure only"


def test_a_change_of_measure_is_named_and_a_restatement_counted(view):
    assert view.profit_concept == "OperatingIncomeLoss"
    assert view.older_concepts == ["OldEbitda"]
    assert view.restated == 1


def test_the_fourth_quarter_is_derived_from_the_year_and_dated_by_the_10k():
    q = sg.quarterly(OBS.assign(
        member=OBS["series_id"].str.split(":").str[1],
        concept=OBS["series_id"].str.split(":").str[2],
        label=OBS["series_id"].str.split(":").str[3],
        ts=pd.to_datetime(OBS["ts"])))
    q4 = q[(q["member"] == "Mobility") & (q["concept"] == "Revenues")
           & (q["ts"] == pd.Timestamp("2025-12-31"))].iloc[0]
    assert q4["value"] == pytest.approx(29670 - 21466) and q4["derived"]
    assert q4["ts_release"] == "2026-02-13"


def test_a_filing_is_invisible_before_it_is_filed():
    before = sg.assess(OBS, CIK, "2026-08-04", REVENUE, PROFIT)
    assert before.quarter == "2025-12-31", "the 2026-Q2 10-Q lands on 2026-08-05"
    assert before.profit_concept is None or before.profit_concept == "OldEbitda"


def test_no_segment_revenue_no_view():
    assert sg.assess(OBS[OBS["series_id"].str.contains("Ebitda")], CIK, "2026-09-28",
                     REVENUE, PROFIT) is None
