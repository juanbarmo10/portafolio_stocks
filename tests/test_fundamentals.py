"""Level 3 transformations, against the frozen slice of Microsoft's real filings.

Three things are worth testing here and the rest follows from them:

1. **The derived fourth quarter is right.** ``FY − Q1 − Q2 − Q3`` for Microsoft's fiscal
   2024 gives 64.727 billion, which is what Microsoft reported for that quarter. A wrong
   derivation would still produce a plausible bar on a chart.
2. **It is not knowable early.** The same figure is invisible on 2024-07-29 and appears on
   2024-07-30, the day the 10-K was filed. That one-day step is the whole of section 9.4.
3. **Growth is geometric.** ``+100%`` then ``−50%`` is 0% a year, not the ``+25%`` an
   arithmetic mean produces (section 9.10).

The cases the fixture cannot contain — a fiscal year with a missing quarter, a negative
base for a growth rate — are synthetic and marked as such.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from core.config import load_settings
from ingest.sec_xbrl import extract_metric
from transform import fundamentals as fun

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "sec_companyfacts_msft.json"
CIK = "0000789019"
TODAY = "2026-09-17"


@pytest.fixture(scope="module")
def observations() -> pd.DataFrame:
    """The fixture run through the real ingester, as the database would hold it."""
    cfg = load_settings().source("sec")
    facts = json.loads(FIXTURE.read_text(encoding="utf-8"))["facts"]
    records = []
    for metric, spec in cfg["concepts"].items():
        rows, _stats = extract_metric(facts, metric, spec, CIK, cfg["duration_windows"])
        records.extend(rows)
    return pd.DataFrame(records)


def frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """``[(ts, ts_release, value)]`` as the shape the transforms consume."""
    return pd.DataFrame(rows, columns=["ts", "ts_release", "value"])


# --- The quarter nobody files -------------------------------------------------


def test_the_derived_fourth_quarter_matches_what_microsoft_reported(observations):
    """FY2024 minus its three filed quarters is 64.727 B — Microsoft's actual Q4."""
    quarters, _skipped = fun.quarterly(observations, CIK, "revenue", TODAY)
    q4 = quarters[quarters["ts"] == "2024-06-30"]

    assert len(q4) == 1
    assert float(q4["value"].iloc[0]) == pytest.approx(64_727_000_000, abs=1)


def test_the_derived_quarter_is_dated_by_the_10k_not_by_the_quarter_end(observations):
    """It becomes knowable when the annual report lands, not when the quarter closed.

    Dating it 30 June would hand a backtest a month of the future every single year —
    invisibly, because the number itself is correct.
    """
    quarters, _skipped = fun.quarterly(observations, CIK, "revenue", TODAY)
    q4 = quarters[quarters["ts"] == "2024-06-30"].iloc[0]
    assert q4["ts_release"] > "2024-06-30"
    assert q4["ts_release"][:4] >= "2024"


def test_a_fiscal_year_missing_a_quarter_is_skipped_and_reported(observations):
    """The fixture starts in 2023, so older years have gaps. None of them is guessed."""
    _quarters, skipped = fun.quarterly(observations, CIK, "revenue", TODAY)
    assert skipped, "the incomplete years were derived anyway"
    assert any("expected 3" in reason for reason in skipped)


def test_a_year_whose_fourth_quarter_was_filed_is_left_alone():
    """Synthetic: nothing to derive when the quarter already exists — no double counting."""
    quarters = frame([
        ("2025-03-31", "2025-04-20", 10.0), ("2025-06-30", "2025-07-20", 11.0),
        ("2025-09-30", "2025-10-20", 12.0), ("2025-12-31", "2026-01-20", 13.0),
    ])
    annual = frame([("2025-12-31", "2026-02-20", 46.0)])

    completed, skipped = fun.derive_fourth_quarter(quarters, annual)
    assert len(completed) == 4
    assert skipped == []


def test_deriving_needs_exactly_three_quarters():
    """Synthetic: two quarters plus a year is not a subtraction, it is a guess."""
    quarters = frame([("2025-03-31", "2025-04-20", 10.0), ("2025-06-30", "2025-07-20", 11.0)])
    annual = frame([("2025-12-31", "2026-02-20", 46.0)])

    completed, skipped = fun.derive_fourth_quarter(quarters, annual)
    assert len(completed) == 2, "a fourth quarter was invented"
    assert len(skipped) == 1


# --- Point-in-time (section 9.4) ----------------------------------------------


def test_the_trailing_year_changes_the_day_the_10k_is_filed(observations):
    """The cleanest demonstration of the rule this project is built around.

    On 2024-07-29 the trailing year is 236.584 B: three quarters of fiscal 2024 plus the
    last one of 2023. On 2024-07-30 the 10-K lands, the fourth quarter becomes derivable,
    and it jumps to 245.122 B — the reported fiscal year. Filtering by period end instead
    of by filing date would have shown the second number for a month before it existed.
    """
    before = fun.ttm_at(observations, CIK, "revenue", "2024-07-29")
    after = fun.ttm_at(observations, CIK, "revenue", "2024-07-30")

    assert before == pytest.approx(236_584_000_000, abs=1)
    assert after == pytest.approx(245_122_000_000, abs=1)


def test_at_fiscal_year_end_the_trailing_year_equals_the_reported_year(observations):
    """An invariant worth pinning: four quarters of a year add up to the year."""
    annual = fun.known(observations, CIK, "revenue", fun.ANNUAL, TODAY)
    fy2025 = float(annual[annual["ts"] == "2025-06-30"]["value"].iloc[0])

    quarters, _ = fun.quarterly(observations, CIK, "revenue", "2025-08-01")
    assert fun.ttm(quarters) == pytest.approx(fy2025, rel=1e-9)


def test_a_restatement_does_not_rewrite_what_was_known_before_it(observations):
    """Section 9.6: the original print stands until the revision is published."""
    early = fun.known(observations, CIK, "revenue", fun.ANNUAL, "2024-08-01")
    late = fun.known(observations, CIK, "revenue", fun.ANNUAL, TODAY)
    assert set(early["ts"]) <= set(late["ts"])
    assert len(late) >= len(early)


# --- Trailing twelve months ---------------------------------------------------


def test_ttm_refuses_to_add_up_three_quarters():
    """Section 12: a three-quarter "year" is a hole, not a smaller year."""
    assert fun.ttm(frame([
        ("2025-03-31", "2025-04-20", 10.0), ("2025-06-30", "2025-07-20", 11.0),
        ("2025-09-30", "2025-10-20", 12.0),
    ])) is None


def test_ttm_refuses_four_quarters_with_a_hole_in_the_middle():
    """Four rows spanning two years are not twelve months, however they add up."""
    assert fun.ttm(frame([
        ("2024-03-31", "2024-04-20", 10.0), ("2024-06-30", "2024-07-20", 11.0),
        ("2025-03-31", "2025-04-20", 12.0), ("2025-06-30", "2025-07-20", 13.0),
    ])) is None


def test_ttm_adds_four_contiguous_quarters():
    assert fun.ttm(frame([
        ("2025-03-31", "2025-04-20", 10.0), ("2025-06-30", "2025-07-20", 11.0),
        ("2025-09-30", "2025-10-20", 12.0), ("2025-12-31", "2026-01-20", 13.0),
    ])) == pytest.approx(46.0)


# --- Growth (section 9.10) ----------------------------------------------------


def test_cagr_is_geometric_not_an_average_of_percentages():
    """+100% then −50% leaves the value where it started. The arithmetic mean says +25%."""
    assert fun.cagr(100.0, 100.0, 2) == pytest.approx(0.0)
    assert fun.cagr(100.0, 121.0, 2) == pytest.approx(0.10)


def test_cagr_is_undefined_over_a_non_positive_base():
    """Not zero, not the arithmetic mean "because it produces a number" (section 12)."""
    assert fun.cagr(0.0, 100.0, 2) is None
    assert fun.cagr(-10.0, 100.0, 2) is None
    assert fun.cagr(None, 100.0, 2) is None


def test_growth_over_a_loss_is_none_rather_than_nonsense():
    """A growth rate over a negative base is the easiest way to publish a meaningless %."""
    assert fun.growth(50.0, -10.0) is None
    assert fun.growth(50.0, 0.0) is None
    assert fun.growth(110.0, 100.0) == pytest.approx(0.10)


def test_ratio_is_none_without_a_denominator():
    assert fun.ratio(10.0, 0) is None
    assert fun.ratio(10.0, None) is None
    assert fun.ratio(1.0, 4.0) == 0.25


# --- Snapshot -----------------------------------------------------------------


def test_snapshot_reproduces_microsofts_reported_year(observations):
    snapshot = fun.snapshot(observations, CIK, TODAY)

    assert snapshot.revenue_ttm == pytest.approx(331_839_000_000, abs=1)
    assert snapshot.net_margin == pytest.approx(0.403, abs=0.01)
    assert snapshot.fcf_ttm is not None and snapshot.fcf_ttm > 0
    # Free cash flow well below net income: the capital expenditure of the AI build-out is
    # real and this is the ratio that shows it (section 2, cash conversion).
    assert 0 < snapshot.cash_conversion < 1
    assert snapshot.quarters_available >= 8
    assert snapshot.stale_days is not None and snapshot.stale_days >= 0


def test_free_cash_flow_needs_both_legs(observations):
    """Operating cash flow over four quarters against capex over three is not FCF."""
    assert fun.free_cash_flow(observations, CIK, "2009-01-01") is None


def test_a_company_with_no_facts_reads_as_holes_not_zeros(observations):
    snapshot = fun.snapshot(observations, "0000000000", TODAY)

    assert snapshot.revenue_ttm is None
    assert snapshot.net_margin is None
    assert snapshot.assets is None
    assert snapshot.quarters_available == 0


def test_nothing_is_knowable_before_the_first_filing(observations):
    snapshot = fun.snapshot(observations, CIK, "1990-01-01")
    assert snapshot.revenue_ttm is None
    assert snapshot.equity is None
