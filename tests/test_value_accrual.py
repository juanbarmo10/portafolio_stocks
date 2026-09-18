"""Shareholder value accrual, against the frozen slice of Microsoft's real filings.

Two tests carry the module:

:func:`test_a_buyback_that_mostly_offsets_dilution_is_visible` is the reason section 2
calls this the governing concept. Microsoft spent 20.3 billion net on repurchases over the
trailing year and the diluted share count fell by 12 million shares — about 1,689 dollars
per share actually removed, against a stock trading near 500. The gross buyback figure says
"22 billion returned to shareholders". The net effect says most of it offset stock-based
compensation. Both numbers are true; only one answers "how do I benefit?".

:func:`test_treating_a_share_count_as_a_flow_produces_nonsense` documents the trap this
module exists to avoid, by running the wrong path on purpose and showing what it returns.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from core.config import load_settings
from ingest.sec_xbrl import extract_metric
from transform import fundamentals as fun
from transform import value_accrual as va

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "sec_companyfacts_msft.json"
CIK = "0000789019"
TODAY = "2026-09-17"


@pytest.fixture(scope="module")
def observations() -> pd.DataFrame:
    cfg = load_settings().source("sec")
    facts = json.loads(FIXTURE.read_text(encoding="utf-8"))["facts"]
    records = []
    for metric, spec in cfg["concepts"].items():
        rows, _stats = extract_metric(facts, metric, spec, CIK, cfg["duration_windows"])
        records.extend(rows)
    return pd.DataFrame(records)


def obs_frame(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """``[(series_id, ts, ts_release, value)]`` in the shape the transforms consume."""
    return pd.DataFrame(
        [{"source": "sec", "series_id": s, "ts": t, "ts_release": r, "value": v}
         for s, t, r, v in rows]
    )


# --- The share count is not a flow --------------------------------------------


def test_the_share_count_is_the_reported_average_not_a_sum(observations):
    """Microsoft has ~7.45 billion diluted shares. Any answer near 30 billion is a sum."""
    count = va.share_count(observations, CIK, TODAY)
    assert count == pytest.approx(7.45e9, rel=0.01)


def test_treating_a_share_count_as_a_flow_produces_nonsense():
    """The trap, run on purpose so the guard has a reason on the record.

    A weighted-average share count is a *state*, not something that accumulates. Summing
    four quarters multiplies it by four, and deriving the fourth quarter by subtraction
    turns it negative — on Microsoft's real numbers, ``FY − Q1 − Q2 − Q3`` gives about
    minus fifteen billion shares. Neither operation raises anything; both produce a float.
    """
    quarters = fun.known(
        obs_frame([
            (f"{CIK}:diluted_shares:q", "2025-09-30", "2025-10-29", 7.466e9),
            (f"{CIK}:diluted_shares:q", "2025-12-31", "2026-01-28", 7.460e9),
            (f"{CIK}:diluted_shares:q", "2026-03-31", "2026-04-29", 7.445e9),
        ]), CIK, "diluted_shares", fun.QUARTER, TODAY)
    annual = fun.known(
        obs_frame([(f"{CIK}:diluted_shares:fy", "2026-06-30", "2026-07-29", 7.453e9)]),
        CIK, "diluted_shares", fun.ANNUAL, TODAY)

    completed, _skipped = fun.derive_fourth_quarter(quarters, annual)
    derived = completed[completed["ts"] == "2026-06-30"]["value"].iloc[0]

    assert derived < -1e10, "the wrong path used to give a negative share count"
    assert "diluted_shares" in va.AVERAGE_METRICS, "the metric must stay marked as average"


def test_dilution_is_positive_when_the_count_grows():
    """Sign convention, fixed by a test because it is the one everyone gets backwards."""
    diluting = obs_frame([
        (f"{CIK}:diluted_shares:fy", "2025-06-30", "2025-07-30", 1_000_000_000.0),
        (f"{CIK}:diluted_shares:fy", "2026-06-30", "2026-07-29", 1_100_000_000.0),
    ])
    assert va.dilution(diluting, CIK, TODAY) == pytest.approx(0.10)


def test_microsoft_is_shrinking_its_count_slightly(observations):
    value = va.dilution(observations, CIK, TODAY)
    assert value is not None and -0.01 < value < 0, "expected a small net reduction"


# --- The governing test -------------------------------------------------------


def test_a_buyback_that_mostly_offsets_dilution_is_visible(observations):
    """Section 2: "the gross buyback flatters if SBC cancels it". On real numbers.

    20.3 billion of net repurchases removed 12 million shares — roughly 1,689 dollars per
    share retired, against a stock near 500. The difference is stock-based compensation
    issuing shares as fast as the buyback retires them. A panel that showed only the gross
    figure would report this as capital returned to shareholders.
    """
    accrual = va.assess(observations, CIK, TODAY)

    assert accrual.buybacks_ttm == pytest.approx(22.271e9, rel=1e-3)
    assert accrual.issuance_ttm == pytest.approx(2.009e9, rel=1e-3)
    assert accrual.net_buybacks_ttm == pytest.approx(20.262e9, rel=1e-3)

    assert accrual.shares_removed == pytest.approx(12e6, rel=0.05)
    assert accrual.buyback_per_share_removed > 1000, (
        "the cost per share actually removed must dwarf the market price when SBC is "
        "absorbing the buyback"
    )


def test_no_shares_removed_means_no_cost_per_share_rather_than_a_negative_one():
    """Synthetic: a count that grew must not produce a negative "cost per share"."""
    observations = obs_frame([
        (f"{CIK}:diluted_shares:fy", "2025-06-30", "2025-07-30", 1_000_000_000.0),
        (f"{CIK}:diluted_shares:fy", "2026-06-30", "2026-07-29", 1_050_000_000.0),
        (f"{CIK}:buybacks:fy", "2026-06-30", "2026-07-29", 5_000_000_000.0),
    ])
    accrual = va.assess(observations, CIK, TODAY)

    assert accrual.shares_removed == pytest.approx(-50_000_000)
    assert accrual.buyback_per_share_removed is None
    assert accrual.dilution_yoy == pytest.approx(0.05)


def test_net_buybacks_subtract_issuance(observations):
    accrual = va.assess(observations, CIK, TODAY)
    assert accrual.net_buybacks_ttm < accrual.buybacks_ttm


# --- Stock-based compensation -------------------------------------------------


def test_sbc_is_reported_against_both_revenue_and_cash(observations):
    """Over revenue it looks modest; over free cash flow it is the real bite."""
    accrual = va.assess(observations, CIK, TODAY)

    assert accrual.sbc_ttm == pytest.approx(12.405e9, rel=1e-3)
    assert accrual.sbc_over_revenue == pytest.approx(0.037, abs=0.005)
    assert accrual.sbc_over_fcf == pytest.approx(0.185, abs=0.01)
    assert accrual.sbc_over_fcf > accrual.sbc_over_revenue * 3


# --- ROIC and the hurdle ------------------------------------------------------


def test_the_tax_rate_is_the_one_in_the_filings(observations):
    """Not 21% because that is the statutory rate — the observed one (section 12)."""
    rate = va.effective_tax_rate(observations, CIK, TODAY)
    assert rate == pytest.approx(0.194, abs=0.01)


def test_roic_uses_the_observed_tax_rate_and_reports_its_nopat(observations):
    accrual = va.assess(observations, CIK, TODAY)

    assert accrual.invested_capital == pytest.approx(452.5e9, rel=0.01)
    assert accrual.nopat == pytest.approx(125.1e9, rel=0.01)
    assert accrual.roic == pytest.approx(0.276, abs=0.01)


def test_roic_is_none_when_any_leg_is_missing():
    """No stand-in tax rate, no assumed capital: a missing leg means no ratio."""
    observations = obs_frame([
        (f"{CIK}:operating_income:fy", "2026-06-30", "2026-07-29", 100e9),
        (f"{CIK}:equity", "2026-06-30", "2026-07-29", 400e9),
    ])
    value, nopat = va.roic(observations, CIK, TODAY)
    assert value is None and nopat is None


def test_without_a_hurdle_the_comparison_is_unknown_not_a_pass(observations):
    """Section 12: an unset parameter renders as "no calculable", never as success."""
    assert va.assess(observations, CIK, TODAY).roic_above_hurdle is None

    with_hurdle = va.assess(observations, CIK, TODAY, hurdle_rate=0.10)
    assert with_hurdle.roic_above_hurdle == pytest.approx(with_hurdle.roic - 0.10)


def test_negative_invested_capital_is_none_rather_than_a_negative_roic():
    """Synthetic: a company whose cash exceeds equity plus debt would flip the sign."""
    observations = obs_frame([
        (f"{CIK}:equity", "2026-06-30", "2026-07-29", 10e9),
        (f"{CIK}:cash", "2026-06-30", "2026-07-29", 40e9),
    ])
    assert va.invested_capital(observations, CIK, TODAY) is None


def test_an_unknown_company_reads_as_holes(observations):
    accrual = va.assess(observations, "0000000000", TODAY)
    assert accrual.diluted_shares is None
    assert accrual.sbc_over_fcf is None
    assert accrual.roic is None
