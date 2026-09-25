"""Valuation (CLAUDE.md §15.1): point-in-time multiples and the reverse DCF.

Fundamentals from the frozen Microsoft companyfacts fixture; prices synthetic, because the
price is the input whose effect each test isolates.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from core import config
from ingest.sec_xbrl import extract_metric
from transform import valuation as val

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "sec_companyfacts_msft.json"
CIK = "0000789019"


@pytest.fixture(scope="module")
def observations() -> pd.DataFrame:
    facts = json.loads(FIXTURE.read_text(encoding="utf-8"))["facts"]
    cfg = config.load_settings().source("sec")
    records = []
    for metric, spec in cfg["concepts"].items():
        rows, _ = extract_metric(facts, metric, spec, CIK, cfg["duration_windows"])
        records.extend(rows)
    return pd.DataFrame(records)


def closes(*pairs) -> pd.DataFrame:
    return pd.DataFrame([{"series_id": "MSFT:close_raw", "ts": f"{d}T00:00:00+00:00",
                          "value": v} for d, v in pairs])


def test_the_price_is_the_last_close_on_or_before_the_date():
    prices = closes(("2025-09-10", 500.0), ("2025-09-12", 510.0))
    assert val.price_at(prices, "MSFT", "2025-09-11") == (500.0, "2025-09-10")
    assert val.price_at(prices, "MSFT", "2025-09-09") == (None, None)


def test_the_market_cap_is_price_times_the_newest_count(observations):
    v = val.assess(observations, closes(("2025-09-12", 500.0)), [], CIK, "MSFT", "2025-09-12")
    shares, _ = val.latest_share_count(observations, CIK, "2025-09-12")
    assert v.market_cap == pytest.approx(500.0 * shares)
    assert 7e9 < shares < 8e9, "Microsoft has ~7,45 B diluted shares"


def test_a_split_after_the_filing_scales_the_count_not_the_cap(observations):
    """A 4:1 split between the filing and the price: the raw price quarters, the count must
    quadruple, or the market cap would quarter too (section 9.1)."""
    split = [{"ticker": "MSFT", "kind": "split", "ex_date": "2025-09-01T00:00:00+00:00",
              "ratio": 4.0, "source": "yfinance"}]
    before = val.assess(observations, closes(("2025-08-29", 500.0)), split, CIK, "MSFT", "2025-08-29")
    after = val.assess(observations, closes(("2025-09-02", 125.0)), split, CIK, "MSFT", "2025-09-02")
    assert after.market_cap == pytest.approx(before.market_cap)


def test_a_multiple_over_a_negative_base_is_not_shown_and_the_yield_keeps_its_sign():
    bases = {"shares": 100.0, "shares_date": "2025-01-01", "debt": None, "cash": 50.0,
             "revenue": 1000.0, "ebit": -10.0, "net_income": -20.0, "fcf": 30.0, "sbc": 60.0}
    v = val.combine(bases, pd.DataFrame([{"series_id": "X:close_raw", "ts": "2025-06-02",
                                          "value": 10.0}]), [], "X", "2025-06-02")
    assert v.market_cap == 1000.0 and v.enterprise_value == 950.0, "no debt concept → none"
    assert v.pe is None and v.ev_ebit is None, "P/E −50 would read as cheap"
    assert v.earnings_yield == pytest.approx(-0.02)
    assert v.fcf_after_sbc_yield == pytest.approx(-0.03), "SBC is a real cost"
    assert v.p_fcf == pytest.approx(1000 / 30)


def test_the_history_is_the_same_as_assessing_every_date(observations):
    """Computing fundamentals only at filing dates must not change a single figure."""
    prices = closes(*[(d.date().isoformat(), 400.0 + i)
                      for i, d in enumerate(pd.bdate_range("2023-01-02", "2025-09-30"))])
    dates = pd.date_range("2023-06-30", "2025-09-30", freq="ME")
    fast = val.history(observations, prices, [], CIK, "MSFT", dates)
    slow = [val.assess(observations, prices, [], CIK, "MSFT", d) for d in dates]
    for m in val.MULTIPLES:
        expected = pd.Series([getattr(v, m) for v in slow], dtype=float)
        pd.testing.assert_series_equal(fast[m].astype(float), expected, check_names=False)


def test_the_percentile_places_today_in_its_own_history():
    assert val.percentile_in_history(pd.Series([1.0, 2.0, 3.0, 4.0]), 3.0) == 0.75
    assert val.percentile_in_history(pd.Series([None, None]), 3.0) is None


def test_the_reverse_dcf_recovers_the_growth_it_was_built_with():
    target = val.present_value(100.0, 0.15, 0.10, 0.025, 10)
    found = val.implied_growth(target, 100.0, 0.10, terminal_growth=0.025, years=10)
    assert found.growth == pytest.approx(0.15, abs=1e-6)


def test_the_reverse_dcf_refuses_what_it_cannot_answer():
    assert val.implied_growth(1e9, -5.0, 0.10).growth is None, "negative base"
    assert "negativa" in val.implied_growth(1e9, -5.0, 0.10).note
    assert val.implied_growth(1e9, 5.0, 0.02, terminal_growth=0.025).growth is None
    assert val.implied_growth(None, 5.0, 0.10).growth is None
    assert "más de" in val.implied_growth(1e15, 1.0, 0.10).note, "beyond the search range"


def test_the_band_places_today_within_its_own_history_and_skips_thin_ones():
    bases = {"shares": 100.0, "shares_date": "2025-01-01", "debt": 0.0, "cash": 0.0,
             "revenue": 1000.0, "ebit": 100.0, "net_income": 80.0, "fcf": 50.0, "sbc": 0.0}
    now = val.combine(bases, pd.DataFrame([{"series_id": "X:close_raw", "ts": "2025-06-02",
                                            "value": 10.0}]), [], "X", "2025-06-02")
    history = pd.DataFrame({"ev_sales": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                            "pe": [10.0, 12.0, None, None, None, None]})
    band = val.band(now, history).set_index("multiple")
    assert list(band.index) == ["ev_sales"], "four months of P/E is not a band"
    assert band.loc["ev_sales", "today"] == pytest.approx(1.0)
    assert band.loc["ev_sales", "median"] == pytest.approx(1.75)
    assert band.loc["ev_sales", "percentile"] == pytest.approx(2 / 6)
