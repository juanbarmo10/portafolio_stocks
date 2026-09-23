"""Corporate-action adjustment — **CI-blocking** (CLAUDE.md sections 9.1, 10).

The counterpart of ``test_prices.py``. That file defends the way in: raw prices are
reconstructed and stored, so the history never moves. This one defends the way out: the
adjusted series is rebuilt *to a stated date*, and rebuilding it for a past date must give
the same answer forever, no matter what has happened since.

The test section 9.1 mandates by name is
:func:`test_a_past_series_does_not_change_when_a_later_dividend_is_ingested`, with its
stronger sibling for a later split — the case that actually moves the numbers.

Against AAPL's real bars around its 4:1 split of 2020-08-31: the close was 499.23 on
2020-08-28 and 129.04 the day after the split. Both facts, three days apart, and any
adjustment that cannot hold both is wrong.
"""

from __future__ import annotations

import datetime as dt
import pathlib

import pandas as pd
import pytest

from ingest.prices import split_factor
from transform.adjustments import (
    actions_until,
    adjustment_factor,
    split_adjusted,
    total_return_index,
    unclean_split_ratios,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
SPLIT_EX_DATE = dt.date(2020, 8, 31)
AAPL_SPLITS = {SPLIT_EX_DATE: 4.0}


def raw_closes() -> pd.DataFrame:
    """AAPL raw closes as this project stores them: ``[ts, value]``, un-adjusted."""
    frame = pd.read_csv(FIXTURES / "yfinance_aapl_split_2020.csv", index_col="date")
    rows = []
    for stamp, row in frame.iterrows():
        day = dt.date.fromisoformat(str(stamp)[:10])
        rows.append({"ts": day.isoformat(),
                     "value": float(row["Close"]) * split_factor(day, AAPL_SPLITS)})
    return pd.DataFrame(rows)


def split_row(ex_date: dt.date, ratio: float, ticker: str = "AAPL") -> dict:
    return {"action_id": f"{ticker}:split:{ex_date}", "cik": None, "ticker": ticker,
            "kind": "split", "ex_date": ex_date.isoformat(), "ratio": ratio,
            "amount": None, "currency": None, "source": "yfinance"}


def dividend_row(ex_date: dt.date, amount: float, ticker: str = "AAPL") -> dict:
    return {"action_id": f"{ticker}:dividend:{ex_date}", "cik": None, "ticker": ticker,
            "kind": "dividend", "ex_date": ex_date.isoformat(), "ratio": None,
            "amount": amount, "currency": "USD", "source": "yfinance"}


ACTIONS = [split_row(SPLIT_EX_DATE, 4.0)]


# --- The stored facts ---------------------------------------------------------


def test_the_stored_series_holds_both_real_prices():
    """Guards the guard: if the fixture stopped being raw, everything below is vacuous."""
    closes = raw_closes().set_index("ts")["value"]

    assert closes["2020-08-28"] == pytest.approx(499.23, abs=0.01)
    assert closes["2020-08-31"] == pytest.approx(129.04, abs=0.01)


# --- The look-ahead guard (section 9.1) ---------------------------------------


def test_a_past_series_does_not_change_when_a_later_dividend_is_ingested():
    """**The test section 9.1 names.** An adjusted past must be immutable.

    A dividend paid after the cutoff is information the cutoff date did not have. Because
    a price adjustment is applied to *every earlier bar*, letting one in does not corrupt a
    row, it rewrites the whole history — which is why section 9.1 calls this trap more
    insidious than the ``ts_release`` look-ahead it mirrors: the series still looks clean.
    """
    cutoff = "2020-08-28"
    # Synthetic ex-date, and deliberately so: it has to fall *inside* the fixture's ten
    # bars or the assertion has nothing to bite on. AAPL's real next dividend was
    # 2020-11-06, outside this window — using it would make the test pass by accident,
    # which was how this test was first written.
    later = [*ACTIONS, dividend_row(dt.date(2020, 9, 3), 0.82)]

    pd.testing.assert_frame_equal(
        split_adjusted(raw_closes(), ACTIONS, "AAPL", cutoff),
        split_adjusted(raw_closes(), later, "AAPL", cutoff),
    )
    # On the price series alone this holds trivially, because this project never
    # back-adjusts a price for a dividend — which is the design section 9.1 asks for, and
    # the reason a downloaded "adjusted close" is not reproducible. So the same invariant
    # is asserted where a dividend *does* move the number: the total-return index.
    pd.testing.assert_frame_equal(
        total_return_index(raw_closes(), ACTIONS, "AAPL", cutoff),
        total_return_index(raw_closes(), later, "AAPL", cutoff),
    )


def test_a_past_series_does_not_change_when_a_later_split_is_ingested():
    """The stronger case: a later split *would* move every number if it leaked in."""
    cutoff = "2020-09-01"
    before = split_adjusted(raw_closes(), ACTIONS, "AAPL", cutoff)

    later = [*ACTIONS, split_row(dt.date(2024, 6, 10), 10.0)]
    after = split_adjusted(raw_closes(), later, "AAPL", cutoff)

    pd.testing.assert_frame_equal(before, after)
    # And with the cutoff moved past it, the same call must change — otherwise this test
    # would also pass against a function that ignores splits entirely.
    moved = split_adjusted(raw_closes(), later, "AAPL", "2024-07-01")
    assert not moved["value"].equals(before["value"])


def test_the_cutoff_filters_the_actions_and_not_the_bars():
    """A deliberate boundary, asserted so it stays a decision.

    ``as_of`` says which corporate actions were *knowable*; it does not truncate the price
    history. Truncating by publication date is ``transform.macro.point_in_time``'s job and
    lives there once — daily bars are stored with ``ts_release == ts``, so that function
    already answers it. Doing it in both places invites the two to disagree.
    """
    adjusted = split_adjusted(raw_closes(), ACTIONS, "AAPL", "2020-08-28")
    assert adjusted["ts"].iloc[-1] == "2020-09-04"


def test_before_the_split_the_adjusted_price_is_the_price_that_existed():
    """Standing on 2020-08-28, AAPL cost 499.23. Nothing had happened to change that."""
    adjusted = split_adjusted(raw_closes(), ACTIONS, "AAPL", "2020-08-28")
    value = adjusted.set_index("ts")["value"]["2020-08-28"]

    assert value == pytest.approx(499.23, abs=0.01)


def test_after_the_split_the_same_bar_reads_in_new_share_units():
    """The same day, seen from after the split: 124.81. Both readings are correct."""
    adjusted = split_adjusted(raw_closes(), ACTIONS, "AAPL", "2020-09-30")
    value = adjusted.set_index("ts")["value"]["2020-08-28"]

    assert value == pytest.approx(124.81, abs=0.01)


# --- The ex-date boundary -----------------------------------------------------


def test_the_ex_date_bar_is_not_rescaled():
    """Strictly after, never on. Including the ex-date bar spikes it by the full ratio."""
    assert adjustment_factor(dt.date(2020, 8, 31), AAPL_SPLITS) == 1.0
    assert adjustment_factor(dt.date(2020, 8, 30), AAPL_SPLITS) == 4.0

    adjusted = split_adjusted(raw_closes(), ACTIONS, "AAPL", "2020-09-30")
    assert adjusted.set_index("ts")["value"]["2020-08-31"] == pytest.approx(129.04, abs=0.01)


def test_adjustment_is_the_exact_inverse_of_the_ingest_reconstruction():
    """A round trip through the database must not change a number.

    ``ingest.prices.split_factor`` multiplies on the way in; this divides on the way out.
    If the two ever disagree on the boundary rule, prices drift by a whole split factor on
    one side of one date, which is invisible in a chart.
    """
    for day in (dt.date(2020, 8, 28), SPLIT_EX_DATE, dt.date(2020, 9, 4)):
        assert adjustment_factor(day, AAPL_SPLITS) == split_factor(day, AAPL_SPLITS)


# --- Reading the actions table ------------------------------------------------


def test_actions_are_filtered_by_cutoff_and_source():
    ibkr = {**split_row(dt.date(2021, 1, 4), 2.0), "source": "ibkr"}
    actions = [*ACTIONS, split_row(dt.date(2024, 6, 10), 10.0), ibkr]

    assert actions_until(actions, "AAPL", "2020-12-31", "split", "yfinance") == AAPL_SPLITS
    assert actions_until(actions, "AAPL", "2026-01-01", "split", "yfinance") == {
        SPLIT_EX_DATE: 4.0, dt.date(2024, 6, 10): 10.0,
    }
    assert actions_until(actions, "AAPL", "2026-01-01", "split", "ibkr") == {
        dt.date(2021, 1, 4): 2.0,
    }


def test_the_same_event_reported_twice_is_not_applied_twice():
    """Two providers, one split. Multiplying both would square the ratio."""
    duplicated = [split_row(SPLIT_EX_DATE, 4.0),
                  {**split_row(SPLIT_EX_DATE, 4.0), "source": "yfinance",
                   "action_id": "other"}]

    assert actions_until(duplicated, "AAPL", "2026-01-01", "split", "yfinance") == AAPL_SPLITS


def test_no_actions_returns_the_raw_series_unchanged():
    """Not a no-op to optimize away: with no split, raw *is* the adjusted series."""
    prices = raw_closes()
    pd.testing.assert_frame_equal(
        split_adjusted(prices, [], "AAPL", "2026-01-01"),
        prices.sort_values("ts")[["ts", "value"]].reset_index(drop=True),
    )


def test_an_empty_price_frame_is_empty_not_an_error():
    assert split_adjusted(pd.DataFrame(), ACTIONS, "AAPL", "2026-01-01").empty
    assert total_return_index(pd.DataFrame(), ACTIONS, "AAPL", "2026-01-01").empty


# --- Total return (section 9.1: add the dividend, never back-adjust the price) -


def test_the_dividend_is_added_on_its_date_and_the_price_series_is_untouched():
    """Section 9.1 in one assertion, on a two-bar series where the sum is checkable.

    100 -> 110 is +10%. With a 2.00 dividend on the second day the total return is
    (110 + 2) / 100 - 1 = +12%. The price series still reads 100 and 110: nothing was
    back-adjusted, which is what keeps it reproducible.
    """
    prices = pd.DataFrame([{"ts": "2026-01-02", "value": 100.0},
                           {"ts": "2026-01-03", "value": 110.0}])
    actions = [dividend_row(dt.date(2026, 1, 3), 2.0, ticker="X")]

    index = total_return_index(prices, actions, "X", "2026-01-31")
    assert index["value"].tolist() == pytest.approx([100.0, 112.0])

    untouched = split_adjusted(prices, actions, "X", "2026-01-31")
    assert untouched["value"].tolist() == [100.0, 110.0], "the price was back-adjusted"


def test_a_dividend_after_the_cutoff_does_not_reach_the_index():
    prices = pd.DataFrame([{"ts": "2026-01-02", "value": 100.0},
                           {"ts": "2026-01-03", "value": 110.0}])
    actions = [dividend_row(dt.date(2026, 1, 3), 2.0, ticker="X")]

    index = total_return_index(prices, actions, "X", "2026-01-02")
    assert index["value"].tolist() == pytest.approx([100.0, 110.0])


def test_the_total_return_index_is_scale_invariant():
    """It is rebased, so it carries no size — safe to publish (RESEARCH.md 2.8)."""
    prices = pd.DataFrame([{"ts": "2026-01-02", "value": 100.0},
                           {"ts": "2026-01-03", "value": 110.0}])
    scaled = prices.assign(value=prices["value"] * 37.0)

    pd.testing.assert_series_equal(
        total_return_index(prices, [], "X", "2026-01-31")["value"],
        total_return_index(scaled, [], "X", "2026-01-31")["value"],
    )


# --- Section 9.2: a ratio that is not a split ---------------------------------


def test_a_spinoff_encoded_as_a_split_is_flagged_not_corrected():
    """The real XLF case: GICS split out real estate in 2016 and yfinance calls it a split.

    A spin-off divides the cost base between two entities. Treating it as a split leaves
    the base whole on the original and falsifies the PnL permanently — so it is raised for
    a human, never silently reinterpreted (section 9.2).
    """
    flagged = unclean_split_ratios([
        split_row(dt.date(2016, 9, 19), 1.231, ticker="XLF"),
        split_row(SPLIT_EX_DATE, 4.0),
    ])

    assert [row["ticker"] for row in flagged] == ["XLF"]
    assert "spin-off" in flagged[0]["detail"] or "spin" in flagged[0]["detail"].lower()


def test_ordinary_split_ratios_are_not_flagged():
    """An alarm that fires on every 3:2 split is an alarm nobody reads."""
    ordinary = [split_row(dt.date(2020, 1, 2), ratio) for ratio in (2.0, 3.0, 4.0, 1.5, 0.1)]
    assert unclean_split_ratios(ordinary) == []
