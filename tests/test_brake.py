"""The brake study's machinery (RESEARCH.md §2.30), on data whose answer is known.

Written and passing before the study was run on real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validation.brake import forward_drawdown, price_trend_brake, simulate_dca

DAYS = pd.bdate_range("2020-01-01", "2020-06-30")


def flat(value=100.0):
    return pd.Series(value, index=DAYS)


def test_forward_drawdown_is_the_worst_fall_after_entering():
    prices = pd.Series([100, 100, 90, 80, 95, 120], index=pd.bdate_range("2024-01-01", periods=6),
                       dtype=float)
    assert forward_drawdown(prices, prices.index[0], 5) == pytest.approx(-0.20)
    assert forward_drawdown(prices, prices.index[3], 1) == pytest.approx(0.0), "never below"
    assert forward_drawdown(prices, prices.index[4], 30) is None, "incomplete window"


def test_the_trend_brake_engages_below_the_average_only():
    prices = pd.Series(np.r_[np.full(10, 100.0), np.full(5, 50.0)],
                       index=pd.bdate_range("2024-01-01", periods=15))
    brake = price_trend_brake(prices, window=10, min_window_fraction=1.0)
    assert brake.iloc[:9].isna().all(), "no average yet, no decision"
    assert brake.iloc[9] == 0.0 and (brake.iloc[10:] == 1.0).all()


def test_always_investing_in_a_flat_market_keeps_every_unit():
    result = simulate_dca(flat(), None, pd.Series(0.0, index=DAYS), DAYS[0])
    assert result.contributions == 6
    assert result.final_wealth == pytest.approx(1.0)
    assert result.months_deferred == 0


def test_the_brake_holds_cash_that_earns_the_rate_and_invests_on_release():
    """Engaged every close of January, starting on 2 January: the January unit waits from
    2 January to 4 February (the first session whose previous close was released), 33 days
    at 3.65 % a year; the February unit, contributed on 3 February, waits one day."""
    brake = pd.Series(0.0, index=DAYS)
    brake.loc[:"2020-01-31"] = 1.0
    rate = pd.Series(3.65, index=DAYS)
    result = simulate_dca(flat(), brake, rate, DAYS[1])
    assert result.months_deferred == 2
    waited = DAYS[(DAYS >= "2020-01-02") & (DAYS <= "2020-02-04")]
    january = np.prod([1 + 0.0001 * g.days for g in waited[1:] - waited[:-1]])
    assert january == pytest.approx(1 + 33 * 0.0001, abs=1e-4), "33 days of interest"
    assert result.final_wealth == pytest.approx((4 + january + 1.0001) / 6, abs=1e-5)


def test_the_brake_decided_at_a_close_applies_to_the_next_session():
    """Engaged only on the first contribution day itself: that close cannot have been known
    when buying at it, so the purchase goes through."""
    brake = pd.Series(0.0, index=DAYS)
    brake.iloc[0] = 1.0
    assert simulate_dca(flat(), brake, flat(0.0), DAYS[0]).months_deferred == 0


def test_an_undecided_brake_does_not_block():
    brake = pd.Series(np.nan, index=DAYS)
    assert simulate_dca(flat(), brake, flat(0.0), DAYS[0]).months_deferred == 0


def test_waiting_through_a_fall_buys_cheaper():
    """The case a brake exists for: prices halve in February and stay there."""
    prices = pd.Series(100.0, index=DAYS)
    prices.loc["2020-02-15":] = 50.0
    brake = pd.Series(0.0, index=DAYS)
    brake.loc["2020-01-15":"2020-02-20"] = 1.0
    zero = flat(0.0)
    held = simulate_dca(prices, brake, zero, DAYS[0]).final_wealth
    always = simulate_dca(prices, None, zero, DAYS[0]).final_wealth
    assert held > always
