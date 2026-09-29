"""The small-company insider study's plumbing (CLAUDE.md §15.5, point 7), checked before
the single run with known answers.

What is defended: liquidity is judged only by the days before the event, an S&P 500 member
is out, excess is over the small-cap benchmark, and the decision needs the event to beat
its placebo.
"""

from __future__ import annotations

import pandas as pd
import pytest

from validation import insiders_small as st
from validation.insiders import Result

DAYS = pd.bdate_range("2020-01-01", periods=300)
CFG = {"min_price": 2.0, "min_median_dollar_volume": 1_000_000, "decision_horizons": [90]}


def frame(close, volume):
    return pd.DataFrame({"close": close, "volume": volume}, index=DAYS)


def test_liquidity_uses_only_the_days_before():
    """Volume jumps on the event day: that day is judged by the thin weeks before it, and
    the company only becomes eligible once the 60 days behind it are liquid."""
    volume = pd.Series(1_000.0, index=DAYS)
    volume.iloc[100:] = 1_000_000.0
    liquid = st.liquidity({"AAA": frame(pd.Series(10.0, index=DAYS), volume)}, 60)
    member = pd.DataFrame(False, index=DAYS, columns=["AAA"])
    assert not st.eligible("AAA", DAYS[100], member, liquid, CFG), "the event day's own volume"
    assert st.eligible("AAA", DAYS[160], member, liquid, CFG), "60 liquid days behind it"


def test_an_sp500_member_and_a_penny_stock_are_out():
    liquid = st.liquidity({"AAA": frame(pd.Series(10.0, index=DAYS),
                                        pd.Series(1e6, index=DAYS)),
                           "PEN": frame(pd.Series(1.0, index=DAYS),
                                        pd.Series(1e9, index=DAYS))}, 60)
    member = pd.DataFrame({"AAA": True, "PEN": False}, index=DAYS)
    assert not st.eligible("AAA", DAYS[200], member, liquid, CFG), "S&P member that day"
    assert not st.eligible("PEN", DAYS[200], member, liquid, CFG), "below 2 USD"


def test_excess_is_over_the_small_cap_benchmark():
    close = pd.Series(10.0, index=DAYS)
    close.iloc[150:] = 12.0
    bench = pd.Series(100.0, index=DAYS)
    bench.iloc[150:] = 105.0
    closes = {"AAA": close}
    liquid = st.liquidity({"AAA": frame(close, pd.Series(1e6, index=DAYS))}, 60)
    member = pd.DataFrame(False, index=DAYS, columns=["AAA"])
    points = pd.DataFrame({"signal": ["cluster"], "symbol": ["AAA"], "date": [DAYS[120]]})
    out = st.excess(points, closes, bench, member, liquid, CFG, [90])
    assert out.iloc[0]["excess_90"] == pytest.approx(0.20 - 0.05)


def result(mean, base, halves):
    return Result(signal="cluster", horizon=90, n=10, n_base=10, mean=mean, base_mean=base,
                  median=None, hit_rate=None, base_hit_rate=None, p=0.01, q=0.01,
                  significant=True, halves=halves)


def test_the_decision_needs_the_event_to_beat_its_placebo():
    good = [result(0.05, 0.01, (0.03, 0.02))]
    assert st.decision(good, {("cluster", 90): 0.02}, CFG)["cluster"]
    assert not st.decision(good, {("cluster", 90): -0.01}, CFG)["cluster"], \
        "the placebo did as well: the excess belongs to the company, not the purchase"
    assert not st.decision(good, {("cluster", 90): None}, CFG)["cluster"]


def test_the_median_is_exactly_the_one_of_the_prior_days():
    close = pd.Series(1.0, index=DAYS)
    volume = pd.Series(range(len(DAYS)), index=DAYS, dtype=float)   # all different
    table = st.liquidity({"AAA": frame(close, volume)}, 60)["AAA"]
    day = DAYS[120]
    prior = volume[(volume.index < day) & (volume.index >= day - pd.Timedelta(days=60))]
    assert table.loc[day, "median_dv"] == pytest.approx(prior.median())


def test_a_price_jump_on_the_event_day_does_not_count():
    close = pd.Series(1.0, index=DAYS)
    close.iloc[150:] = 10.0
    liquid = st.liquidity({"AAA": frame(close, pd.Series(1e7, index=DAYS))}, 60)
    member = pd.DataFrame(False, index=DAYS, columns=["AAA"])
    assert not st.eligible("AAA", DAYS[150], member, liquid, CFG), "yesterday's close was 1"
