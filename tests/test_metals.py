"""Metals as context, oil as a risk factor (CLAUDE.md §15.5, points 13 and 14).

What is defended: the gold/silver ratio only on days both prices exist; the market's falls
are found from a running peak and never overlap; an asset's return over a fall needs a close
near both ends; a crude price through zero is not a return; an absent factor is absent, not
a column of blanks that would empty every regression; and the basket correlation is right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transform import metals, risk

DAYS = pd.bdate_range("2020-01-01", periods=300)


def test_the_ratio_needs_both_prices():
    gold = pd.Series([2000.0, 2100.0, None], index=DAYS[:3])
    silver = pd.Series([25.0, 0.0, 30.0], index=DAYS[:3])
    out = metals.ratio(gold, silver)
    assert out.tolist() == [80.0], "a zero or missing price is not a ratio"
    assert metals.percentile_of_last(pd.Series([1.0, 3.0, 2.0])) == pytest.approx(2 / 3)


def test_falls_are_found_from_a_running_peak_and_do_not_overlap():
    path = [100, 110, 95, 90, 111, 120, 115, 100, 125, 130, 118]
    index = pd.Series(path, index=DAYS[:len(path)], dtype=float)
    falls = metals.drawdown_episodes(index, threshold=0.10)
    assert [(round(e.depth, 4), e.recovered) for e in falls] == [
        (round(90 / 110 - 1, 4), True), (round(100 / 120 - 1, 4), True)]
    assert falls[0].peak == DAYS[1] and falls[0].trough == DAYS[3]
    open_fall = metals.drawdown_episodes(pd.Series([100.0, 80.0], index=DAYS[:2]))
    assert len(open_fall) == 1 and not open_fall[0].recovered


def test_an_asset_that_did_not_exist_yet_is_blank_over_that_fall():
    spy = pd.Series([100.0, 80.0, 100.0], index=[DAYS[0], DAYS[50], DAYS[100]])
    falls = metals.drawdown_episodes(spy)
    late = pd.Series([10.0, 12.0], index=[DAYS[40], DAYS[50]])
    table = metals.cushion(falls, {"LATE": late, "SPY": spy})
    assert pd.isna(table.iloc[0]["LATE"]) and table.iloc[0]["SPY"] == pytest.approx(-0.2)


def test_oil_through_zero_is_not_a_return():
    weeks = pd.date_range("2020-03-06", periods=5, freq="W-FRI")
    wti = pd.Series([40.0, 20.0, -10.0, 15.0, 30.0], index=weeks)
    out = risk.oil_returns(wti)
    assert out.index.tolist() == [weeks[1], weeks[2], weeks[4]], "the week after −10 is out"
    assert out.iloc[0] == pytest.approx(-0.5)


def test_an_absent_factor_is_not_a_blank_column():
    spy = pd.Series(np.linspace(100, 130, 300), index=DAYS)
    with_oil = risk.factor_returns({"SPY": spy}, pd.Series(dtype=float),
                                   pd.Series(dtype=float), spy * 0.5)
    without = risk.factor_returns({"SPY": spy}, pd.Series(dtype=float), pd.Series(dtype=float))
    assert "oil" in with_oil.columns and "oil" not in without.columns
    assert not without.empty and without.notna().all().all()


def test_the_basket_correlation():
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2025-01-03", periods=60, freq="W-FRI")
    a = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 60)), index=weeks)
    b = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 60)), index=weeks)
    same = risk.correlation_with_basket({"A": 1.0}, {"A": a, "G": a}, ["G"], weeks[-1], 400)
    assert same["G"] == pytest.approx(1.0)
    mixed = risk.correlation_with_basket({"A": 0.5, "B": 0.5}, {"A": a, "B": b, "G": a},
                                         ["G", "NONE"], weeks[-1], 400)
    assert 0.3 < mixed["G"] < 0.99 and "NONE" not in mixed


def test_with_no_factor_at_all_there_is_no_sensitivity_and_no_crash():
    nothing = risk.factor_returns({}, pd.Series(dtype=float), pd.Series(dtype=float))
    stock = pd.Series(np.linspace(10, 20, 300), index=DAYS)
    assert risk.sensitivity(stock, nothing, "AAA") is None
