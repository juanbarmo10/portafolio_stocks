"""Validation machinery (section 8 phase 4): each piece checked on data whose answer is known.

Written and passing before the real battery was run once, so nothing here was shaped by
its result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validation.backtest import LOWER, Signal, discrimination, test_signal as run_test
from validation.metrics import (
    benjamini_hochberg,
    forward_return,
    grid_dates,
    permutation_pvalue,
)

DAYS = pd.bdate_range("2010-01-01", "2020-12-31")


# --- metrics ----------------------------------------------------------------------------


def test_grid_dates_never_overlap_their_windows():
    grid = grid_dates(DAYS, 90)
    gaps = pd.Series(grid[1:] - grid[:-1]).dt.days
    assert (gaps >= 90).all() and gaps.max() <= 93, "a weekend may push it a couple of days"
    assert grid[0] == DAYS[0]
    assert grid_dates(DAYS, 90, offset=5)[0] == DAYS[5]


def test_forward_return_enters_the_session_after_the_signal():
    """The signal's own close is not tradeable on data published after it (H.4.1, 16:30)."""
    prices = pd.Series([100.0, 110.0, 121.0, 133.1], index=pd.bdate_range("2024-01-01", periods=4))
    r = forward_return(prices, prices.index[0], 1)
    assert r == pytest.approx(0.10), "entry at day 1 (110), exit at day 2 (121)"
    assert forward_return(prices, prices.index[-1], 1) is None


def test_permutation_pvalue_separates_signal_from_noise():
    rng = np.random.default_rng(1)
    same = permutation_pvalue(rng.normal(0, 1, 40), rng.normal(0, 1, 40), n=2000)
    apart = permutation_pvalue(rng.normal(2, 1, 40), rng.normal(0, 1, 40), n=2000)
    assert same > 0.05 and apart < 0.001
    assert apart > 0, "a finite shuffle cannot prove p = 0"
    assert permutation_pvalue([], [1.0]) is None


def test_benjamini_hochberg_matches_the_textbook_example():
    """p = (0.01, 0.04, 0.03, 0.20): q = (0.04, 0.0533, 0.0533, 0.20)."""
    out = benjamini_hochberg([0.01, 0.04, 0.03, 0.20], alpha=0.05)
    assert [round(q, 4) for q, _ in out] == [0.04, 0.0533, 0.0533, 0.2]
    assert [s for _, s in out] == [True, False, False, False]


# --- the battery ------------------------------------------------------------------------


def planted(edge: float, share_on: float = 0.3, seed: int = 0):
    """Prices whose 30-day forward return is ``edge`` lower after signal days."""
    rng = np.random.default_rng(seed)
    on = pd.Series(rng.random(len(DAYS)) < share_on, index=DAYS)
    # Signal persists in monthly blocks, as a regime would.
    on = on.groupby(DAYS.to_period("M")).transform("first")
    daily = rng.normal(0.0004, 0.01, len(DAYS))
    # Returns in the month after a signal month are shifted down.
    after = on.shift(21, fill_value=False).to_numpy()
    daily = daily - np.where(after, edge / 21, 0.0)
    prices = pd.Series(100 * np.cumprod(1 + daily), index=DAYS)
    return Signal("s", "s", LOWER, on.astype(float)), prices


def test_a_planted_edge_is_found_with_the_right_sign():
    signal, prices = planted(edge=0.08)
    result = run_test(signal, prices, 30, permutations=2000)
    assert result.status == "ok"
    assert result.edge < 0 and result.agrees
    assert result.pvalue < 0.01
    assert result.sign_stability == 1.0


def test_no_edge_is_not_significant():
    signal, prices = planted(edge=0.0, seed=3)
    assert run_test(signal, prices, 30, permutations=2000).pvalue > 0.05


def test_too_few_signal_dates_is_insufficient_not_a_verdict():
    signal, prices = planted(edge=0.08)
    sparse = signal.series.copy()
    sparse[:] = 0.0
    sparse.iloc[:40] = 1.0
    result = run_test(Signal("s", "s", LOWER, sparse), prices, 180)
    assert result.status == "insufficient" and result.pvalue is None


def test_where_the_signal_does_not_exist_is_neither_group():
    """NaN (the component abstains) must not be counted as "signal off"."""
    signal, prices = planted(edge=0.0)
    series = signal.series.copy()
    series.loc[:"2015-12-31"] = np.nan
    result = run_test(Signal("s", "s", LOWER, series), prices, 30, permutations=500)
    assert result.window.startswith("2016-01-01")
    assert result.n_signal + result.n_baseline <= 5 * 12 + 1


def test_discrimination_respects_its_window():
    class Built:
        votes = {"k": pd.DataFrame({"vote": pd.Series(-1.0, index=DAYS)})}
        benchmark = pd.Series(np.linspace(100, 50, len(DAYS)), index=DAYS)

    out = discrimination(Built(), "k", start="2012-01-01", end="2013-01-01")
    assert out["window"] == "2012-01-02 → 2012-12-31"
    assert out["off_in_correction"] == 1.0


def test_the_permutation_in_blocks_gives_the_same_pvalue_as_all_at_once():
    """Blocks keep memory bounded (the PEAD study was killed for a 10.000 × 60.000 matrix)
    and, drawing the same random numbers in the same order, change nothing."""
    import numpy as np  # noqa: PLC0415

    from validation.metrics import permutation_pvalue  # noqa: PLC0415

    rng = np.random.default_rng(1)
    signal, base = rng.normal(0.01, 1, 40), rng.normal(0, 1, 300)
    whole = permutation_pvalue(signal, base, n=2000, seed=7, max_cells=10**9)
    blocks = permutation_pvalue(signal, base, n=2000, seed=7, max_cells=1000)
    assert whole == blocks


def test_a_forward_return_never_enters_years_after_its_date():
    """Found 2026-09-29: a series starting in 2017 gave a 2017 return for a 1999 date, and
    the PEAD and insider baselines compared it with SPY's 1999 return."""
    from validation.metrics import forward_return

    s = pd.Series([10.0, 11.0, 12.0, 13.0],
                  index=pd.to_datetime(["2017-01-03", "2017-02-01", "2017-04-10", "2017-06-01"]))
    assert forward_return(s, pd.Timestamp("1999-03-01"), 90) is None, "no close near the date"
    assert forward_return(s, pd.Timestamp("2017-01-02"), 90) == pytest.approx(12 / 10 - 1)
    holed = s.drop(pd.Timestamp("2017-04-10"))
    assert forward_return(holed, pd.Timestamp("2017-01-02"), 90) is None, "exit 7 weeks late"
