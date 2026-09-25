"""How the stock behaved (CLAUDE.md §15.1.2), on series whose answers are known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transform import price_action as pa

DAYS = pd.bdate_range("2025-01-01", periods=300)


def test_a_stock_that_is_the_market_has_beta_one_and_no_excess():
    rng = np.random.default_rng(0)
    market = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(DAYS))), index=DAYS)
    b = pa.behaviour(market, market, DAYS[-1], days=365)
    assert b.beta == pytest.approx(1.0)
    assert b.excess == pytest.approx(0.0)


def test_a_twice_leveraged_stock_has_beta_two():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0003, 0.01, len(DAYS))
    market = pd.Series(100 * np.cumprod(1 + r), index=DAYS)
    stock = pd.Series(100 * np.cumprod(1 + 2 * r), index=DAYS)
    assert pa.behaviour(stock, market, DAYS[-1]).beta == pytest.approx(2.0)


def test_drawdowns_are_measured_from_the_running_peak():
    series = pd.Series([100, 120, 90, 110], index=pd.bdate_range("2025-01-01", periods=4),
                       dtype=float)
    b = pa.behaviour(series, pd.Series(dtype=float, index=pd.DatetimeIndex([])),
                     series.index[-1])
    assert b.max_drawdown == pytest.approx(90 / 120 - 1)
    assert b.drawdown_now == pytest.approx(110 / 120 - 1)
    assert b.total_return == pytest.approx(0.10)
    assert b.beta is None and b.benchmark_return is None, "no benchmark, no comparison"


def test_the_reaction_window_spans_the_session_before_and_after_the_filing():
    """Filed on day 2 — before the open or after the close, the SEC does not say: the
    window runs from day 1's close to day 3's close, which holds the reaction either way."""
    days = pd.bdate_range("2025-03-03", periods=5)
    stock = pd.Series([100, 100, 100, 120, 120], index=days, dtype=float)
    market = pd.Series([100, 100, 100, 101, 101], index=days, dtype=float)
    r = pa.earnings_reactions(stock, market, [days[2].date().isoformat()])
    assert r.loc[0, "move"] == pytest.approx(0.20)
    assert r.loc[0, "excess"] == pytest.approx(0.19)
    assert pa.typical_reaction(r) == pytest.approx(0.19)


def test_a_filing_at_the_edge_of_the_data_is_left_out():
    stock = pd.Series([100.0, 101.0], index=pd.bdate_range("2025-03-03", periods=2))
    assert pa.earnings_reactions(stock, stock, ["2025-03-04"]).empty


def test_rebasing_starts_at_one_hundred():
    series = pd.Series([50.0, 55.0, 60.0], index=pd.bdate_range("2025-01-01", periods=3))
    assert list(pa.rebased(series, "2025-01-02")) == [100.0, pytest.approx(60 / 55 * 100)]
