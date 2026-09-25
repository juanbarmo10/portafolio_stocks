"""Portfolio risk and shared failure modes (CLAUDE.md §15.1.8, §5.3), synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transform import risk

DAYS = pd.bdate_range("2024-01-01", periods=300)


def test_risk_contributions_add_up_to_the_portfolio_volatility_and_cash_adds_none():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame({"A": rng.normal(0, 0.02, 300), "B": rng.normal(0, 0.01, 300)},
                           index=DAYS)
    out = risk.risk_breakdown({"A": 0.3, "B": 0.3, "CASH": 0.4}, returns)
    table = out.table.set_index("ticker")
    assert table["contribution"].sum() == pytest.approx(out.portfolio_volatility)
    assert table.loc["CASH", "contribution"] == 0.0
    assert table.loc["A", "share_of_risk"] > table.loc["B", "share_of_risk"], \
        "same weight, twice the volatility, more of the risk"


def test_only_written_limits_are_checked():
    weights = {"A": 0.25, "B": 0.10, "CASH": 0.65}
    assert risk.limit_breaches(weights, {}, None, None) == [], "unwritten checks nothing"
    breaches = risk.limit_breaches(weights, {"A": "x", "B": "x"}, 0.20, 0.30)
    assert len(breaches) == 2 and "A" in breaches[0] and "«x»" in breaches[1]


def test_a_sensitivity_is_recovered_and_its_noise_is_labelled():
    weeks = pd.date_range("2022-01-07", periods=160, freq="W-FRI")
    rng = np.random.default_rng(1)
    factors = pd.DataFrame({"market": rng.normal(0, 0.02, 160),
                            "size": rng.normal(0, 0.01, 160)}, index=weeks)
    weekly = 1.5 * factors["market"] - 2.0 * factors["size"] + rng.normal(0, 0.002, 160)
    level = pd.Series(100 * np.cumprod(1 + weekly.to_numpy()), index=weeks)
    s = risk.sensitivity(level, factors, "X")
    assert s.betas["market"] == pytest.approx(1.5, abs=0.1)
    assert s.betas["size"] == pytest.approx(-2.0, abs=0.2)
    modes = risk.failure_modes([s]).set_index("factor")
    assert modes.loc["size", "move"] == pytest.approx(-2.0 * -0.10, abs=0.02)
    assert bool(modes.loc["size", "reliable"])


def test_too_little_history_gives_no_sensitivity():
    weeks = pd.date_range("2025-01-03", periods=30, freq="W-FRI")
    factors = pd.DataFrame({"market": np.linspace(0, 0.01, 30)}, index=weeks)
    level = pd.Series(np.linspace(100, 110, 30), index=weeks)
    assert risk.sensitivity(level, factors, "X") is None
