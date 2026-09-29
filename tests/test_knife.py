"""The falling-knife study's plumbing (CLAUDE.md §15.5, point 12), checked before the single
run with known answers.

What is defended: the conditions use only what was known on the date; the forward return
starts at the next session; the comparison is within each date (a market-wide fall moves both
sides equally and cancels); and a gate needs the condition to do worse in both halves.
"""

from __future__ import annotations

import pandas as pd
import pytest

from validation import knife

DAYS = pd.bdate_range("2020-01-01", periods=400)
CFG = {"conditions": {"knife": {"sma_sessions": 5, "return_days": 20, "max_return": -0.20},
                      "overextended": {"sma_sessions": 5, "min_ratio_to_sma": 1.30}},
       "horizons_days": [30], "min_names": 1, "grid_step_days": 60,
       "first_date": "2020-03-02", "decision_horizons": [30], "permutations": 200, "seed": 0,
       "fdr_alpha": 0.10, "halves_split": "2020-12-31"}


def test_the_flags_use_only_what_was_known_on_the_date():
    fall = pd.Series(100.0, index=DAYS)
    fall.iloc[100:] = 70.0                                  # −30 % on day 100
    jump = pd.Series(100.0, index=DAYS)
    jump.iloc[100:] = 200.0                                 # far above its average
    closes = pd.DataFrame({"FALL": fall, "JUMP": jump, "FLAT": 100.0}, index=DAYS)
    sma = closes.rolling(5, min_periods=5).mean()
    before = knife.flags_at(closes, sma, DAYS[99], CFG)
    assert not before["knife"].any() and not before["overextended"].any(), "not yet"
    on = knife.flags_at(closes, sma, DAYS[100], CFG)
    assert on.loc["FALL", "knife"] and not on.loc["FLAT", "knife"]
    assert on.loc["JUMP", "overextended"]


def test_the_forward_return_starts_at_the_next_session():
    closes = pd.DataFrame({"A": pd.Series(100.0, index=DAYS)})
    closes.iloc[51:, 0] = 110.0                             # moves on the entry session itself
    bench = pd.Series(100.0, index=DAYS)
    out = knife.forward_excess(closes, bench, DAYS[50], 30)
    assert out["A"] == pytest.approx(0.0), "entry at DAYS[51]'s close, already 110"


def test_a_market_wide_fall_cancels_within_the_date():
    """Every stock falls with the market after the date: the spread is the relative move."""
    base = pd.Series(100.0, index=DAYS)
    closes = pd.DataFrame({"K": base.copy(), "R1": base.copy(), "R2": base.copy()})
    # K slides from 100 to 70 over days 90-110: on day 110 it is below its 5-session average
    # and −30 % in 20 days — a knife — and then stays flat.
    closes.loc[DAYS[90]:DAYS[110], "K"] = [100 - 1.5 * i for i in range(21)]
    closes.loc[DAYS[111]:, "K"] = 70.0
    closes.loc[DAYS[130]:, ["K", "R1", "R2"]] *= 0.8        # everyone falls 20 % later
    closes.loc[DAYS[130]:, "K"] *= 0.9                      # K falls 10 % more
    bench = pd.Series(100.0, index=DAYS)
    member = pd.DataFrame(True, index=DAYS, columns=closes.columns)
    cfg = {**CFG, "horizons_days": [90], "first_date": str(DAYS[110].date()),
           "grid_step_days": 1000}
    table = knife.spreads(closes, bench, member, cfg)
    row = table[table["condition"] == "knife"].iloc[0]
    assert row["spread"] == pytest.approx(0.8 * 0.9 - 0.8, abs=1e-9)


def result(mean, halves):
    return knife.Result(condition="knife", horizon=30, dates=10, mean_spread=mean,
                        median_names=5, negative_share=0.7, p=0.01, q=0.01, significant=True,
                        halves=halves)


def test_a_gate_needs_worse_in_both_halves():
    assert knife.decision([result(-0.02, (-0.03, -0.01))], CFG)["knife"]
    assert not knife.decision([result(-0.02, (-0.05, 0.01))], CFG)["knife"]
    assert not knife.decision([result(0.02, (0.03, 0.01))], CFG)["knife"], "better is no gate"
