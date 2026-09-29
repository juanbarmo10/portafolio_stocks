"""The post-earnings drift study's plumbing (CLAUDE.md §15.5, point 6), checked before the
single run with known answers.

What is defended: the reaction spans the session before the filing date to the session after
it, the drift starts only after that window (never counting the reaction as drift), older
submission pages are read, and the decision rule reads the direction of each signal.
"""

from __future__ import annotations

import pandas as pd
import pytest

from validation import pead

DAYS = pd.bdate_range("2026-01-05", periods=80)


def series(values):
    return pd.Series(values, index=DAYS[: len(values)], dtype=float)


def test_the_reaction_is_three_sessions_around_the_filing_date_net_of_spy():
    # Filing on DAYS[10]: window closes DAYS[9] → DAYS[11].
    stock = series([100.0] * 10 + [100.0, 112.0] + [112.0] * 68)
    spy = series([100.0] * 11 + [102.0] + [102.0] * 68)
    value, end = pead.reaction(stock, spy, DAYS[10])
    assert value == pytest.approx(0.12 - 0.02)
    assert end == DAYS[11]


def test_a_filing_on_a_weekend_starts_at_the_next_session():
    stock = series([100.0] * 80)
    _value, end = pead.reaction(stock, stock, pd.Timestamp("2026-01-10"))   # a Saturday
    assert end == pd.Timestamp("2026-01-13"), "session 0 = Monday 12th, +1 = Tuesday 13th"


def test_the_drift_starts_after_the_reaction_window():
    """A jump inside the window must not appear as drift."""
    stock = series([100.0] * 11 + [120.0] * 69)
    spy = series([100.0] * 80)
    events = pd.DataFrame({"symbol": ["AAA"], "date": [DAYS[10]]})
    member = pd.DataFrame(True, index=DAYS, columns=["AAA"])
    scored = pead.score_events(events, pd.DataFrame({"AAA": stock}), spy, member, (-1, 1), [30])
    assert scored.iloc[0]["reaction"] == pytest.approx(0.20)
    assert scored.iloc[0]["excess_30"] == pytest.approx(0.0)


def test_events_outside_the_index_are_left_out():
    stock = series([100.0] * 80)
    member = pd.DataFrame(False, index=DAYS, columns=["AAA"])
    events = pd.DataFrame({"symbol": ["AAA"], "date": [DAYS[10]]})
    assert pead.score_events(events, pd.DataFrame({"AAA": stock}), stock, member, (-1, 1),
                             [30]).empty


def test_earnings_dates_come_from_every_page_and_only_item_202():
    recent = {"form": ["8-K", "8-K", "10-Q"], "items": ["2.02,9.01", "5.02", ""],
              "filingDate": ["2026-08-05", "2026-07-01", "2026-08-05"]}
    older = {"form": ["8-K", "8-K/A"], "items": ["2.02", "2.02"],
             "filingDate": ["2018-02-01", "2018-02-03"]}
    assert pead.earnings_dates_from([recent, older]) == ["2018-02-01", "2026-08-05"]


def result(signal, mean, base, halves, significant=True):
    return pead.Result(signal=signal, horizon=90, n=10, n_base=10, mean=mean, base_mean=base,
                       median=None, hit_rate=None, p=0.01, q=0.01, significant=significant,
                       halves=halves)


def test_the_decision_reads_the_direction_of_each_signal():
    cfg = {"decision_horizons": [90]}
    assert pead.decision([result("positive", 0.03, 0.01, (0.02, 0.01))], cfg)["positive"]
    assert pead.decision([result("negative", -0.03, 0.01, (-0.02, -0.05))], cfg)["negative"]
    assert not pead.decision([result("negative", 0.03, 0.01, (0.02, 0.01))], cfg)["negative"], \
        "a negative reaction followed by a RISE is the opposite of drift"
    assert not pead.decision([result("positive", 0.03, 0.01, (0.02, -0.01))], cfg)["positive"], \
        "the sign must hold in both halves"
    assert not pead.decision([result("positive", 0.03, 0.01, (0.02, 0.01), False)],
                             cfg)["positive"]
