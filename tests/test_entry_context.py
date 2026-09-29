"""Where the price stands before buying (CLAUDE.md §15.5, points 8 and 12), known answers.

What is defended: the conditions are exactly the study's, evaluated on what was known at the
date; and not enough history leaves them unknown, never false.
"""

from __future__ import annotations

import pandas as pd
import pytest

from transform import entry_context as ec

CONDITIONS = {"knife": {"sma_sessions": 5, "return_days": 20, "max_return": -0.20},
              "overextended": {"sma_sessions": 5, "min_ratio_to_sma": 1.30}}
DAYS = pd.bdate_range("2025-01-01", periods=300)


def test_a_hard_fall_is_a_knife_known_at_the_date():
    closes = pd.Series(100.0, index=DAYS)
    closes.iloc[250:] = 70.0
    e = ec.assess(closes, DAYS[250], CONDITIONS)
    assert e.knife and not e.overextended
    assert e.return_3m == pytest.approx(-0.30)
    assert not ec.assess(closes, DAYS[249], CONDITIONS).knife, "the day before, nothing"


def test_a_stretched_price_is_overextended():
    closes = pd.Series(100.0, index=DAYS)
    closes.iloc[250:] = 200.0
    e = ec.assess(closes, DAYS[250], CONDITIONS)
    assert e.overextended and e.distance_to_sma == pytest.approx(200 / 120 - 1)


def test_not_enough_history_is_unknown_not_false():
    e = ec.assess(pd.Series(100.0, index=DAYS[:3]), DAYS[2], CONDITIONS)
    assert e.knife is None and e.overextended is None and e.sma is None
