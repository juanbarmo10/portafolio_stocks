"""Fundamental inflection (CLAUDE.md §15.5, point 5), on synthetic quarters with known answers.

What is defended: acceleration is this quarter's year-on-year growth minus the previous
quarter's; the gross margin compares with the same quarter a year earlier (seasonality
cancels); thresholds are inclusive and the user's; and the alert is news once per filing —
a quarter the user has had months to read never fires.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alerts import rules
from transform import inflection as infl

CIK = "0000000007"
ENDS = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"]
FILED = ["2025-05-01", "2025-08-01", "2025-11-01", "2026-02-15", "2026-05-01", "2026-08-01"]


def observations(revenue, gross):
    rows = []
    for end, filed, r, g in zip(ENDS, FILED, revenue, gross):
        for metric, value in (("revenue", r), ("gross_profit", g)):
            if value is not None:
                rows.append({"source": "sec", "series_id": f"{CIK}:{metric}:q", "ts": end,
                             "ts_release": filed, "value": float(value)})
    return pd.DataFrame(rows)


# Q1-26 grows 10 % on Q1-25, Q2-26 grows 30 % on Q2-25: +20 points. Gross margin Q2-25 60 %,
# Q2-26 50 %: −10 points year on year — and 0 against Q1-26 (also 50 %), so a sequential
# comparison would miss it.
OBS = observations(revenue=[100, 100, 100, 100, 110, 130],
                   gross=[60, 60, 60, 60, 55, 65])


def test_acceleration_and_margin_turn_of_the_latest_quarter():
    f = infl.assess(OBS, CIK, "2026-09-28")
    assert f.quarter == "2026-06-30"
    assert f.growth == pytest.approx(0.30) and f.previous_growth == pytest.approx(0.10)
    assert f.acceleration == pytest.approx(0.20)
    assert f.gross_margin_change == pytest.approx(0.50 - 0.60)
    assert [k for k, _ in f.signals] == ["accelerates", "margin_down"]
    assert "ACELERA" in f.signals[0][1] and "+20,0 pp" in f.signals[0][1]


def test_below_the_thresholds_there_is_no_signal():
    f = infl.assess(OBS, CIK, "2026-09-28", acceleration=0.25, gross_margin=0.15)
    assert f.signals == [] and f.acceleration == pytest.approx(0.20)


def test_a_slowdown_is_named_as_such():
    f = infl.assess(observations([100, 100, 100, 100, 130, 110], [None] * 6), CIK,
                    "2026-09-28")
    assert [k for k, _ in f.signals] == ["slows"] and f.gross_margin_change is None


def test_a_quarter_is_news_only_within_the_lookback():
    assert infl.is_new(OBS, CIK, "2026-08-15", 30), "filed 2026-08-01"
    assert not infl.is_new(OBS, CIK, "2026-09-28", 30), "read for two months already"
    assert infl.assess(OBS, CIK, "2026-07-31").quarter == "2026-03-31", "point-in-time"


def snapshot(today):
    return rules.Snapshot(now=pd.Timestamp(today, tz="UTC"), fundamentals=OBS,
                          researched={CIK: "AAA"})


def test_the_alert_fires_on_the_filing_once_and_never_on_an_old_quarter():
    params = {"acceleration": 0.10, "gross_margin": 0.03, "lookback_days": 30}
    fired = rules.fundamental_inflection(snapshot("2026-08-15"), params)
    assert [a.key for a in fired] == [f"fundamental_inflection:{CIK}:2026-06-30:accelerates",
                                      f"fundamental_inflection:{CIK}:2026-06-30:margin_down"]
    assert "AAA" in fired[0].text
    assert rules.fundamental_inflection(snapshot("2026-09-28"), params) == []
