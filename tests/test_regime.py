"""The regime traffic light (CLAUDE.md sections 2, 8 phase 3, 9.4, 9.7).

What is defended here, in order of how badly it would hurt if broken:

1. **Point-in-time.** A past verdict must not change when data published later arrives.
   A light that reads the future would validate beautifully and be useless live.
2. **The votes are what the rules say.** Level against the natural reference, trend
   against its own average, abstention when there is nothing honest to say.
3. **The majority is a majority.** Neutral votes count as votes, so a mixed picture stays
   amber instead of being pushed to a colour by the loudest minority.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transform.regime import (
    INSUFFICIENT,
    NEUTRAL,
    RISK_OFF,
    RISK_ON,
    Rule,
    asof,
    drawdown,
    evaluate,
    reading_at,
    regime_frame,
    votes,
)

CAL = pd.bdate_range("2020-01-01", periods=300)


def fred_rows(series_id, rows):
    """``[(ts, ts_release, value)]`` as stored observations."""
    return pd.DataFrame([
        {"source": "fred", "series_id": series_id, "ts": ts, "ts_release": rel, "value": v}
        for ts, rel, v in rows
    ])


def level_rule(**kw):
    base = dict(key="nfci", label="NFCI", kind="level", off_when="above",
                reference=0.0, dead_zone=0.10, series="NFCI")
    base.update(kw)
    return Rule(**base)


def component(values, dates=None):
    idx = CAL[: len(values)] if dates is None else dates
    return pd.DataFrame({"value": values, "value_date": idx}, index=idx)


# --- 1. Point-in-time -------------------------------------------------------------------


def test_a_value_is_invisible_before_its_publication():
    obs = fred_rows("NFCI", [("2020-01-03", "2020-01-08", 0.5)])
    frame = asof(obs, "NFCI", CAL[:10])

    assert pd.isna(frame.loc["2020-01-07", "value"])
    assert frame.loc["2020-01-08", "value"] == 0.5


def test_the_reference_date_travels_with_the_value():
    """So staleness is measured from what the value describes, not when it was fetched."""
    obs = fred_rows("NFCI", [("2020-01-03", "2020-01-08", 0.5)])
    frame = asof(obs, "NFCI", CAL[:10])
    assert frame.loc["2020-01-10", "value_date"] == pd.Timestamp("2020-01-03")


def test_a_revision_of_an_older_period_does_not_move_the_reading():
    """Only the latest reference period known that day is the reading."""
    obs = fred_rows("X", [
        ("2020-01-03", "2020-01-06", 1.0),
        ("2020-01-10", "2020-01-13", 2.0),
        ("2020-01-03", "2020-01-15", 9.0),       # a late revision of the OLD period
    ])
    frame = asof(obs, "X", CAL[:20])
    assert frame.loc["2020-01-16", "value"] == 2.0


def test_a_past_verdict_does_not_change_when_later_data_arrives():
    """Section 9.4 for the whole light: append data released after D, re-read D."""
    rule = level_rule()
    early = fred_rows("NFCI", [("2020-01-03", "2020-01-06", 0.5)])
    later = pd.concat([early, fred_rows("NFCI", [("2020-01-10", "2020-01-20", -0.5)])])

    def verdict_on(obs, day):
        comp = {"nfci": asof(obs, "NFCI", CAL[:30])}
        vt = {"nfci": votes(comp["nfci"], rule, max_staleness_days=60)}
        frame = regime_frame(vt, min_components=1)
        return reading_at(day, [rule], comp, vt, frame)

    before = verdict_on(early, "2020-01-15")
    after = verdict_on(later, "2020-01-15")
    assert before == after
    assert before.verdict == RISK_OFF


# --- 2. Votes ----------------------------------------------------------------------------


def test_a_level_rule_votes_against_its_natural_reference():
    rule = level_rule()
    v = votes(component([0.5, -0.5, 0.05]), rule)["vote"].tolist()
    assert v == [-1.0, 1.0, 0.0], "tighter / looser / within the dead zone"


def test_off_when_below_flips_the_sides():
    """The curve is risk-off when it inverts — below its reference."""
    rule = level_rule(key="curve", off_when="below", series="T10Y2Y")
    v = votes(component([-0.5, 0.5]), rule)["vote"].tolist()
    assert v == [-1.0, 1.0]


def test_a_trend_rule_votes_against_its_own_average():
    rule = Rule(key="credit", label="Crédito", kind="trend", off_when="above", series="BAA10Y")
    values = np.r_[np.full(250, 2.0), np.full(50, 3.0)]      # a spread that widens
    values = values + np.random.default_rng(0).normal(0, 0.01, len(values))

    v = votes(component(values), rule, window=200)["vote"]
    assert v.iloc[-1] == -1.0, "a spread above its average is widening: risk-off"


def test_a_trend_rule_abstains_until_its_window_is_full():
    rule = Rule(key="credit", label="Crédito", kind="trend", off_when="above", series="BAA10Y")
    v = votes(component(np.full(300, 2.0)), rule, window=200)["vote"]
    assert pd.isna(v.iloc[100])


def test_a_stale_component_abstains_instead_of_repeating_its_last_word():
    """A series that stopped updating must not keep voting."""
    rule = level_rule()
    stale = pd.DataFrame(
        {"value": [0.5] * 60, "value_date": [CAL[0]] * 60}, index=CAL[:60]
    )
    v = votes(stale, rule, max_staleness_days=30)["vote"]
    assert v.iloc[5] == -1.0
    assert pd.isna(v.iloc[59])


def test_a_malformed_rule_fails_at_load():
    with pytest.raises(ValueError, match="reference"):
        Rule.from_config({"key": "x", "kind": "level", "off_when": "above", "series": "X"})
    with pytest.raises(ValueError, match="exactly one"):
        Rule.from_config({"key": "x", "kind": "trend", "off_when": "above"})
    with pytest.raises(ValueError, match="off_when"):
        Rule.from_config({"key": "x", "kind": "trend", "off_when": "up", "series": "X"})


# --- 3. The majority ---------------------------------------------------------------------


def grid(*rows):
    """Votes per component for one session each: ``grid([1, -1, 0], ...)``."""
    idx = CAL[: len(rows)]
    cols = list(zip(*rows))
    return {f"c{i}": pd.DataFrame({"vote": list(col)}, index=idx) for i, col in enumerate(cols)}


def test_more_than_half_off_is_red_and_more_than_half_on_is_green():
    frame = regime_frame(grid([-1, -1, -1, 1], [1, 1, 1, -1]), min_components=2)
    assert frame["verdict"].tolist() == [RISK_OFF, RISK_ON]


def test_neutral_votes_count_so_a_loud_minority_does_not_colour_the_light():
    """Three off, five neutral: three is not more than half of eight."""
    frame = regime_frame(grid([-1, -1, -1, 0, 0, 0, 0, 0]), min_components=4)
    assert frame["verdict"].iloc[0] == NEUTRAL


def test_a_tie_is_amber():
    frame = regime_frame(grid([-1, -1, 1, 1]), min_components=2)
    assert frame["verdict"].iloc[0] == NEUTRAL


def test_too_few_components_is_no_verdict_not_a_verdict_from_three():
    frame = regime_frame(grid([-1, -1, -1, np.nan, np.nan, np.nan, np.nan, np.nan]),
                         min_components=4)
    assert frame["verdict"].iloc[0] == INSUFFICIENT


def test_abstentions_do_not_count_as_votes():
    """Five off, three abstaining: five of five is a majority."""
    frame = regime_frame(grid([-1, -1, -1, -1, -1, np.nan, np.nan, np.nan]), min_components=4)
    assert frame["verdict"].iloc[0] == RISK_OFF
    assert frame["available"].iloc[0] == 5


# --- Validation arithmetic -----------------------------------------------------------------


def test_drawdown_is_measured_from_the_running_peak():
    dd = drawdown(pd.Series([100.0, 110.0, 99.0, 121.0]))
    assert dd.tolist() == pytest.approx([0.0, 0.0, -0.1, 0.0])


def test_evaluate_counts_verdicts_in_corrections_and_in_calm():
    days = CAL[:6]
    prices = pd.Series([100, 100, 88, 85, 99, 100], index=days, dtype=float)
    frame = pd.DataFrame({
        "c0": [1, 1, -1, -1, 1, 1], "on": 0, "off": 0, "neutral": 0, "available": 1,
        "verdict": [RISK_ON, RISK_ON, RISK_OFF, NEUTRAL, RISK_ON, RISK_ON],
    }, index=days)

    out = evaluate(frame, prices, correction=0.10, calm=0.05)

    assert out["correction_days"] == 2
    assert out["red_in_correction"] == pytest.approx(0.5)
    assert out["red_in_calm"] == 0.0
    assert out["episodes"][0]["first_red"] == days[2].date().isoformat()
    assert out["per_component"]["c0"]["off_in_correction"] == 1.0
