"""Market breadth with its coverage attached (section 9.5, THEORY.md §2.1).

Synthetic prices throughout, and deliberately so: what is tested is arithmetic and
boundaries — who counts as a member on which day, what "covered" means, and whether a
split can move a reading — and each case needs prices shaped to isolate one of them.

The invariant that matters most is the section 9.1 one, restated for breadth: **a split
that happens later must not change an earlier reading.**
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from transform.breadth import (
    advance_decline,
    net_new_highs,
    BreadthReading,
    breadth_series,
    defensive_rotation,
    equal_weight_ratio,
    sector_breadth,
    membership_mask,
    splits_by_ticker,
    usable_from,
    wide_closes,
)

DAYS = pd.bdate_range("2020-01-01", periods=260)


def always(*tickers):
    return [{"ticker": t, "start_date": "2019-01-01", "end_date": None} for t in tickers]


def rising(start=100.0):
    return pd.Series(np.linspace(start, start * 1.5, len(DAYS)), index=DAYS)


def falling(start=100.0):
    return pd.Series(np.linspace(start, start * 0.6, len(DAYS)), index=DAYS)


def last(readings):
    return readings[-1]


# --- The reading itself -----------------------------------------------------------


def test_breadth_is_the_share_above_their_own_average():
    closes = pd.DataFrame({"UP1": rising(), "UP2": rising(50), "DOWN": falling()})
    reading = last(breadth_series(always("UP1", "UP2", "DOWN"), closes, window=200))

    assert reading.computable == 3
    assert reading.above == 2
    assert reading.breadth == pytest.approx(2 / 3)


def test_before_a_full_window_nothing_is_computable():
    closes = pd.DataFrame({"UP1": rising()})
    early = breadth_series(always("UP1"), closes, window=200)[10]
    assert early.priced == 1 and early.computable == 0
    assert early.breadth is None, "no average yet is a hole, not zero breadth"


# --- Coverage: the survivorship gap, made visible ---------------------------------


def test_a_member_with_no_prices_lowers_coverage_not_breadth():
    """The priced subset decides the reading; the missing member is reported, not guessed."""
    closes = pd.DataFrame({"UP1": rising(), "UP2": rising()})
    reading = last(breadth_series(always("UP1", "UP2", "GONE"), closes, window=200))

    assert reading.members == 3
    assert reading.priced == 2
    assert reading.coverage == pytest.approx(2 / 3)
    assert reading.breadth == 1.0


def test_a_reading_below_the_threshold_is_flagged():
    closes = pd.DataFrame({"UP1": rising(), "UP2": rising()})
    reading = last(breadth_series(always("UP1", "UP2", "GONE"), closes,
                                  window=200, threshold=0.85))
    assert reading.meets_threshold is False

    whole = last(breadth_series(always("UP1", "UP2"), closes, window=200, threshold=0.85))
    assert whole.meets_threshold is True


def test_usable_from_is_where_coverage_stays_above_not_where_it_first_touches():
    """The honest start of the series: after it never drops back below."""
    from transform.breadth import BreadthReading

    def r(day, ok):
        return BreadthReading(day, 10, 9 if ok else 7, 9, 5, 0.5, 0.9 if ok else 0.7, ok)

    readings = [r("2019-01-01", True), r("2019-06-01", False), r("2019-10-01", True),
                r("2020-01-01", True)]
    assert usable_from(readings) == "2019-10-01"
    assert usable_from([r("2019-01-01", False)]) is None


# --- Membership boundaries ----------------------------------------------------------


def test_a_company_is_not_counted_before_it_joins_the_index():
    """Point-in-time for the list itself: membership starts on its start date."""
    intervals = [*always("OLD"),
                 {"ticker": "NEW", "start_date": DAYS[240].date().isoformat(),
                  "end_date": None}]
    closes = pd.DataFrame({"OLD": rising(), "NEW": rising()})
    readings = {r.date: r for r in breadth_series(intervals, closes, window=200)}

    assert readings[DAYS[239].date().isoformat()].members == 1
    assert readings[DAYS[240].date().isoformat()].members == 2


def test_the_end_date_is_exclusive():
    intervals = [{"ticker": "X", "start_date": "2019-01-01",
                  "end_date": DAYS[5].date().isoformat()}]
    mask = membership_mask(intervals, DAYS, ["X"])
    assert bool(mask.loc[DAYS[4], "X"]) is True
    assert bool(mask.loc[DAYS[5], "X"]) is False


# --- Splits: the section 9.1 invariant for breadth -----------------------------------


def split_prices(split_day: int, ratio: float = 2.0) -> pd.Series:
    """A steadily rising company whose RAW close drops by the ratio on its split day."""
    adjusted = rising()
    raw = adjusted.copy()
    raw.iloc[:split_day] = raw.iloc[:split_day] * ratio      # pre-split raw is higher
    return raw


def test_a_later_split_does_not_change_an_earlier_reading():
    """The invariant that makes today's split adjustment safe for past dates.

    A split after date D multiplies every bar in D's window by the same constant, and
    close / average is invariant to a constant. So knowing about the split must not move
    any reading before it — if it did, the breadth history would be rewritten by every
    future corporate action, which is the section 9.1 trap.
    """
    split_day = 250
    closes = pd.DataFrame({"X": split_prices(split_day), "Y": rising()})
    ex_date = DAYS[split_day].date()

    unaware = breadth_series(always("X", "Y"), closes, {}, window=200)
    aware = breadth_series(always("X", "Y"), closes, {"X": {ex_date: 2.0}}, window=200)

    before = [(a.above, a.computable) for a in aware[:split_day]]
    assert before == [(u.above, u.computable) for u in unaware[:split_day]]


def test_a_split_inside_the_window_is_corrected():
    """Without the adjustment a rising company falls "below its average" for 200 days."""
    split_day = 230
    closes = pd.DataFrame({"X": split_prices(split_day)})
    ex_date = DAYS[split_day].date()

    unaware = last(breadth_series(always("X"), closes, {}, window=200))
    aware = last(breadth_series(always("X"), closes, {"X": {ex_date: 2.0}}, window=200))

    assert unaware.above == 0, "the raw drop should have fooled the unadjusted reading"
    assert aware.above == 1, "the split-adjusted reading must see the real uptrend"


# --- Plumbing ----------------------------------------------------------------------------


def test_long_observations_become_a_session_by_ticker_frame():
    obs = pd.DataFrame([
        {"source": "yfinance", "series_id": "A:close_raw", "ts": "2020-01-02T00:00:00+00:00",
         "ts_release": "2020-01-02", "value": 1.0},
        {"source": "yfinance", "series_id": "B:close_raw", "ts": "2020-01-02T00:00:00+00:00",
         "ts_release": "2020-01-02", "value": 2.0},
        {"source": "yfinance", "series_id": "A:open_raw", "ts": "2020-01-02T00:00:00+00:00",
         "ts_release": "2020-01-02", "value": 9.0},
    ])
    wide = wide_closes(obs)
    assert list(wide.columns) == ["A", "B"]
    assert wide.loc[pd.Timestamp("2020-01-02"), "A"] == 1.0


def test_splits_are_read_from_one_provider_only():
    actions = [
        {"ticker": "X", "kind": "split", "ex_date": "2020-08-31T00:00:00+00:00",
         "ratio": 4.0, "source": "yfinance"},
        {"ticker": "X", "kind": "split", "ex_date": "2020-08-31", "ratio": 4.0,
         "source": "ibkr"},
        {"ticker": "X", "kind": "dividend", "ex_date": "2020-08-07", "ratio": None,
         "source": "yfinance"},
    ]
    assert splits_by_ticker(actions) == {"X": {dt.date(2020, 8, 31): 4.0}}


def test_no_prices_is_no_readings_not_an_error():
    assert breadth_series(always("A"), pd.DataFrame()) == []


# --- Holes in the source are not survivorship -------------------------------------


def test_a_session_the_source_returned_empty_is_a_hole_not_low_coverage():
    """The real case: Yahoo served 2026-09-22 for a few tickers and not the rest.

    Read as coverage it looked like "12 %" and split the usable series in two. It says
    nothing about the index, so its breadth is a hole, not a figure over whatever subset
    the source happened to return.
    """
    tickers = [f"T{i}" for i in range(20)]
    closes = pd.DataFrame({t: rising() for t in tickers})
    closes.iloc[230, 2:] = float("nan")          # one session, most tickers missing

    readings = breadth_series(always(*tickers), closes, window=200)
    hole = readings[230]

    assert hole.source_hole is True
    assert hole.breadth is None
    assert hole.meets_threshold is False
    assert readings[229].source_hole is False and readings[231].source_hole is False


def test_a_hole_does_not_break_the_usable_run():
    def r(day, ok, hole=False):
        return BreadthReading(day, 10, 9, 9, 5, None if hole else 0.5, 0.9, ok, hole)

    readings = [r("2019-10-01", True), r("2019-10-02", False, hole=True),
                r("2019-10-03", True)]
    assert usable_from(readings) == "2019-10-01"


def test_survivorship_still_breaks_it():
    """Only source holes are skipped; a genuine coverage drop still resets the start."""
    def r(day, ok):
        return BreadthReading(day, 10, 9 if ok else 7, 9, 5, 0.5, 0.9 if ok else 0.7, ok)

    assert usable_from([r("2019-10-01", True), r("2019-10-02", False),
                        r("2019-10-03", True)]) == "2019-10-03"


# --- One missing day must not poison the next 200 -----------------------------------


def test_one_missing_day_does_not_disable_the_average_for_200_sessions():
    """Found on the first real run, and silent: with a strict 200-of-200 rule, the source
    hole of 2026-09-22 disabled the average of every affected ticker for the next 200
    sessions. The breadth reported for 2026-09-23 rested on 60 of 503 members."""
    tickers = [f"T{i}" for i in range(20)]
    closes = pd.DataFrame({t: rising() for t in tickers})
    closes.iloc[230, 2:] = float("nan")

    readings = breadth_series(always(*tickers), closes, window=200)

    assert readings[231].computable == readings[229].computable == 20


def test_coverage_is_what_the_reading_rests_on_not_who_has_a_price():
    """Priced-but-not-computable members must lower coverage.

    Defined as priced / members, coverage read 100 % on the day breadth rested on 60 of
    503 — and the reading passed the threshold unflagged. Defined as computable / members,
    the same fault is caught.
    """
    old = [f"O{i}" for i in range(10)]
    young = [f"Y{i}" for i in range(10)]
    closes = pd.DataFrame({t: rising() for t in old})
    for t in young:                                   # listed 30 sessions ago
        series = rising()
        series.iloc[:-30] = float("nan")
        closes[t] = series

    reading = last(breadth_series(always(*old, *young), closes, window=200, threshold=0.85))

    assert reading.priced == 20
    assert reading.computable == 10
    assert reading.coverage == pytest.approx(0.5)
    assert reading.meets_threshold is False


# --- ETF-built measures -------------------------------------------------------------


SECTORS = ["S1", "S2", "S3"]


def test_sector_breadth_counts_sectors_above_their_average():
    closes = pd.DataFrame({"S1": rising(), "S2": rising(), "S3": falling()})
    out = sector_breadth(closes, {}, SECTORS, window=200)
    assert out["share"].iloc[-1] == pytest.approx(2 / 3)


def test_sector_breadth_is_undefined_until_every_sector_is_computable():
    """A fixed denominator: a reading over two of three sectors is a different measure."""
    closes = pd.DataFrame({"S1": rising(), "S2": rising()})
    young = rising()
    young.iloc[:-30] = float("nan")
    closes["S3"] = young

    out = sector_breadth(closes, {}, SECTORS, window=200)
    assert out["share"].iloc[-1] is None


def test_equal_weight_ratio_is_equal_over_cap():
    closes = pd.DataFrame({"RSP": rising(50), "SPY": rising(100)})
    out = equal_weight_ratio(closes, {})
    assert out["ratio"].iloc[0] == pytest.approx(0.5)


def test_rotation_rises_when_defensives_beat_cyclicals():
    closes = pd.DataFrame({"XLU": rising(), "XLP": rising(), "XLK": falling(), "XLY": falling()})
    out = defensive_rotation(closes, {})
    assert out["rotation"].iloc[0] == pytest.approx(100.0)
    assert out["rotation"].iloc[-1] > 100.0


def test_rotation_does_not_depend_on_any_etfs_share_price():
    """Why the literal (XLU + XLP) / (XLK + XLY) of section 8 was not used.

    A sum of share prices weights each ETF by its per-share price, which is arbitrary —
    XLK trades at about three times XLP. With geometric means, multiplying one ETF's
    price by any constant leaves the rebased series unchanged.
    """
    closes = pd.DataFrame({"XLU": rising(), "XLP": rising(80), "XLK": falling(),
                           "XLY": rising(120)})
    scaled = closes.assign(XLK=closes["XLK"] * 10)

    pd.testing.assert_series_equal(
        defensive_rotation(closes, {})["rotation"],
        defensive_rotation(scaled, {})["rotation"],
    )

    def literal(frame):
        raw = (frame["XLU"] + frame["XLP"]) / (frame["XLK"] + frame["XLY"])
        return raw / raw.iloc[0]

    assert not np.allclose(literal(closes), literal(scaled)), (
        "the literal formula should have moved — otherwise this test proves nothing"
    )


def test_etf_measures_are_empty_without_their_tickers():
    empty = pd.DataFrame({"X": rising()})
    assert sector_breadth(empty, {}, SECTORS).empty
    assert equal_weight_ratio(empty, {}).empty
    assert defensive_rotation(empty, {}).empty


# --- Advance-decline and new highs / lows (2026-09-25) ---------------------------------------


def test_advancers_decliners_and_the_line():
    closes = pd.DataFrame({"UP1": rising(), "UP2": rising(), "DOWN": falling()}, index=DAYS)
    readings = breadth_series(always("UP1", "UP2", "DOWN"), closes, window=200)
    assert (last(readings).advancers, last(readings).decliners) == (2, 1)
    line = advance_decline(readings)
    assert line["net"].iloc[-1] == 1
    assert line["line"].iloc[-1] == line["net"].sum(), "the line is the running sum"


def test_a_year_of_history_is_needed_for_a_new_high():
    closes = pd.DataFrame({"UP": rising(), "DOWN": falling()}, index=DAYS)
    readings = breadth_series(always("UP", "DOWN"), closes, window=200)
    assert readings[100].new_highs == 0, "a stock with 100 sessions has no 52-week high"
    assert (last(readings).new_highs, last(readings).new_lows) == (1, 1)
    net = net_new_highs(readings)
    assert net["net_share"].iloc[-1] == pytest.approx(0.0)


def test_a_source_hole_adds_nothing_to_the_line():
    closes = pd.DataFrame({t: rising() for t in [f"T{i}" for i in range(30)]}, index=DAYS)
    closes.iloc[-5, :25] = np.nan                     # the source returns 5 of 30 that day
    readings = breadth_series(always(*closes.columns), closes, window=200)
    hole = readings[-5]
    assert hole.source_hole and hole.advancers is None
    assert readings[-5].date not in set(advance_decline(readings)["date"])
