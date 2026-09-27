"""Chart axes (2026-09-26): months as numbers with the year under them, amounts in words.

The label expressions run in the browser (Vega), so what is tested here is the choice each
chart gets — which expression, how many ticks, which unit — and that every page still
renders with it (``tests/test_app.py`` serializes each chart through the same wrapper).
"""

from __future__ import annotations

import pandas as pd
import pytest

alt = pytest.importorskip("altair")

from app import format as fmt  # noqa: E402


def test_up_to_a_year_ticks_are_monthly_with_the_year_under_january():
    axis = fmt.time_axis("2025-09-26", "2026-09-26")
    assert axis["labelExpr"] == fmt._MONTHS_EXPR
    assert axis["tickCount"] == {"interval": "month", "step": 1}


def test_a_few_years_tick_by_quarter_then_by_half_year():
    assert fmt.time_axis("2022-01-01", "2026-01-01")["tickCount"] == {"interval": "month",
                                                                      "step": 3}
    seven = fmt.time_axis("2019-12-09", "2026-09-25")
    assert seven["labelExpr"] == fmt._MONTHS_EXPR
    assert seven["tickCount"] == {"interval": "month", "step": 6}, \
        "an explicit interval: Vega never picks half-years from a count"


def test_long_histories_show_only_years_and_thin_them_out():
    decade = fmt.time_axis("2014-01-01", "2026-01-01")
    assert decade["labelExpr"] == fmt._YEARS_EXPR
    assert decade["tickCount"] == {"interval": "year", "step": 1}
    long = fmt.time_axis("1999-01-01", "2026-09-26")
    assert long["tickCount"] == {"interval": "year", "step": 2}


def test_a_zoomable_chart_gets_a_count_so_zooming_in_brings_finer_ticks():
    assert fmt.time_axis("2019-12-09", "2026-09-25", zoomable=True)["tickCount"] == 20
    chart = alt.Chart(frame("2020-01-01", "2021-01-01", 1.0)).mark_line().encode(
        x="date:T", y="value:Q")
    assert fmt._zoomable(chart.interactive(bind_y=False))
    assert fmt._zoomable(alt.layer(chart.interactive(), chart))
    assert not fmt._zoomable(chart)


def test_the_year_goes_under_the_month_on_january_and_on_the_first_tick():
    expr = fmt._MONTHS_EXPR
    assert "month(datum.value) == 0 || datum.index == 0" in expr
    assert "[timeFormat(datum.value, '%m'), timeFormat(datum.value, '%Y')]" in expr


def test_amounts_use_one_scale_word_per_chart():
    assert "/ 1000000000," in fmt.amount_axis(2.8e11)["labelExpr"]
    assert "' mil M'" in fmt.amount_axis(2.8e11)["labelExpr"]
    assert "' M'" in fmt.amount_axis(4.5e7)["labelExpr"]
    assert "' bill.'" in fmt.amount_axis(5.4e12)["labelExpr"]
    assert fmt.amount_axis(950.0)["labelExpr"] == fmt.PERCENT_SPACE_EXPR


def frame(start, end, value):
    days = pd.date_range(start, end, freq="MS")
    return pd.DataFrame({"date": days, "value": [value] * len(days)})


def test_ranges_are_read_from_every_layer_and_skip_an_axisless_y():
    a = alt.Chart(frame("2020-01-01", "2021-01-01", 5.0)).mark_line().encode(
        x="date:T", y="value:Q")
    b = alt.Chart(frame("2019-06-01", "2020-06-01", 9e9)).mark_line().encode(
        x=alt.X("date:T"), y=alt.Y("value:Q", axis=None))
    start, end, largest = fmt.axis_ranges(alt.layer(a, b))
    assert start == pd.Timestamp("2019-06-01") and end == pd.Timestamp("2021-01-01")
    assert largest == 5.0, "the hidden axis does not choose the unit"


def test_scale_words_only_when_the_chart_asks_for_them():
    """The Fed publishes its balance sheet in millions: scaling it automatically would
    read "6,6 M" for 6,6 billones."""
    chart = alt.Chart(frame("2020-01-01", "2026-01-01", 6.6e6)).mark_line().encode(
        x="date:T", y="value:Q")
    plain = fmt.configured(chart).to_dict()["config"]
    assert plain["axisY"]["labelExpr"] == fmt.PERCENT_SPACE_EXPR
    asked = fmt.configured(chart, amounts=True).to_dict()["config"]
    assert "' M'" in asked["axisY"]["labelExpr"]
    assert plain["axisTemporal"]["labelExpr"] == fmt._MONTHS_EXPR
    assert plain["axisY"]["titleAngle"] == 0, "the y title reads horizontally"
    assert plain["locale"]["number"]["decimal"] == ","
