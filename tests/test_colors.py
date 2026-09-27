"""Green = favourable, red = unfavourable (2026-09-26): one meaning across the panel."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from app import format as fmt  # noqa: E402
from core.config import load_settings  # noqa: E402
from transform import regime as rg  # noqa: E402


def test_tone_follows_the_sign_and_the_good_direction():
    assert fmt.tone(0.05) == fmt.GOOD
    assert fmt.tone(-0.05) == fmt.BAD
    assert fmt.tone(0.05, higher_is_better=False) == fmt.BAD, "dilution up is red"
    assert fmt.tone(0.0) is None and fmt.tone(None) is None and fmt.tone(float("nan")) is None


def test_formatted_cells_are_read_by_their_sign():
    assert fmt._sign_of("-12,1 %") < 0 and fmt._sign_of("−0,4") < 0
    assert fmt._sign_of("+3,2 %") > 0 and fmt._sign_of("119,40 COP") > 0
    assert fmt._sign_of("0,00 COP") == 0
    assert fmt._sign_of("—") is None and fmt._sign_of("no calculable") is None


def test_a_table_is_coloured_by_column_and_direction():
    frame = pd.DataFrame({"PnL": [10.0, -5.0, 0.0], "Dilución": ["2,0 %", "-1,0 %", "—"]})
    styler = fmt.color_by_sign(frame, {"PnL": True, "Dilución": False})
    styler._compute()
    colours = {cell: dict(css).get("color") for cell, css in styler.ctx.items() if css}
    green, red = fmt.TABLE_COLORS[fmt.GOOD], fmt.TABLE_COLORS[fmt.BAD]
    assert colours == {(0, 0): green, (1, 0): red,        # +10 good, −5 bad
                       (0, 1): red, (1, 1): green}        # dilution +2 % bad, −1 % good


def test_rows_can_carry_their_own_direction():
    grid = pd.DataFrame({"3 años": ["10 %", "5 %"]}, index=["Ingresos", "Acciones"])
    html = fmt.color_rows_by_sign(grid, {"Ingresos": True, "Acciones": False}).to_html()
    assert fmt.TABLE_COLORS[fmt.GOOD] in html and fmt.TABLE_COLORS[fmt.BAD] in html


def test_a_tile_states_its_verdict_in_colour_without_an_arrow():
    assert fmt.verdict_delta(fmt.BAD, "por debajo de SPY") == {
        "delta": "por debajo de SPY", "delta_color": "red", "delta_arrow": "off"}
    assert fmt.verdict_delta(None, "x") == {}, "unknown direction: the tile stays neutral"


def test_the_regimes_verdicts_have_their_colour():
    assert fmt.VERDICT_TONES[rg.RISK_ON] == fmt.GOOD
    assert fmt.VERDICT_TONES[rg.RISK_OFF] == fmt.BAD
    assert fmt.VERDICT_TONES[rg.NEUTRAL] == fmt.CAUTION


def test_favourable_directions_in_config_are_up_or_down_and_follow_the_regime():
    readings = load_settings().raw["panel"]["level1"]["readings"]
    directions = {r["series_id"]: r.get("favorable") for r in readings}
    assert set(directions.values()) <= {"up", "down", None}
    assert directions["BAMLH0A0HYM2"] == "down" and directions["VIXCLS"] == "down"
    assert directions["T10Y2Y"] == "up"
    # No colour where the good direction is a matter of opinion.
    assert directions["DGS10"] is None and directions["CPIAUCSL"] is None
