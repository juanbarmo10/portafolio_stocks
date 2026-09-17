"""The public-mode rule: a deployment anyone can open shows no absolute amount.

Two layers are tested, because either alone is insufficient:

1. ``app/format.py`` refuses to format an amount publicly. That covers the widgets that
   ask for one.
2. The whole app is rendered in public mode and every rendered string is scanned for a
   currency-shaped figure. That covers the widgets that never ask — the ones that build a
   string themselves and would slip past layer 1.

**Known limit, stated rather than papered over:** the scanner recognises a *currency
mark* next to digits. A raw amount charted with no currency label — a NAV line in dollars
— would pass it. The defence there is at the data layer instead: a portfolio series goes
through ``rebase_100`` before it reaches a public chart, and ``test_rebase_100_is_scale_
invariant`` is what makes that transformation safe to publish.

Skipped when the ``app`` extra is not installed; CI installs ``[dev,ibkr]`` only.
"""

from __future__ import annotations

import pathlib
import re

import pandas as pd
import pytest

from app.format import (
    MISSING,
    PUBLIC_PAGES,
    PublicModeViolation,
    drop_absolute_columns,
    money,
    pct,
    rebase_100,
    share,
)

# --- Layer 1: the formatter ---------------------------------------------------


def test_money_refuses_to_format_publicly():
    """The refusal is the mechanism. A quieter amount is still a published amount."""
    with pytest.raises(PublicModeViolation):
        money(1138.9675, public=True)


def test_money_formats_in_spanish_notation_privately():
    """Privately the operator sees the real figure, in the notation the panel uses."""
    assert money(1138.9675, public=False) == "1.138,97 USD"
    assert money(-9.37078721, public=False) == "-9,37 USD"


def test_a_missing_amount_is_a_hole_not_a_zero():
    """Section 12: a value nobody has is never rendered as 0."""
    assert money(None, public=False) == MISSING
    assert money(float("nan"), public=False) == MISSING
    assert pct(None) == MISSING


def test_percentages_are_safe_in_both_modes():
    """A ratio carries the reading without carrying the size, so it needs no gate."""
    assert pct(0.0827) == "8,3 %"
    assert pct(0.0082, decimals=2) == "0,82 %"


def test_share_returns_none_instead_of_inventing_a_weight():
    assert share(2.0, 8.0) == 0.25
    assert share(2.0, 0) is None
    assert share(None, 8.0) is None


def test_rebase_100_is_scale_invariant():
    """The argument for publishing the curve at all, as an executable property.

    A secret factor is recoverable from one leaked anchor; a rebased index has no anchor.
    If this property ever breaks, the public curve starts carrying size information.
    """
    nav = pd.Series([524.09, 812.44, 1138.97])
    baseline = rebase_100(nav)

    for factor in (3.0, 0.1, 1000.0):
        pd.testing.assert_series_equal(rebase_100(nav * factor), baseline)

    assert baseline.iloc[0] == 100.0


def test_rebase_100_survives_holes_and_a_useless_base():
    """Leading gaps and zeros must not turn into a division by zero or a fake start."""
    rebased = rebase_100(pd.Series([None, 0.0, 50.0, 75.0]))
    assert rebased.iloc[2] == 100.0
    assert rebased.iloc[3] == 150.0
    assert pd.isna(rebased.iloc[0])

    assert rebase_100(pd.Series([0.0, 0.0])).isna().all()


def test_drop_absolute_columns_only_drops_publicly():
    frame = pd.DataFrame({"Posición": ["TMUS"], "Valor USD": [180.47], "Peso": [0.56]})

    assert list(drop_absolute_columns(frame, ["Valor USD"], public=False).columns) == list(
        frame.columns
    )
    public = drop_absolute_columns(frame, ["Valor USD"], public=True)
    assert list(public.columns) == ["Posición", "Peso"], "the amount column survived"


def test_the_fiscal_page_can_never_be_allowed_publicly():
    """Section 11: the fiscal layer is local. Its configuration alone names a country."""
    assert "Fiscal" not in PUBLIC_PAGES


# --- Layer 2: the whole rendered app ------------------------------------------

pytest.importorskip("streamlit", reason="the 'app' extra is not installed")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import data as app_data  # noqa: E402
from core import config  # noqa: E402
from db import loader  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = str(REPO_ROOT / "app" / "main.py")
PORTFOLIO = str(REPO_ROOT / "app" / "pages" / "portfolio.py")

# A figure with a currency mark glued to it, in either notation.
CURRENCY_FIGURE = re.compile(r"(?:US\$|\$|€)\s*-?[\d.,]*\d|-?[\d.,]*\d\s*(?:USD|EUR|COP)\b")

# Element kinds that render text a reader can see. Chart specs are handled separately.
TEXT_ELEMENTS = (
    "title", "header", "subheader", "markdown", "caption", "text",
    "info", "warning", "error", "success", "code", "metric",
)


def _rendered_text(app: AppTest) -> str:
    """Every visible string of the current render, tables included."""
    chunks: list[str] = []
    for kind in TEXT_ELEMENTS:
        for element in app.get(kind):
            for attribute in ("label", "value", "delta", "body"):
                piece = getattr(element, attribute, None)
                if isinstance(piece, str):
                    chunks.append(piece)
    for kind in ("dataframe", "table"):
        for element in app.get(kind):
            frame = element.value
            if frame is not None:
                chunks.append(pd.DataFrame(frame).to_string())
    return "\n".join(chunks)


@pytest.fixture()
def public_app(tmp_path, monkeypatch):
    """A public deployment pointed at a throwaway database with macro rows in it."""
    db_path = tmp_path / "public.db"
    conn = loader.init_db(db_path)
    try:
        loader.upsert_observations(conn, pd.DataFrame([
            {"source": "fred", "series_id": "T10Y2Y", "ts": "2026-08-27",
             "ts_release": "2026-08-28", "value": 0.39},
            {"source": "fred", "series_id": "VIXCLS", "ts": "2026-08-27",
             "ts_release": "2026-08-27", "value": 15.2},
        ]))
    finally:
        conn.close()

    monkeypatch.setattr(app_data, "db_path", lambda: db_path)
    monkeypatch.setenv("PUBLIC_MODE", "1")
    config.load_settings.cache_clear()
    st.cache_data.clear()
    yield db_path
    st.cache_data.clear()


def test_a_public_render_carries_no_currency_figure(public_app):
    """The backstop: walk every public page and fail on anything shaped like money."""
    app = AppTest.from_file(MAIN, default_timeout=60).run()
    assert not app.exception, [e.value for e in app.exception]

    pages = [MAIN, PORTFOLIO]
    for page in pages:
        if page != MAIN:
            app.switch_page(page)
            app.run()
            assert not app.exception, [e.value for e in app.exception]

        text = _rendered_text(app)
        # Without this the scan can quietly become vacuous: an accessor that stops
        # returning elements would turn the guard into an assertion about nothing.
        assert len(text) > 200, f"{page} rendered almost nothing — the scan proves nothing"

        leaks = CURRENCY_FIGURE.findall(text)
        assert not leaks, f"{page} publishes an absolute amount: {leaks}"


def test_the_public_portfolio_page_states_the_rule_and_drops_the_operator_notes(public_app):
    """Publicly the page explains what it shows; the unblocking steps are for the operator."""
    app = AppTest.from_file(MAIN, default_timeout=60).run()
    app.switch_page(PORTFOLIO)
    app.run()
    assert not app.exception, [e.value for e in app.exception]

    text = _rendered_text(app)
    assert "base 100" in text, "the public view does not say its figures are relative"
    assert "IBKR_FLEX_TOKEN" not in text, "operator instructions leaked into the public view"
