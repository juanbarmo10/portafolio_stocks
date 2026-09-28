"""The growth screen (CLAUDE.md §15.5, point 2), on synthetic filers with known answers.

What is defended: each company is read at its own latest reported quarter (the calendar
Q4 barely exists in ``frames``, section 9.12); dilution uses basic shares and a jump is not
dilution (section 9.15); ratios over non-positive bases are unknown; the market cap applies
the splits after the quarter (section 9.1); and the page's defaults are the user's.
"""

from __future__ import annotations

import datetime as dt
import pathlib

import pandas as pd
import pytest

from core.config import load_settings
from ingest import growth_screen as ig
from transform import growth_screen as gs
from transform import screen as sc

A, B, C, D = "0000000001", "0000000002", "0000000003", "0000000004"


def fact(cik, metric, period, value, end="2026-06-30"):
    return {"source": ig.SOURCE, "series_id": f"{cik}:{metric}:{period}",
            "ts": f"{end}T00:00:00+00:00", "ts_release": "", "value": value}


def price(cik, kind, value, day="2026-09-25"):
    return {"source": ig.PRICE_SOURCE, "series_id": f"{cik}:{kind}",
            "ts": f"{day}T00:00:00+00:00", "ts_release": "", "value": value}


FUNDAMENTALS = pd.DataFrame([
    # A: calendar year. +50 % this quarter after +20 % the one before.
    fact(A, "revenue", "CY2026Q2", 150.0), fact(A, "revenue", "CY2025Q2", 100.0, "2025-06-30"),
    fact(A, "revenue", "CY2026Q1", 120.0, "2026-03-31"),
    fact(A, "revenue", "CY2025Q1", 100.0, "2025-03-31"),
    fact(A, "gross_profit", "CY2026Q2", 90.0), fact(A, "gross_profit", "CY2025Q2", 50.0),
    fact(A, "operating_income", "CY2026Q2", 30.0),
    fact(A, "operating_income", "CY2025Q2", 5.0),
    fact(A, "basic_shares", "CY2026Q2", 110.0), fact(A, "basic_shares", "CY2025Q2", 100.0),
    fact(A, "cash", "CY2026Q2I", 80.0), fact(A, "short_term_investments", "CY2026Q2I", 20.0),
    fact(A, "revenue", "CY2025", 400.0, "2025-12-31"),
    fact(A, "operating_cash_flow", "CY2025", 20.0, "2025-12-31"),
    fact(A, "capex", "CY2025", 60.0, "2025-12-31"),
    fact(A, "sbc", "CY2025", 40.0, "2025-12-31"),
    # B: fiscal year to June — its fiscal Q4 is the calendar Q2, absent from the frame.
    fact(B, "revenue", "CY2026Q1", 60.0, "2026-03-31"),
    fact(B, "revenue", "CY2025Q1", 50.0, "2025-03-31"),
    fact(B, "revenue", "CY2025Q4", 55.0, "2025-12-31"),
    fact(B, "revenue", "CY2024Q4", 50.0, "2024-12-31"),
    # C: an IPO in the year — the share count more than doubles.
    fact(C, "revenue", "CY2026Q2", 10.0), fact(C, "revenue", "CY2025Q2", 8.0),
    fact(C, "basic_shares", "CY2026Q2", 250.0), fact(C, "basic_shares", "CY2025Q2", 100.0),
    # D: shrinking revenue — no incremental margin over a negative change.
    fact(D, "revenue", "CY2026Q2", 80.0), fact(D, "revenue", "CY2025Q2", 100.0),
    fact(D, "operating_income", "CY2026Q2", 10.0),
    fact(D, "operating_income", "CY2025Q2", 20.0),
])
PRICES = pd.DataFrame([
    price(A, "close", 10.0), price(A, "dollar_volume", 500.0),
    price(A, "split", 2.0, "2026-08-01"),           # after the quarter: counts
    price(A, "split", 3.0, "2026-05-01"),           # before it: already in the count
])
REGISTRY = {A: ("AAA", "Alpha"), B: ("BBB", "Beta"), C: ("CCC", "Gamma"), D: ("DDD", "Delta")}


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return gs.growth_table(FUNDAMENTALS, PRICES, REGISTRY, {A: "7372", D: "6022"}).set_index(
        "ticker")


def test_growth_acceleration_and_margins_of_a_calendar_year_filer(table):
    a = table.loc["AAA"]
    assert a["quarter"] == "CY2026Q2" and not a["stale"]
    assert a["revenue_growth"] == pytest.approx(0.5)
    assert a["previous_growth"] == pytest.approx(0.2)
    assert a["acceleration"] == pytest.approx(0.3)
    assert a["gross_margin"] == pytest.approx(0.6)
    assert a["gross_margin_change"] == pytest.approx(0.1)
    assert a["incremental_margin"] == pytest.approx(25.0 / 50.0)
    assert a["dilution"] == pytest.approx(0.1)


def test_cash_figures_are_annual_and_the_rule_of_40_says_so(table):
    a = table.loc["AAA"]
    assert a["fcf_annual"] == pytest.approx(-40.0)
    assert a["fcf_margin"] == pytest.approx(-0.1)
    assert a["sbc_over_revenue"] == pytest.approx(0.1)
    assert a["rule_of_40"] == pytest.approx(0.5 - 0.1), "quarter growth + annual FCF margin"
    assert a["runway_years"] == pytest.approx(100.0 / 40.0), "cash + short-term investments"


def test_market_cap_applies_only_the_splits_after_the_quarter(table):
    a = table.loc["AAA"]
    assert a["split_factor"] == pytest.approx(2.0)
    assert a["market_cap"] == pytest.approx(10.0 * 110.0 * 2.0)
    assert a["price_to_sales"] == pytest.approx(2200.0 / 400.0)
    assert a["dollar_volume"] == pytest.approx(500.0)


def test_a_company_missing_the_newest_quarter_is_read_at_its_own_latest(table):
    """B's fiscal Q4 is the calendar Q2: it is read at Q1 and compared with its Q4."""
    b = table.loc["BBB"]
    assert b["quarter"] == "CY2026Q1" and b["stale"]
    assert b["revenue_growth"] == pytest.approx(0.2)
    assert b["previous_growth"] == pytest.approx(0.1)
    assert b["market_cap"] is None or pd.isna(b["market_cap"]), "no price: unknown, not zero"


def test_a_share_count_jump_is_not_dilution(table):
    c = table.loc["CCC"]
    assert c["share_jump"] and pd.isna(c["dilution"])


def test_no_incremental_margin_over_shrinking_revenue_and_financials_are_flagged(table):
    d = table.loc["DDD"]
    assert pd.isna(d["incremental_margin"])
    assert d["revenue_growth"] == pytest.approx(-0.2)
    assert d["financial"] is True and table.loc["AAA", "financial"] is False


def test_filters_count_the_unknown_apart_and_financials_aside(table):
    limits = {"min_market_cap": 1000.0, "min_dollar_volume": 100.0}
    result = sc.apply_filters(table.reset_index(), limits, filters=gs.FILTERS,
                              exclude_financials=True)
    assert list(result.table["ticker"]) == ["AAA"]
    assert result.excluded_financials == 1
    assert result.unknown["market_cap"] == 2, "B and C have no price: unknown, not failed"


def test_the_defaults_are_the_ones_the_user_chose():
    defaults = load_settings().raw["screen"]["growth"]["defaults"]
    assert defaults["min_market_cap"] == 300_000_000
    assert defaults["min_dollar_volume"] == 1_000_000
    assert defaults["exclude_financials"] is True


# --- Ingest helpers -----------------------------------------------------------------------


@pytest.mark.parametrize("today, expected", [
    ("2026-09-28", (2026, 2)), ("2026-08-18", (2026, 1)), ("2026-08-19", (2026, 2)),
    ("2026-02-15", (2025, 3)), ("2026-05-20", (2026, 1)),
])
def test_a_quarter_enters_once_its_10q_are_due(today, expected):
    assert ig.latest_quarter(dt.date.fromisoformat(today), 50) == expected


def test_the_periods_are_three_quarters_and_the_same_three_a_year_before():
    assert ig.quarter_periods(2026, 1) == (["CY2026Q1", "CY2025Q4", "CY2025Q3"],
                                           ["CY2025Q1", "CY2024Q4", "CY2024Q3"])


def test_the_price_summary_is_the_last_close_the_median_dollar_volume_and_the_splits():
    days = pd.bdate_range("2026-06-01", periods=5)
    history = pd.DataFrame({"Close": [10.0, 11.0, 12.0, None, 13.0],
                            "Volume": [100, 200, 300, 400, 500],
                            "Stock Splits": [0, 0, 2.0, 0, 0]}, index=days)
    summary = ig.price_summary(history, sessions=3)
    assert summary["close"] == 13.0 and summary["day"] == days[-1].date().isoformat()
    assert summary["dollar_volume"] == pytest.approx(3600.0), "median of 2200, 3600, 6500"
    assert summary["splits"] == {days[2].date().isoformat(): 2.0}
    assert ig.price_summary(pd.DataFrame({"Close": [None]}), 3) is None


def test_the_registry_takes_each_companys_first_listing_and_leaves_researched_alone():
    raw = {"0": {"cik_str": 1, "ticker": "aaa", "title": "Alpha"},
           "1": {"cik_str": 1, "ticker": "AAA-WT", "title": "Alpha"},
           "2": {"cik_str": 2, "ticker": "BBB", "title": "Beta"}}
    registry = ig.first_tickers(raw)
    assert registry[A] == ("AAA", "Alpha")
    rows = ig.registry_rows(registry, {A: "AAA", B: "BBB"}, {A: ["7372", "Software", "x"]},
                            {B}, "2026-09-28")
    assert [r["cik"] for r in rows] == [A] and rows[0]["sic"] == "7372"


# --- The page -------------------------------------------------------------------------------

st = pytest.importorskip("streamlit", reason="the 'app' extra is not installed")


def test_the_growth_mode_starts_at_the_users_defaults(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app import data as app_data
    from db import loader

    db_path = tmp_path / "growth.db"
    conn = loader.init_db(db_path)
    loader.upsert_observations(conn, pd.concat([FUNDAMENTALS, PRICES]).assign(
        ingested_at="2026-09-28T00:00:00+00:00"))
    loader.upsert_companies(conn, [
        {"cik": c, "ticker": t, "name": n, "sector": None, "thesis_category": None,
         "first_seen": "2026-09-28", "status": "active", "sic": None, "sic_description": None}
        for c, (t, n) in REGISTRY.items()])
    conn.close()
    monkeypatch.setattr(app_data, "db_path", lambda: db_path)
    st.cache_data.clear()
    root = pathlib.Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app" / "main.py"), default_timeout=120).run()
    app.switch_page(str(root / "app" / "pages" / "screen.py"))
    app.run()
    app.radio(key="screen_mode").set_value("Crecimiento · todas las empresas de la SEC").run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.number_input(key="g_min_market_cap").value == pytest.approx(300.0)
    assert app.number_input(key="g_min_dollar_volume").value == pytest.approx(1.0)
    assert app.checkbox(key="g_exclude_financials").value is True
    # AAA's cap is 2.200 USD: under 300 M it fails, so nothing passes with the defaults…
    assert any("**0** pasan" in m.value for m in app.markdown)
    # …and switching the cap filter off lets it through: every default is editable.
    app.checkbox(key="g_on_min_market_cap").uncheck().run()
    app.checkbox(key="g_on_min_dollar_volume").uncheck().run()
    shown = next(pd.DataFrame(d.value) for d in app.dataframe
                 if "Regla del 40" in pd.DataFrame(d.value).columns)
    assert "AAA" in list(shown["Ticker"])
    st.cache_data.clear()


# --- One tag per company (found on the first real run, 2026-09-28) --------------------------

from ingest.screen import resolve_tags  # noqa: E402


def rows(end="2026-06-30", **values):
    return {period: {"value": v, "series_id": period, "ts": end} for period, v in values.items()}


def test_a_part_never_stands_in_for_the_total():
    """NVE Corp: ``Revenues`` is the total (11,0 M), ``RevenueFromContract…`` one line of it
    (0,3 M), and the contract tag comes first by preference."""
    out = resolve_tags({0: rows(CY2026Q2=299_322, CY2025Q2=196_074),
                        2: rows(CY2026Q2=11_034_057, CY2025Q2=6_104_644)}, largest=True)
    assert out["CY2026Q2"]["value"] == 11_034_057 and out["CY2025Q2"]["value"] == 6_104_644


def test_tags_are_not_mixed_across_periods():
    """Valaris: the part exists in the year and Q1, the total everywhere. First-tag-per-
    period gave 11,9 M of annual revenue and a 1.700 % FCF margin."""
    out = resolve_tags({0: rows(CY2026Q1=7.5e6, CY2025=11.9e6),
                        2: rows(CY2026Q2=539.2e6, CY2026Q1=465.4e6, CY2025=2369e6)},
                       largest=True)
    assert out["CY2025"]["value"] == 2369e6 and out["CY2026Q1"]["value"] == 465.4e6


def test_a_company_that_changed_tags_keeps_every_period():
    out = resolve_tags({0: rows(CY2026Q2=120.0), 2: rows(CY2025Q2=100.0)}, largest=True)
    assert set(out) == {"CY2026Q2", "CY2025Q2"}


def test_without_largest_the_preference_order_rules_as_before():
    """Cash: the first tag excludes restricted cash, the second includes it — the smaller
    one is the wanted one, so the total-versus-part rule is for revenue only."""
    out = resolve_tags({0: rows(CY2026Q2I=80.0), 1: rows(CY2026Q2I=95.0, CY2026Q1I=90.0)},
                       largest=False)
    assert out["CY2026Q2I"]["value"] == 80.0 and out["CY2026Q1I"]["value"] == 90.0


def test_two_share_classes_take_the_diluted_count_for_the_size_only():
    extra = pd.DataFrame([fact(B, "diluted_shares", "CY2026Q1", 40.0, "2026-03-31")])
    table = gs.growth_table(pd.concat([FUNDAMENTALS, extra]),
                            pd.concat([PRICES, pd.DataFrame([price(B, "close", 5.0)])]),
                            REGISTRY).set_index("ticker")
    assert table.loc["BBB", "market_cap"] == pytest.approx(200.0)
    assert pd.isna(table.loc["BBB", "dilution"]), "dilution stays on basic shares"


def test_two_tags_over_different_periods_are_not_a_total_and_its_part():
    """Amcor: ``Revenues`` for the year to June 2024, ``RevenueFromContract…`` for a year
    to September 2024 and then from 2025 on. Not comparable, so the switch fills the year."""
    out = resolve_tags({0: {"CY2024": {"value": 10.07e9, "ts": "2024-09-28"},
                            "CY2025": {"value": 15.01e9, "ts": "2025-06-30"}},
                        2: {"CY2024": {"value": 13.64e9, "ts": "2024-06-30"}}}, largest=True)
    assert out["CY2025"]["value"] == 15.01e9


def test_a_share_count_in_the_wrong_scale_leaves_the_size_unknown():
    """Iovance files 450.189 shares for ~450 M: its "market cap" of 5 M would trade 10 times
    over every day. Unknown and flagged, never rescaled."""
    extra = pd.DataFrame([price(C, "close", 11.0), price(C, "dollar_volume", 5e7)])
    table = gs.growth_table(FUNDAMENTALS, pd.concat([PRICES, extra]), REGISTRY).set_index(
        "ticker")
    assert table.loc["CCC", "cap_suspect"] and pd.isna(table.loc["CCC", "market_cap"])
    assert not table.loc["AAA", "cap_suspect"]


def test_a_wrong_basic_count_falls_back_to_a_plausible_diluted_one():
    """Amcor: 1,5 M basic shares filed against 463,8 M diluted."""
    extra_f = pd.DataFrame([fact(C, "diluted_shares", "CY2026Q2", 1e6)])
    extra_p = pd.DataFrame([price(C, "close", 11.0), price(C, "dollar_volume", 5e6)])
    table = gs.growth_table(pd.concat([FUNDAMENTALS, extra_f]),
                            pd.concat([PRICES, extra_p]), REGISTRY).set_index("ticker")
    assert table.loc["CCC", "market_cap"] == pytest.approx(11.0 * 1e6)
    assert not table.loc["CCC", "cap_suspect"]
