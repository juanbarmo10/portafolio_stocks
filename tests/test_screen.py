"""The quarterly screen (CLAUDE.md §15.1): SEC frames in, candidates to study out.

Synthetic payloads and companies, built so every answer is known in advance; no network.
"""

from __future__ import annotations

import datetime as dt
import pathlib

import pandas as pd
import pytest

from core import config
from db import loader
from ingest import screen as ingest_screen
from transform import screen as sc

CIK_A, CIK_B = "0000000001", "0000000002"


# --- Ingest -------------------------------------------------------------------------------


def test_the_screen_year_waits_for_the_annual_reports():
    """10-Ks land from February to April: before April last year is still thin."""
    assert ingest_screen.screen_year(dt.date(2026, 3, 31)) == 2024
    assert ingest_screen.screen_year(dt.date(2026, 4, 1)) == 2025


def test_instants_take_the_year_end_and_growth_metrics_the_year_before():
    assert ingest_screen.periods_for("cash", {"kind": "instant"}, 2025) == ["CY2025Q4I"]
    assert ingest_screen.periods_for("revenue", {"kind": "duration"}, 2025) == ["CY2025", "CY2024"]
    assert ingest_screen.periods_for("sbc", {"kind": "duration"}, 2025) == ["CY2025"]


def test_frame_rows_keeps_the_universe_and_fails_loudly_on_a_changed_payload():
    payload = {"data": [{"cik": 1, "end": "2025-12-31", "val": 10.0},
                        {"cik": 99, "end": "2025-12-31", "val": 5.0}]}
    rows = ingest_screen.frame_rows(payload, "revenue", "CY2025", {CIK_A}, 0)
    assert list(rows) == [CIK_A]
    assert rows[CIK_A]["series_id"] == f"{CIK_A}:revenue:CY2025"
    assert rows[CIK_A]["ts_release"] == loader.TS_RELEASE_UNKNOWN, "frames has no filing date"
    with pytest.raises(ValueError):
        ingest_screen.frame_rows({"rows": []}, "revenue", "CY2025", {CIK_A}, 0)


def test_the_screen_runs_once_a_quarter_unless_forced(tmp_path):
    conn = loader.init_db(tmp_path / "s.db")
    loader.upsert_observations(conn, pd.DataFrame([{
        "source": ingest_screen.SOURCE, "series_id": f"{CIK_A}:revenue:CY2025",
        "ts": "2025-12-31", "ts_release": "", "value": 1.0}]))
    settings = config.load_settings()
    fresh = ingest_screen.ScreenIngester(settings)
    fresh.attach_database(conn)
    assert fresh.fetch().empty, "ran today: not due for 90 days"
    forced = ingest_screen.ScreenIngester(settings)
    forced.force = True
    forced.attach_database(conn)
    assert forced._skip_reason is None
    conn.close()


# --- Transform ----------------------------------------------------------------------------


def frames(cik: str, **metrics: float) -> list[dict]:
    """``metric=value`` for CY2025; ``metric_prev=value`` for CY2024; instants at Q4."""
    rows = []
    for key, value in metrics.items():
        metric, prev = (key[:-5], True) if key.endswith("_prev") else (key, False)
        instant = metric in ("cash", "long_term_debt", "equity")
        period = "CY2025Q4I" if instant else ("CY2024" if prev else "CY2025")
        rows.append({"source": "sec_frames", "series_id": f"{cik}:{metric}:{period}",
                     "ts": "2024-12-31" if prev else "2025-12-31", "ts_release": "",
                     "value": value})
    return rows


def closes(ticker: str, value: float, day: str = "2026-09-24") -> list[dict]:
    return [{"series_id": f"{ticker}:close_raw", "ts": day, "value": value}]


REGISTRY = {CIK_A: ("AAA", "Alpha Inc"), CIK_B: ("BBB", "Beta Corp")}


def test_the_ratios_of_a_healthy_company():
    obs = pd.DataFrame(frames(CIK_A, revenue=1000.0, revenue_prev=800.0, operating_income=200.0,
                              net_income=150.0, operating_cash_flow=250.0, capex=50.0,
                              sbc=30.0, diluted_shares=100.0, diluted_shares_prev=102.0,
                              cash=100.0, long_term_debt=300.0))
    row = sc.screen_table(obs, pd.DataFrame(closes("AAA", 20.0)), [], REGISTRY).iloc[0]
    assert row["revenue_growth"] == pytest.approx(0.25)
    assert row["fcf_margin"] == pytest.approx(0.20)
    assert row["cash_conversion"] == pytest.approx(200 / 150)
    assert row["dilution"] == pytest.approx(100 / 102 - 1), "fewer shares: negative dilution"
    assert row["market_cap"] == pytest.approx(2000.0)
    assert row["fcf_after_sbc_yield"] == pytest.approx(170 / 2000)
    assert row["ev_ebit"] == pytest.approx((2000 + 300 - 100) / 200)
    assert not row["debt_missing"]


def test_nothing_is_divided_by_a_negative_base():
    """A multiple over losses would read as cheap, cash conversion over a loss as the
    worst case: both are left empty (section 12)."""
    obs = pd.DataFrame(frames(CIK_A, revenue=1000.0, revenue_prev=-5.0, operating_income=-50.0,
                              net_income=-80.0, operating_cash_flow=60.0, capex=10.0,
                              diluted_shares=100.0, cash=10.0))
    row = sc.screen_table(obs, pd.DataFrame(closes("AAA", 20.0)), [], REGISTRY).iloc[0]
    for column in ("revenue_growth", "ev_ebit", "cash_conversion"):
        assert pd.isna(row[column]), column
    assert row["fcf_margin"] == pytest.approx(0.05), "a positive ratio still stands"
    assert row["debt_missing"], "no debt concept: flagged, counted as zero in the EV"
    assert pd.isna(row["fcf_after_sbc_yield"]), "no SBC filed is unknown, not zero"


def test_a_split_after_the_fiscal_year_does_not_divide_the_market_cap():
    obs = pd.DataFrame(frames(CIK_A, revenue=1.0, diluted_shares=100.0))
    split = [{"ticker": "AAA", "kind": "split", "ex_date": "2026-06-01", "ratio": 4.0,
              "source": "yfinance"}]
    row = sc.screen_table(obs, pd.DataFrame(closes("AAA", 5.0)), split, REGISTRY).iloc[0]
    assert row["market_cap"] == pytest.approx(5.0 * 100 * 4), "count brought to today's units"


def test_class_shares_find_their_price_with_a_dash():
    obs = pd.DataFrame(frames(CIK_A, revenue=1.0, diluted_shares=10.0))
    row = sc.screen_table(obs, pd.DataFrame(closes("BRK-B", 3.0)), [],
                          {CIK_A: ("BRK.B", "Berkshire")}).iloc[0]
    assert row["market_cap"] == pytest.approx(30.0)


def test_a_filter_counts_what_it_cannot_judge_instead_of_hiding_it():
    table = pd.DataFrame({"dilution": [0.00, 0.10, None], "sbc_over_revenue": [0.01] * 3,
                          "fcf_margin": [0.1] * 3, "market_cap": [5e9] * 3})
    result = sc.apply_filters(table, {"max_dilution": 0.02})
    assert len(result.table) == 1 and result.failed == 1
    assert result.unknown == {"dilution": 1}
    assert len(sc.apply_filters(table, {"max_dilution": 0.02}, keep_unknown=True).table) == 2
    assert len(sc.apply_filters(table, {"max_dilution": None}).table) == 3, "off means off"


# --- The page -----------------------------------------------------------------------------

st = pytest.importorskip("streamlit", reason="the 'app' extra is not installed")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import data as app_data  # noqa: E402
from app.format import PUBLIC_PAGES  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = str(REPO_ROOT / "app" / "main.py")
PAGE = str(REPO_ROOT / "app" / "pages" / "screen.py")


@pytest.fixture()
def screen_db(tmp_path, monkeypatch):
    db_path = tmp_path / "screen.db"
    monkeypatch.setattr(app_data, "db_path", lambda: db_path)
    st.cache_data.clear()
    conn = loader.init_db(db_path)
    today = dt.date.today().isoformat()
    rows = frames(CIK_A, revenue=1000.0, revenue_prev=900.0, operating_income=200.0,
                  net_income=150.0, operating_cash_flow=250.0, capex=50.0, sbc=10.0,
                  diluted_shares=1e9, diluted_shares_prev=1e9, cash=10.0)
    rows += frames(CIK_B, revenue=1000.0, revenue_prev=900.0, operating_cash_flow=250.0,
                   capex=50.0, sbc=10.0, diluted_shares=1.2e9, diluted_shares_prev=1e9)
    rows += [{**r, "source": "yfinance", "ts_release": today, "ts": today}
             for r in closes("AAA", 20.0) + closes("BBB", 20.0)]
    loader.upsert_observations(conn, pd.DataFrame(rows))
    loader.upsert_companies(conn, [
        {"cik": c, "ticker": t, "name": n, "sector": None, "thesis_category": None,
         "first_seen": today, "status": "active"} for c, (t, n) in REGISTRY.items()])
    conn.close()
    yield db_path
    st.cache_data.clear()


def render() -> AppTest:
    app = AppTest.from_file(MAIN, default_timeout=120).run()
    app.switch_page(PAGE)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def test_the_page_keeps_the_candidates_and_says_what_it_dropped(screen_db):
    app = render()
    table = pd.DataFrame(app.dataframe[0].value)
    assert list(table["Ticker"]) == ["AAA"], "BBB diluted 20 %: out by the default filter"
    assert any("1** pasan · 1 no pasan" in m.value for m in app.markdown)


def test_a_candidate_goes_to_study_with_its_cik_quoted(screen_db):
    app = render()
    app.selectbox[0].select("AAA").run()
    assert f'cik: "{CIK_A}"' in app.code[0].value, "an unquoted CIK can turn octal (§9.13)"


def test_the_screen_is_never_public():
    """It marks what is held and studied — account information."""
    assert "Cribado" not in PUBLIC_PAGES


# --- Peers ------------------------------------------------------------------------------------


def peer_table(n: int, sic_of_others: str) -> tuple[pd.DataFrame, dict]:
    ciks = [f"{i:010d}" for i in range(1, n + 2)]
    table = pd.DataFrame({"cik": ciks, "ticker": [f"T{i}" for i in range(len(ciks))],
                          "revenue_growth": [0.30] + [0.10 * i / n for i in range(n + 1)][1:],
                          "dilution": [None] + [0.01] * n})
    for column, _ in sc.PEER_METRICS:
        if column not in table:
            table[column] = None
    sics = {ciks[0]: "7372", **{c: sic_of_others for c in ciks[1:]}}
    return table, sics


def test_the_company_is_placed_within_its_own_industry():
    table, sics = peer_table(8, "7372")
    result = sc.peer_comparison(table, table["cik"][0], sics)
    assert result.digits == 4 and len(result.peers) == 8
    growth = result.rows.set_index("metric").loc["revenue_growth"]
    assert growth["percentile"] == 1.0, "grows faster than every peer"
    assert growth["median"] == pytest.approx(0.05625)
    assert pd.isna(result.rows.set_index("metric").loc["dilution", "percentile"]), \
        "its own value missing: no position, never a guess"


def test_a_thin_industry_widens_the_code_and_says_so():
    table, sics = peer_table(8, "7379")          # same 3 digits, different 4
    result = sc.peer_comparison(table, table["cik"][0], sics)
    assert result.digits == 3 and result.code == "737"


def test_without_enough_peers_there_is_no_comparison():
    table, sics = peer_table(3, "7372")
    assert sc.peer_comparison(table, table["cik"][0], sics) is None
