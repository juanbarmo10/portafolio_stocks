"""The per-company page: the one place where a thesis meets its own numbers.

Rendered end to end with Streamlit's test harness against a database seeded from the real
Microsoft fixture and a thesis card written through the real config path — so what these
tests exercise is the same loading, merging and validation a running panel does.

The asymmetry they defend: **public mode hides the position, not the company.** Microsoft's
revenue is in a 10-K anyone can download; the size of the holding is not.

Skipped when the ``app`` extra is not installed.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

pytest.importorskip("streamlit", reason="the 'app' extra is not installed")
pytest.importorskip("ibflex", reason="the 'ibkr' extra is not installed")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import data as app_data  # noqa: E402
from core import config  # noqa: E402
from db import loader  # noqa: E402
from ingest.sec_xbrl import extract_metric  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = str(REPO_ROOT / "app" / "main.py")
COMPANY = str(REPO_ROOT / "app" / "pages" / "company.py")
FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "sec_companyfacts_msft.json"
CIK = "0000789019"

CANDIDATE = """
universe:
  watchlist:
    - ticker: HIMS
      cik: "0001773751"
"""

CARD = """
universe:
  tracked:
    - ticker: MSFT
      cik: "0000789019"
      sector: "Services-Prepackaged Software"
      thesis_category: "infraestructura de software"
      thesis: "Costes de cambio altos en la nube empresarial."
      value_accrual: "Margen creciente con recompras que superan al SBC."
      key_metric: "cash_conversion"
      invalidation: "Conversión a caja por debajo de 0,6 dos años seguidos."
      review_date: 2027-06-30
      invalidation_rule: { metric: cash_conversion, operator: "<", threshold: 0.6 }
"""


def seed_database(db_path) -> None:
    """The Microsoft fixture, a filing, an earnings date and a position in MSFT."""
    facts = json.loads(FIXTURE.read_text(encoding="utf-8"))["facts"]
    cfg = config.load_settings().source("sec")
    records = []
    for metric, spec in cfg["concepts"].items():
        rows, _ = extract_metric(facts, metric, spec, CIK, cfg["duration_windows"])
        records.extend(rows)

    conn = loader.init_db(db_path)
    try:
        loader.upsert_observations(conn, pd.DataFrame(records))
        loader.upsert_observations(conn, pd.DataFrame([
            {"source": "ibkr", "series_id": "MSFT:position_qty", "ts": "2026-09-15",
             "ts_release": "2026-09-15", "value": 2.0},
            {"source": "ibkr", "series_id": "MSFT:position_cost_basis", "ts": "2026-09-15",
             "ts_release": "2026-09-15", "value": 900.0},
            {"source": "yfinance", "series_id": "MSFT:close_raw", "ts": "2026-09-15",
             "ts_release": "2026-09-15", "value": 500.0},
        ]))
        loader.upsert_trades(conn, pd.DataFrame([
            {"trade_id": "1", "conid": "272093", "cik": CIK, "ticker": "MSFT",
             "ts": "2026-02-02T15:00:00+00:00",
             "side": "buy", "quantity": 2.0, "price": 449.0, "currency": "USD",
             "commission": -0.35, "fx_rate": 1.0},
        ]))
        loader.upsert_filings(conn, [
            {"accession": "0000950170-26-000001", "cik": CIK, "form": "10-K",
             "period_end": "2026-06-30", "filed_date": "2026-07-29", "is_amended": 0,
             "url": "https://www.sec.gov/Archives/edgar/data/789019/x/msft-20260630.htm"},
            {"accession": "0000950170-20-000002", "cik": CIK, "form": "10-Q/A",
             "period_end": "2020-03-31", "filed_date": "2020-08-10", "is_amended": 1,
             "url": "https://www.sec.gov/Archives/edgar/data/789019/y/msft-a.htm"},
        ])
        loader.upsert_events(conn, [
            {"event_id": f"{CIK}:earnings:next", "category": "earnings", "cik": CIK,
             "ts": "2026-10-28", "is_estimated": 1, "label": "MSFT: próximos resultados",
             "payload": json.dumps({"method": "misma fecha del año anterior + 364 días",
                                    "last_confirmed": "2026-07-29"})},
        ])
    finally:
        conn.close()


@pytest.fixture()
def company_app(tmp_path, monkeypatch):
    """A panel with one written thesis, its filings and a position in it."""
    local = tmp_path / "settings.local.yaml"
    local.write_text(CARD, encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", local)
    config.load_settings.cache_clear()

    db_path = tmp_path / "company.db"
    seed_database(db_path)
    monkeypatch.setattr(app_data, "db_path", lambda: db_path)
    st.cache_data.clear()
    yield db_path
    st.cache_data.clear()


def render(page: str = COMPANY) -> AppTest:
    app = AppTest.from_file(MAIN, default_timeout=90).run()
    app.switch_page(page)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def rendered_text(app: AppTest) -> str:
    chunks = []
    for kind in ("title", "header", "subheader", "markdown", "caption", "text",
                 "info", "warning", "error", "success", "metric"):
        for element in app.get(kind):
            for attribute in ("label", "value", "body", "delta"):
                piece = getattr(element, attribute, None)
                if isinstance(piece, str):
                    chunks.append(piece)
    for element in app.get("dataframe"):
        if element.value is not None:
            chunks.append(pd.DataFrame(element.value).to_string())
    return "\n".join(chunks)


# --- With a written thesis ----------------------------------------------------


def test_the_page_shows_the_thesis_and_its_own_numbers(company_app):
    text = rendered_text(render())

    assert "Costes de cambio altos" in text, "the thesis is not shown"
    assert "Conversión a caja por debajo de 0,6" in text, "the invalidation text is not shown"
    assert "331,84 mil millones USD" in text, "TTM revenue did not reach the page"
    assert "no lo interpreta" in text, "the page must say it does not judge the prose"


def test_the_structured_rule_is_evaluated_and_shown(company_app):
    """Microsoft's cash conversion is ~0.50, so the written "below 0.6" rule is breached."""
    text = rendered_text(render())
    assert "CRUZADO" in text


def test_the_amended_filing_is_raised_as_a_governance_flag(company_app):
    text = rendered_text(render())
    assert "10-Q/A" in text
    assert "para investigar" in text, "it must read as a flag, not as proof"


def test_an_estimated_earnings_date_is_never_shown_as_confirmed(company_app):
    text = rendered_text(render())
    assert "2026-10-28" in text
    assert "estimado" in text
    assert "misma fecha del año anterior" in text, "the method must travel with the date"


def test_the_position_block_shows_the_real_holding_locally(company_app):
    text = rendered_text(render())
    assert "Tu posición" in text
    assert "1.000,00 USD" in text, "2 shares at 500 should be valued"


# --- Public mode --------------------------------------------------------------


def test_public_mode_hides_the_position_but_keeps_the_company(company_app, monkeypatch):
    """The asymmetry: a 10-K is public, the size of your holding is not."""
    monkeypatch.setenv("PUBLIC_MODE", "1")
    config.load_settings.cache_clear()
    st.cache_data.clear()

    text = rendered_text(render())

    assert "331,84 mil millones USD" in text, "public filings must stay visible"
    assert "Tu posición" not in text, "the holding block leaked into the public view"
    assert "1.000,00 USD" not in text, "the position value leaked"
    assert "Vista pública" in text


# --- Without a written thesis -------------------------------------------------


def test_with_nothing_written_the_page_explains_both_routes(tmp_path, monkeypatch):
    """The empty state has to teach the distinction, because it is the whole design.

    A company reaches this page either as a candidate under study (ticker and CIK) or with
    a full thesis card. Offering only the second is what made the panel unusable *during*
    the research, and the empty state is where a new reader learns there are two.
    """
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", tmp_path / "absent.yaml")
    config.load_settings.cache_clear()
    monkeypatch.setattr(app_data, "db_path", lambda: tmp_path / "empty.db")
    st.cache_data.clear()

    text = rendered_text(render())
    assert "watchlist" in text, "the under-study route is not explained"
    assert "invalidation" in text, "the required thesis fields must be spelled out"
    assert "después de mirar los números" in text, "the order is the point"


def test_a_card_without_a_cik_never_reaches_the_page(tmp_path, monkeypatch):
    """Section 9.3, and the protection turns out to live one layer earlier than expected.

    The page carries its own guard for a card with no CIK, but it is unreachable through
    the normal path: ``validate_theses`` refuses to load the configuration at all and names
    every missing field. That is the better failure — it happens once, at startup, instead
    of once per page view, and it names what to fix.
    """
    local = tmp_path / "settings.local.yaml"
    local.write_text(
        'universe:\n  tracked:\n    - ticker: XXXX\n      invalidation: "algo"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", local)
    config.load_settings.cache_clear()

    with pytest.raises(ValueError, match="cik"):
        config.load_settings()


# --- A candidate under study (section 5.1) ------------------------------------


@pytest.fixture()
def study_app(tmp_path, monkeypatch):
    """A panel whose only company is one being read about, with no thesis."""
    local = tmp_path / "settings.local.yaml"
    local.write_text(CANDIDATE, encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", local)
    config.load_settings.cache_clear()

    db_path = tmp_path / "study.db"
    seed_database(db_path)          # the Microsoft figures; the point is the thesis block
    monkeypatch.setattr(app_data, "db_path", lambda: db_path)
    st.cache_data.clear()
    yield db_path
    st.cache_data.clear()


def test_a_candidate_gets_the_numbers_but_no_opinion(study_app):
    """The asymmetry of the fourth circle: full data, zero judgement.

    The numbers are what you study a company *with*, so withholding them would defeat the
    purpose. What is withheld is the thesis block — there is nothing written to show, and
    rendering empty fields would suggest the panel is waiting for something rather than
    that the research is in progress.
    """
    text = rendered_text(render())

    assert "En estudio" in text
    assert "sin ficha de tesis" in text
    assert "Criterio de invalidación" not in text, "an empty thesis block was rendered"
    assert "no entra al tablero de tesis" in text


def test_the_page_says_how_to_promote_it(study_app):
    """A dead end with no exit is a worse deadlock than the one this replaced."""
    text = rendered_text(render())

    assert "watchlist" in text and "tracked" in text
    assert "después de mirar los números" in text, "the order is the point"


def test_a_candidate_never_reaches_the_thesis_board(study_app):
    """Section 5.2, defended at the layer that decides: the board reads tracked only."""
    from transform import thesis as th

    settings = config.load_settings()
    assert settings.tracked_companies == []
    assert th.board(settings.tracked_companies, pd.DataFrame(), None, None,
                    "2026-09-23") == []


# --- The public deployment: no local config, the list comes from the public copy --------


def test_the_public_copy_lists_its_published_companies_without_opinion(tmp_path, monkeypatch):
    """In the cloud there is no settings.local.yaml. The page lists the ``companies`` the
    public sync uploaded, all as under study: numbers, never a thesis."""
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", tmp_path / "absent.yaml")
    monkeypatch.setenv("PUBLIC_MODE", "1")
    config.load_settings.cache_clear()
    db_path = tmp_path / "public_copy.db"
    seed_database(db_path)
    conn = loader.init_db(db_path)
    try:
        loader.upsert_companies(conn, [{"cik": CIK, "ticker": "MSFT", "name": "MICROSOFT",
                                        "sector": None, "thesis_category": None,
                                        "first_seen": "2026-09-24", "status": "active"}])
    finally:
        conn.close()
    monkeypatch.setattr(app_data, "db_path", lambda: db_path)
    st.cache_data.clear()

    text = rendered_text(render())
    assert "No hay ninguna empresa escrita" not in text
    assert "Costes de cambio altos" not in text, "no thesis text in the public copy"
    assert "Ingresos" in text or "ingresos" in text, "the audited numbers render"
    assert "tu lista de estudio" not in text and "settings.local.yaml" not in text, (
        "the visitor is not the owner: no operator instructions in public")
