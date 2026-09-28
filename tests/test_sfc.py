"""Colombia's supervisor, the CUIF (CLAUDE.md §15.5, point 11), with known answers.

What is defended: the loan book is classified by risk category from the account's name and
summed per letter; the result accumulates from January, so the return on equity annualizes
by the month and never sums months; and the publication date follows the BCB rule — the
first time a month is seen, or month end + lag for the history of a first run.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from ingest import sfc as ingest
from transform import sfc

ACCOUNTS = {"100000": "total_assets", "140000": "loans_net", "210700": "deposits_term",
            "210800": "deposits_savings", "300000": "equity", "590000": "net_income_ytd"}


def row(codigo, cuenta, nombre, valor, tipo="4"):
    return {"tipo_entidad": tipo, "codigo_entidad": codigo, "cuenta": cuenta,
            "nombre_cuenta": nombre, "valor": str(valor)}


def test_month_ends_are_the_closed_months_before_today():
    assert ingest.month_ends(3, dt.date(2026, 9, 28)) == [
        dt.date(2026, 6, 30), dt.date(2026, 7, 31), dt.date(2026, 8, 31)]


@pytest.mark.parametrize("name, letter", [
    ("CATEGORIA A RIESGO NORMAL CARTERA Y OPERACIONES DE LEASING DE CONSUMO", "A"),
    ("CATEGORIA E RIESGO DE INCOBRABILIDAD", "E"),
    ("CARTERA Y OPERACIONES DE LEASING DE CONSUMO", None),
    ("CATEGORIAS VARIAS", None),
])
def test_the_risk_category_comes_from_the_account_name(name, letter):
    assert ingest.category_of(name) == letter


def test_a_months_rows_become_accounts_and_category_sums_per_entity():
    rows = [row("128", "100000", "ACTIVO", 1000),
            row("128", "140805", "CATEGORIA A RIESGO NORMAL CONSUMO", 300),
            row("128", "141005", "CATEGORIA A RIESGO NORMAL VIVIENDA", 100),
            row("128", "140825", "CATEGORIA E RIESGO DE INCOBRABILIDAD CONSUMO", 50),
            row("999", "100000", "ACTIVO", 7)]                # not configured
    out = ingest.month_values(rows, ACCOUNTS, {"4-128"})
    assert out == {"4-128": {"total_assets": 1000.0, "loans_cat_a": 400.0,
                             "loans_cat_e": 50.0}}


def fetch_with(monkeypatch, pages, *, known=None):
    cfg = {"url": "https://example.invalid/cuif.json", "history_months": 2,
           "recent_months": 1, "derived_lag_days": 60,
           "entities": [{"tipo": 4, "codigo": 128, "name": "Nu Colombia"}],
           "accounts": ACCOUNTS}

    class Settings:
        def source(self, name):
            return cfg

    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    def fake_get(url, params, timeout):
        month = params["$where"].split("fecha_corte='")[1][:10]
        return Response(pages.get(month, []))

    monkeypatch.setattr(ingest.requests, "get", fake_get)
    monkeypatch.setattr(ingest.dt, "date", type("D", (dt.date,), {
        "today": staticmethod(lambda: dt.date(2026, 9, 28))}))
    ing = ingest.SfcIngester(Settings())
    if known:
        ing._known, ing._seen = known
    return ing.fetch()


def test_the_first_run_dates_history_late_and_what_is_visible_now_today(monkeypatch):
    pages = {"2026-07-31": [row("128", "100000", "ACTIVO", 1000)],
             "2026-08-31": [row("128", "100000", "ACTIVO", 1100)]}
    out = fetch_with(monkeypatch, pages).set_index("ts")
    assert out.loc["2026-07-31", "ts_release"] == "2026-09-28", \
        "month end + 60 days is still ahead: it is visible now, so published by now"
    assert out.loc["2026-08-31", "ts_release"] == "2026-09-28"


def test_a_value_already_stored_is_not_written_again(monkeypatch):
    pages = {"2026-08-31": [row("128", "100000", "ACTIVO", 1100)]}
    known = ({"4-128:total_assets|2026-08-31": [("2026-09-20", 1100.0)]}, {"4-128:total_assets"})
    assert fetch_with(monkeypatch, pages, known=known).empty


# --- Transform ------------------------------------------------------------------------------


def obs(key, month, value, release=None):
    return {"source": "sfc_cuif", "series_id": f"4-128:{key}", "ts": month,
            "ts_release": release or month, "value": value}


OBS = pd.DataFrame([
    obs("total_assets", "2026-07-31", 12_000.0),
    obs("loans_net", "2026-07-31", 3_000.0), obs("loans_net", "2025-07-31", 2_000.0),
    obs("deposits_term", "2026-07-31", 4_000.0), obs("deposits_savings", "2026-07-31", 6_000.0),
    obs("deposits_term", "2025-07-31", 3_000.0), obs("deposits_savings", "2025-07-31", 5_000.0),
    obs("loans_cat_a", "2026-07-31", 900.0), obs("loans_cat_c", "2026-07-31", 40.0),
    obs("loans_cat_d", "2026-07-31", 30.0), obs("loans_cat_e", "2026-07-31", 30.0),
    obs("equity", "2026-07-31", 1_200.0), obs("net_income_ytd", "2026-07-31", -70.0),
    obs("total_assets", "2026-08-31", 12_500.0, release="2026-10-15"),   # not public yet
])


def test_the_snapshot_growth_deposits_quality_and_annualized_return():
    snap = sfc.assess(OBS, "4-128", "Nu Colombia", "2026-09-28")
    assert snap.month == "2026-07-31", "August is not published on the 28th of September"
    assert snap.loans_growth == pytest.approx(0.5)
    assert snap.deposits == pytest.approx(10_000.0)
    assert snap.deposits_growth == pytest.approx(0.25)
    assert snap.loans_to_deposits == pytest.approx(0.3)
    assert snap.risk_share == pytest.approx(100.0 / 1_000.0)
    assert snap.roe_annualized == pytest.approx(-70.0 * 12 / 7 / 1_200.0), \
        "seven months to date, annualized — never a sum of months"


def test_a_missing_year_ago_month_leaves_growth_unknown():
    snap = sfc.assess(OBS[OBS["ts"] != "2025-07-31"], "4-128", "Nu Colombia", "2026-09-28")
    assert snap.loans_growth is None and snap.deposits_growth is None


def test_compare_leaves_out_what_was_not_published():
    entities = [{"tipo": 4, "codigo": 128, "name": "Nu Colombia"},
                {"tipo": 1, "codigo": 65, "name": "Lulo Bank"}]
    assert [s.name for s in sfc.compare(OBS, entities, "2026-09-28")] == ["Nu Colombia"]
