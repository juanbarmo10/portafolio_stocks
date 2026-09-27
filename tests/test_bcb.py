"""The Brazilian supervisor's data (RESEARCH.md §2.43), on synthetic rows with known answers.

The semester figures reproduce the shape measured on Nu Holdings (2025: Q1 2,87 → S1 5,97
billion reais); every other value is invented.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from ingest import bcb as ingest_bcb
from ingest.sec_filings import foreign_earnings_dates
from transform import bcb

CODE = "C0000001"


def test_quarter_ends_stop_before_today():
    ends = ingest_bcb.quarter_ends("2025q3", dt.date(2026, 3, 31))
    assert ends == [dt.date(2025, 9, 30), dt.date(2025, 12, 31)], "a quarter ending today is not out"


def test_columns_are_read_by_their_name_before_the_formula_label():
    rows = [{"NomeColuna": "Lucro Líquido \n(z) = (w) + (x)", "Saldo": 5.0},
            {"NomeColuna": "Índice de Basileia", "Saldo": None},
            {"NomeColuna": "Outra", "Saldo": 9.0}]
    out = ingest_bcb.column_values(rows, {"Lucro Líquido": "net", "Índice de Basileia": "basel"})
    assert out == {"net": 5.0}, "an empty value is left out, never read as zero"


def test_the_publication_date_errs_late_and_is_kept_once_seen():
    today, end = dt.date(2026, 9, 25), dt.date(2026, 3, 31)
    # First run, old quarter: derived, quarter end + lag.
    assert ingest_bcb.release_for("s", end, 1.0, {}, today, True, 120) == ("2026-07-29", False)
    # First run, a quarter whose derived date is still ahead: visible now, so seen today.
    assert ingest_bcb.release_for("s", dt.date(2026, 6, 30), 1.0, {}, today, True, 120) \
        == ("2026-09-25", True)
    # Already stored with this value: nothing new.
    known = {"s|2026-03-31": [("2026-07-29", 1.0)]}
    assert ingest_bcb.release_for("s", end, 1.0, known, today, False, 120) is None
    # Stored with another value: a revision, dated the day it was seen; the original stays.
    assert ingest_bcb.release_for("s", end, 1.5, known, today, False, 120) == ("2026-09-25", True)


def obs(key, ts, value, release=None):
    return {"source": "bcb_ifdata", "series_id": f"{CODE}:{key}", "ts": ts,
            "ts_release": release or ts, "value": value}


def test_the_quarter_is_derived_from_the_semester_accumulation():
    semester = pd.Series({pd.Timestamp("2025-03-31"): 2.87, pd.Timestamp("2025-06-30"): 5.97,
                          pd.Timestamp("2025-12-31"): 6.73})
    quarters = bcb.quarterly_from_semester(semester)
    assert quarters[pd.Timestamp("2025-06-30")] == pytest.approx(3.10)
    assert pd.Timestamp("2025-12-31") not in quarters.index, \
        "a Q4 without its Q3 is not the whole semester"


def test_growth_never_crosses_the_2025_change_of_basis():
    rows = [obs("total_assets", "2024-06-30", 1), obs("total_assets", "2025-06-30", 1),
            obs("credit_portfolio_classified", "2024-06-30", 100.0),
            obs("credit_portfolio", "2025-06-30", 150.0)]
    view = bcb.assess(pd.DataFrame(rows), CODE, "2026-01-01")
    assert view.credit_portfolio == 150.0 and view.credit_growth is None
    assert "4.966" in view.credit_basis


def test_a_quarter_is_invisible_before_its_publication_date():
    rows = [obs("total_assets", "2026-03-31", 1, "2026-07-29"),
            obs("total_assets", "2026-06-30", 2, "2026-09-25"),
            obs("problem_assets", "2026-06-30", 12.0, "2026-09-25"),
            obs("exposure_total", "2026-06-30", 100.0, "2026-09-25")]
    assert bcb.assess(pd.DataFrame(rows), CODE, "2026-09-24").quarter == "2026-03-31"
    view = bcb.assess(pd.DataFrame(rows), CODE, "2026-09-25")
    assert view.quarter == "2026-06-30" and view.problem_share == pytest.approx(0.12)


def test_a_foreign_issuers_results_are_the_days_with_several_6ks_on_the_quarter():
    """Nu Holdings' shape: 3-4 6-Ks on results day, all dated to the quarter just closed;
    other 6-Ks on other days (and ones pointing at a future period) are not results."""
    forms, filed, period, acc = [], [], [], []
    for day, per, n in [("2026-08-13", "2026-06-30", 4), ("2026-05-14", "2026-03-31", 3),
                        ("2026-09-10", "2026-09-30", 2), ("2026-07-06", "2026-09-30", 3)]:
        for i in range(n):
            forms.append("6-K"), filed.append(day), period.append(per), acc.append(f"{day}-{i}")
    submissions = {"filings": {"recent": {"form": forms, "filingDate": filed,
                                          "reportDate": period, "accessionNumber": acc}}}
    dates = [row["date"] for row in foreign_earnings_dates(submissions)]
    assert dates == ["2026-08-13", "2026-05-14"]


# --- An outage ends the IF.data requests for the run ----------------------------------


def ingester_with(get):
    from core.config import load_settings  # noqa: PLC0415
    ing = ingest_bcb.BcbIngester(load_settings())
    ing._get = get
    return ing


def test_an_outage_stops_ifdata_after_the_first_failure_with_one_message():
    """2026-09-26: sixteen requests answered HTTP 500 one after another, each after its
    retries. The first one must end IF.data for the run; SGS is another host and still runs."""
    calls = []

    def get(url, params=None):
        calls.append(url)
        if "olinda" in url:
            raise RuntimeError("500 Server Error")
        return []

    ing = ingester_with(get)
    ing.fetch()
    ifdata_calls = [c for c in calls if "olinda" in c]
    assert len(ifdata_calls) == 1, "one failed request, not one per quarter and report"
    assert any("api.bcb.gov.br" in c for c in calls), "the Selic is still asked"
    failures = ing.partial_failures()
    assert len(failures) == 1 and "nothing is lost" in failures[0]


def test_a_quarter_cut_by_an_outage_is_not_kept_half():
    """Report 1 of the first quarter answers, report 5 fails: nothing of that quarter is
    returned, so the next run asks for it whole and writes its release_observed row."""
    def get(url, params=None):
        if "olinda" not in url:
            return []
        if "Relatorio='1'" in url:
            return {"value": [{"NomeColuna": "Ativo Total", "Saldo": 10.0}]}
        raise RuntimeError("500 Server Error")

    ing = ingester_with(get)
    assert ing._ifdata(dt.date(2026, 9, 26)) == []
    assert len(ing.partial_failures()) == 1
