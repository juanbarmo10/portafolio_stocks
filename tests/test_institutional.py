"""Institutional holdings from the 13F data sets (user request 2026-09-29), known answers.

What is defended: a restatement replaces a manager's holdings and a new-holdings amendment
adds to them, each version dated by its filing (section 9.6); a position a restatement
removes becomes zero; windows are read by their filing dates; CUSIPs come from the FTD files.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from ingest import institutional as ing

CUSIP, CIK = "433000106", "0001773751"


def row(accession, filed, amendment, shares, filer="0000000001", period="2026-06-30",
        cusip=CUSIP):
    return {"accession": accession, "filed": filed, "period": period, "filer_cik": filer,
            "manager": "Fondo", "amendment_type": amendment, "cusip": cusip, "shares": shares}


def test_restatement_replaces_new_holdings_adds_each_dated_by_its_filing():
    rows = pd.DataFrame([row("A", "2026-08-10", "", 1000.0),
                         row("B", "2026-08-20", "NEW HOLDINGS", 500.0),
                         row("C", "2026-09-01", "RESTATEMENT", 1200.0)])
    out = [(r["ts_release"], r["value"]) for r in ing.versions(rows, {CUSIP: CIK})]
    assert out == [("2026-08-10", 1000.0), ("2026-08-20", 1500.0), ("2026-09-01", 1200.0)]


def test_a_position_a_restatement_removes_becomes_zero():
    rows = pd.DataFrame([row("A", "2026-08-10", "", 1000.0),
                         row("C", "2026-09-01", "RESTATEMENT", 50.0, cusip="999999999")])
    out = ing.versions(rows, {CUSIP: CIK, "999999999": "0000000002"})
    mine = [(r["ts_release"], r["value"]) for r in out if r["series_id"].startswith(CIK)]
    assert mine == [("2026-08-10", 1000.0), ("2026-09-01", 0.0)]
    assert {r["series_id"] for r in out} == {f"{CIK}:inst:0000000001",
                                             "0000000002:inst:0000000001"}


def test_two_rows_of_the_same_filing_add_up():
    rows = pd.DataFrame([row("A", "2026-08-10", "", 100.0), row("A", "2026-08-10", "", 50.0)])
    assert [r["value"] for r in ing.versions(rows, {CUSIP: CIK})] == [150.0]


@pytest.mark.parametrize("name, end", [
    ("2023q4_form13f.zip", dt.date(2023, 12, 31)),
    ("01jun2026-31aug2026_form13f.zip", dt.date(2026, 8, 31)),
    ("01dec2025-28feb2026_form13f.zip", dt.date(2026, 2, 28)),
    ("readme.zip", None),
])
def test_a_window_ends_on_its_last_filing_date(name, end):
    assert ing.window_end(name) == end


def test_cusips_come_from_the_ftd_pages_first_seen_wins():
    pages = [pd.DataFrame({"SYMBOL": ["HIMS", "AAA"], "CUSIP": ["433000106", "x"]}),
             pd.DataFrame({"SYMBOL": ["HIMS"], "CUSIP": ["other"]})]
    assert ing.ftd_cusips(pages, {"HIMS"}) == {"HIMS": "433000106"}


def test_a_table_inside_a_folder_of_the_zip_is_found():
    import io  # noqa: PLC0415
    import zipfile  # noqa: PLC0415

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("01JUN2025-31AUG2025_form13f/INFOTABLE.tsv", "x")
    archive = zipfile.ZipFile(buffer)
    assert ing.member(archive, "INFOTABLE.tsv") == "01JUN2025-31AUG2025_form13f/INFOTABLE.tsv"
    with pytest.raises(KeyError):
        ing.member(archive, "SUBMISSION.tsv")


def test_filers_are_stored_without_an_ingestion_stamp(tmp_path):
    from db import loader  # noqa: PLC0415

    conn = loader.init_db(tmp_path / "f.db")
    assert loader.upsert_filers(conn, [{"cik": "0000000001", "name": "Fondo"}]) == 1
    assert conn.execute("SELECT name FROM filers").fetchone()[0] == "Fondo"


def test_nothing_new_since_the_last_run_is_skipped_and_force_rebuilds(tmp_path, monkeypatch):
    """The versions cost ~156 s to rebuild and change once a quarter: with the same SEC files
    and the same companies there is nothing to recompute."""
    from types import SimpleNamespace

    from core.config import load_settings

    ingester = ing.InstitutionalIngester(load_settings())
    ingester._cache = tmp_path
    page = '<a href="/files/2026q2_form13f.zip">x</a>'
    monkeypatch.setattr(ingester, "_get", lambda url, **kw: SimpleNamespace(text=page))
    monkeypatch.setattr(ingester, "_cusips", lambda: {CUSIP: CIK})
    cached = tmp_path / "form13f" / "2026q2_form13f.zip.csv.gz"
    cached.parent.mkdir(parents=True)
    pd.DataFrame([row("a1", "2026-08-10", "", 100)]).to_csv(cached, index=False,
                                                             compression="gzip")
    cached.with_suffix(".cusips").write_text(CUSIP, encoding="utf-8")
    assert not ingester.fetch().empty, "first run builds the versions"
    assert ingester.fetch().empty, "same files, same companies: skipped"
    ingester.force = True
    assert not ingester.fetch().empty, "--force rebuilds"
