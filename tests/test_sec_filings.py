"""Filing history, restatement flags and the earnings calendar.

Against a trimmed slice of Microsoft's real ``submissions`` document: 92 filings, which
keeps the 8-K/10-Q/10-K history plus four Form 4s so the exclusion is testable.

The case Microsoft cannot provide is the one that matters most — it has never filed a
``10-K/A`` — so the restatement flag is exercised on a synthetic document, marked as such.
That is also the distinction the test suite is really defending: **most ``/A`` forms are
not restatements**, and treating them as such would cry wolf on every insider-transaction
correction.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import pandas as pd
import pytest

from core.config import Settings, load_settings
from ingest.sec_filings import (
    SecFilingsIngester,
    earnings_dates,
    estimate_next_earnings,
    event_rows,
    filing_rows,
    normalize_ticker,
    restatement_filings,
    ticker_map,
)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "sec_submissions_msft.json"
CIK = "0000789019"
FORMS = ["10-K", "10-Q", "8-K", "20-F", "40-F", "DEF 14A", "S-1"]


@pytest.fixture(scope="module")
def submissions():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def synthetic(forms: list[tuple[str, str, str, str]]) -> dict:
    """``[(form, reportDate, filingDate, items)]`` as a submissions document."""
    return {"filings": {"recent": {
        "accessionNumber": [f"0000000000-00-{i:06d}" for i in range(len(forms))],
        "form": [f[0] for f in forms],
        "reportDate": [f[1] for f in forms],
        "filingDate": [f[2] for f in forms],
        "items": [f[3] for f in forms],
        "primaryDocument": ["doc.htm"] * len(forms),
    }}}


# --- Filing history -----------------------------------------------------------


def test_only_the_configured_forms_are_stored(submissions):
    """Form 4 is hundreds of rows a year per company and nobody reads them yet."""
    rows = filing_rows(submissions, CIK, FORMS)
    stored = {row["form"] for row in rows}

    assert "4" not in stored
    assert {"10-K", "10-Q", "8-K"} <= stored
    assert all(row["cik"] == CIK for row in rows)


def test_an_amendment_is_kept_even_though_its_base_form_was_requested(submissions):
    """Asking for 8-K keeps 8-K/A too: the amendment is the interesting one."""
    rows = filing_rows(submissions, CIK, FORMS)
    amended = [row for row in rows if row["is_amended"]]
    assert amended, "the fixture contains an 8-K/A and it was dropped"
    assert all(row["form"].endswith("/A") for row in amended)


def test_the_url_points_at_the_actual_document(submissions):
    rows = filing_rows(submissions, CIK, FORMS)
    url = next(row["url"] for row in rows if row["url"])
    assert url.startswith("https://www.sec.gov/Archives/edgar/data/789019/")
    assert url.endswith(".htm")


def test_microsoft_has_never_restated(submissions):
    """A fact about the fixture, and the reason the next test is synthetic."""
    assert restatement_filings(filing_rows(submissions, CIK, FORMS)) == []


def test_only_amended_financial_forms_count_as_a_possible_restatement():
    """Synthetic. Section 9.6 is about re-filed *accounts*, not about any ``/A``.

    A ``4/A`` corrects an insider-transaction report. Flagging it as a restatement would
    raise a governance alarm several times a year at most companies, and an alarm that
    cries wolf is worse than no alarm.
    """
    document = synthetic([
        ("10-K/A", "2025-06-30", "2025-11-03", ""),
        ("4/A", "", "2025-11-04", ""),
        ("8-K/A", "2025-10-01", "2025-10-05", "2.02"),
    ])
    rows = filing_rows(document, CIK, [*FORMS, "4"])
    assert len(rows) == 3
    assert all(row["is_amended"] == 1 for row in rows)

    flagged = restatement_filings(rows)
    assert [row["form"] for row in flagged] == ["10-K/A"]


# --- Earnings dates -----------------------------------------------------------


def test_earnings_dates_come_from_8k_item_2_02(submissions):
    """The SEC's own record of when results were announced — a fact, not a scrape."""
    announcements = earnings_dates(submissions)
    assert len(announcements) >= 20
    assert announcements[0]["date"] > announcements[1]["date"], "not newest first"

    dates = [dt.date.fromisoformat(row["date"]) for row in announcements[:5]]
    gaps = [(dates[i] - dates[i + 1]).days for i in range(4)]
    assert all(85 <= gap <= 97 for gap in gaps), f"quarterly cadence broken: {gaps}"


def test_an_8k_without_item_2_02_is_not_an_earnings_announcement():
    """Synthetic: most 8-Ks are not results. Item 8.01 is "other events"."""
    document = synthetic([
        ("8-K", "2025-10-01", "2025-10-02", "8.01,9.01"),
        ("8-K", "2025-10-29", "2025-10-29", "2.02,9.01"),
    ])
    announcements = earnings_dates(document)
    assert [row["date"] for row in announcements] == ["2025-10-29"]


def test_item_2_02_is_matched_exactly_not_as_a_substring():
    """Synthetic: '12.02' or '2.021' must not pass for item 2.02."""
    document = synthetic([("8-K", "2025-10-01", "2025-10-02", "12.02,9.01")])
    assert earnings_dates(document) == []


# --- The estimate -------------------------------------------------------------


def test_the_next_date_is_estimated_from_the_same_quarter_last_year(submissions):
    """Microsoft announced on 2025-10-29, so the next Q1 lands on 2026-10-28."""
    estimate = estimate_next_earnings(earnings_dates(submissions))
    assert estimate is not None
    date, method = estimate
    assert date == "2026-10-28"
    assert "año anterior" in method


def test_the_estimate_preserves_the_weekday(submissions):
    """364 days, not "+1 year": results land midweek and a year drifts into the weekend."""
    announcements = earnings_dates(submissions)
    date, _method = estimate_next_earnings(announcements)
    anchor = dt.date.fromisoformat(announcements[3]["date"])

    assert dt.date.fromisoformat(date).weekday() == anchor.weekday()
    assert dt.date.fromisoformat(date).weekday() <= 4, "results are not announced at weekends"


def test_a_short_history_falls_back_to_the_median_gap_and_says_so():
    """Synthetic: with three announcements there is no same-quarter-last-year anchor."""
    announcements = [{"date": d, "accession": "x"} for d in
                     ("2026-04-29", "2026-01-28", "2025-10-29")]
    date, method = estimate_next_earnings(announcements)

    assert date == "2026-07-29"
    assert "mediana" in method


def test_without_history_there_is_no_estimate():
    """Section 12: no date at all beats a guess dressed as a schedule."""
    assert estimate_next_earnings([]) is None
    assert estimate_next_earnings([{"date": "2026-04-29", "accession": "x"}]) is None


# --- Events -------------------------------------------------------------------


def test_past_announcements_are_facts_and_the_next_one_is_flagged(submissions):
    events = event_rows(submissions, CIK, "MSFT")
    confirmed = [e for e in events if not e["is_estimated"]]
    estimated = [e for e in events if e["is_estimated"]]

    assert len(confirmed) == 8, "the configured history window changed"
    assert len(estimated) == 1
    assert all(e["category"] == "earnings" and e["cik"] == CIK for e in events)


def test_the_estimate_carries_the_method_that_produced_it(submissions):
    """A panel showing an estimated date without its method asks to be trusted blindly."""
    estimated = [e for e in event_rows(submissions, CIK, "MSFT") if e["is_estimated"]][0]
    payload = json.loads(estimated["payload"])

    assert "method" in payload and payload["last_confirmed"] == "2026-07-29"
    assert "estimado" in estimated["label"]


def test_the_next_event_has_a_stable_id_so_it_moves_instead_of_piling_up(submissions):
    """One row per company that walks forward, not a trail of stale predictions."""
    estimated = [e for e in event_rows(submissions, CIK, "MSFT") if e["is_estimated"]][0]
    assert estimated["event_id"] == f"{CIK}:earnings:next"


# --- The ingester -------------------------------------------------------------


def settings_with(tmp_path, tracked, secrets=None) -> Settings:
    base = load_settings()
    raw = json.loads(json.dumps(base.raw, default=str))
    raw["universe"]["tracked"] = tracked
    raw["sources"]["sec"]["cache_dir"] = str(tmp_path / "cache")
    return Settings(raw=raw, db_path=tmp_path / "t.db", log_level="INFO",
                    secrets=secrets or {}, public_mode=False)


def test_the_ingester_reads_the_cache_and_fills_both_tables(tmp_path):
    settings = settings_with(tmp_path, [{"ticker": "MSFT", "cik": CIK}],
                             {"SEC_USER_AGENT": "Nombre correo@x.com"})
    ingester = SecFilingsIngester(settings)
    ingester._base_url = "http://127.0.0.1:1"   # any network attempt fails loudly
    cache = tmp_path / "cache" / f"submissions_CIK{CIK}.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    assert ingester.fetch().empty, "filings and events are not time series"
    tables = ingester.fetch_tables()

    assert set(tables) == {"filings", "events", "companies"}
    assert tables["companies"] == [{
        "cik": CIK, "ticker": "MSFT", "name": "MICROSOFT CORP", "sector": None,
        "thesis_category": None, "first_seen": dt.date.today().isoformat(), "status": "active",
    }], "SIC is not GICS: sector stays empty without a written card"
    assert len(tables["filings"]) > 50
    assert ingester.partial_failures() == []


def test_the_tables_upsert_idempotently(tmp_path):
    from db import loader

    settings = settings_with(tmp_path, [{"ticker": "MSFT", "cik": CIK}],
                             {"SEC_USER_AGENT": "Nombre correo@x.com"})
    ingester = SecFilingsIngester(settings)
    ingester._base_url = "http://127.0.0.1:1"
    cache = tmp_path / "cache" / f"submissions_CIK{CIK}.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    conn = loader.init_db(tmp_path / "filings.db")
    try:
        for _ in range(2):
            ingester.fetch()
            tables = ingester.fetch_tables()
            loader.upsert_filings(conn, tables["filings"])
            loader.upsert_events(conn, tables["events"])
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("filings", "events")
        }
        assert counts["filings"] == len(tables["filings"])
        assert counts["events"] == len(tables["events"])
    finally:
        conn.close()


def test_no_tracked_company_produces_no_tables(tmp_path):
    settings = settings_with(tmp_path, [], {"SEC_USER_AGENT": "Nombre correo@x.com"})
    ingester = SecFilingsIngester(settings)
    assert ingester.fetch().empty
    assert ingester.fetch_tables() == {}


# --- Held positions (phase 4: the alerts look at what is held) ---------------------------

# The documented shape of www.sec.gov/files/company_tickers.json, trimmed to three rows.
TICKER_MAP = {
    "0": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
    "2": {"cik_str": 1067983, "ticker": "BRK-A", "title": "BERKSHIRE HATHAWAY INC"},
}


def test_the_ticker_map_pads_the_cik_and_normalises_the_class_separator():
    mapping = ticker_map(TICKER_MAP)
    assert mapping["MSFT"] == CIK
    assert mapping[normalize_ticker("BRK B")] == "0001067983", "IBKR writes 'BRK B'"
    assert mapping[normalize_ticker("BRK.B")] == "0001067983"


def held_database(tmp_path, tickers):
    from db import loader

    conn = loader.init_db(tmp_path / "held.db")
    loader.upsert_observations(conn, pd.DataFrame([
        {"source": "ibkr", "series_id": f"{t}:position_qty", "ts": "2026-09-01",
         "ts_release": "2026-09-01", "value": 1.0} for t in tickers
    ]))
    return conn


def test_a_held_position_without_a_card_is_fetched_too(tmp_path):
    """TMUS and UBER were held with no card, so they had no earnings calendar at all —
    the one thing the phase-4 alert on positions needs."""
    settings = settings_with(tmp_path, [], {"SEC_USER_AGENT": "Nombre correo@x.com"})
    ingester = SecFilingsIngester(settings)
    ingester._base_url = "http://127.0.0.1:1"
    ingester._ticker_map_url = "http://127.0.0.1:1/map"
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "company_tickers.json").write_text(json.dumps(TICKER_MAP), encoding="utf-8")
    (cache / f"submissions_CIK{CIK}.json").write_text(
        FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    conn = held_database(tmp_path, ["MSFT", "SPY"])
    try:
        ingester.attach_database(conn)
    finally:
        conn.close()
    ingester.fetch()

    assert ingester._companies == [(CIK, "MSFT")]
    assert ingester.fetch_tables()["events"], "the held company got its calendar"
    assert ingester.partial_failures() == [], "an ETF without a CIK is not a failure"


def test_a_held_position_already_written_down_is_not_fetched_twice(tmp_path):
    settings = settings_with(tmp_path, [{"ticker": "MSFT", "cik": CIK}],
                             {"SEC_USER_AGENT": "Nombre correo@x.com"})
    ingester = SecFilingsIngester(settings)
    conn = held_database(tmp_path, ["MSFT"])
    try:
        ingester.attach_database(conn)   # no map needed, so no network attempted
    finally:
        conn.close()
    assert ingester._companies == [(CIK, "MSFT")]


def test_without_the_ticker_map_the_run_says_so(tmp_path):
    settings = settings_with(tmp_path, [], {"SEC_USER_AGENT": "Nombre correo@x.com"})
    ingester = SecFilingsIngester(settings)
    ingester._ticker_map_url = "http://127.0.0.1:1/map"
    ingester._retry_kwargs = {"attempts": 1}
    conn = held_database(tmp_path, ["MSFT"])
    try:
        ingester.attach_database(conn)
    finally:
        conn.close()
    assert any("company_tickers" in f for f in ingester.partial_failures())
