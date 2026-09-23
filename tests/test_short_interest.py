"""FINRA consolidated short interest, and the publication date the source does not give.

Against a frozen slice of HIMS's real filings: 14 cycles, two of which FINRA itself marked
as revisions, so the revision branch is exercised with real data rather than a fabrication.

What these tests mostly defend is a **derived** value, which is the uncomfortable part.
Section 4.2 asks this source to respect ``ts_release`` and the source carries none, so the
project computes one — and the whole risk is that a computed publication date lands
*earlier* than the real one, which is look-ahead (section 9.4) and invisible, because the
number itself is correct.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import pytest

from core.config import Settings, load_settings
from ingest.short_interest import (
    DEFAULT_LAG_BUSINESS_DAYS,
    ShortInterestIngester,
    observation_rows,
    publication_date,
)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "finra_short_interest_hims.json"


@pytest.fixture(scope="module")
def cycles() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# --- The derived publication date ---------------------------------------------


def test_the_lag_is_counted_in_business_days_not_calendar_days():
    """2026-08-31 is a Monday; eight business days later is 2026-09-10, not 09-08."""
    assert publication_date("2026-08-31", 8).startswith("2026-09-10")


def test_a_weekend_never_counts_as_a_business_day():
    friday = dt.date(2026, 9, 4)
    assert friday.weekday() == 4
    # One business day after a Friday is the following Monday.
    assert publication_date(friday.isoformat(), 1).startswith("2026-09-07")


def test_the_default_lag_is_the_maximum_observed_not_the_median():
    """The measurement that fixed this number, kept where it can be checked.

    Across FINRA's 28 published cycles the gap was 7 business days in 19 and 8 in 9 — it is
    not constant. Defaulting to 7 would claim a figure was public a day before it was in
    nine cycles out of twenty-eight. A late date only makes the panel conservative; an
    early one is look-ahead (section 9.4).
    """
    assert DEFAULT_LAG_BUSINESS_DAYS == 8

    settlement = "2026-08-31"
    assert publication_date(settlement, 8) > publication_date(settlement, 7)


def test_an_unreadable_settlement_date_yields_nothing_rather_than_today():
    assert publication_date("", 8) is None
    assert publication_date(None, 8) is None
    assert publication_date("not-a-date", 8) is None


def test_the_derived_date_is_always_after_the_settlement_date(cycles):
    """The invariant that makes the whole series safe to read point-in-time."""
    rows, _problems = observation_rows(cycles, "HIMS", DEFAULT_LAG_BUSINESS_DAYS)
    assert rows
    for row in rows:
        assert row["ts_release"] > row["ts"], row


# --- The rows ------------------------------------------------------------------


def test_each_cycle_produces_short_interest_and_days_to_cover(cycles):
    rows, problems = observation_rows(cycles, "HIMS", DEFAULT_LAG_BUSINESS_DAYS)

    assert problems == []
    series = {row["series_id"] for row in rows}
    assert series == {
        "HIMS:short_interest", "HIMS:days_to_cover", "HIMS:short_interest:revised",
    }


def test_the_real_figures_survive_the_parse(cycles):
    """A spot check against the raw fixture, so a silent unit change would be caught."""
    rows, _ = observation_rows(cycles, "HIMS", DEFAULT_LAG_BUSINESS_DAYS)
    latest = max(c["settlementDate"] for c in cycles)
    raw = next(c for c in cycles if c["settlementDate"] == latest)

    stored = next(
        row for row in rows
        if row["series_id"] == "HIMS:short_interest" and row["ts"].startswith(latest)
    )
    assert stored["value"] == float(raw["currentShortPositionQuantity"])
    assert stored["value"] > 1_000_000, "a short interest of a few shares is not credible"


def test_a_revision_is_recorded_even_though_the_original_cannot_be(cycles):
    """The honest half of a limitation that cannot be fixed at this layer.

    FINRA serves one row per settlement date: a revision *replaces* the original instead of
    adding a row, and a derived publication date cannot tell the two apart. So section 9.6's
    "keep every version" is not achievable here. What is achievable is never showing a
    restated figure as if it were the original, and that is what this series is for.
    """
    revised_in_fixture = [
        c for c in cycles if (c.get("revisionFlag") or "").strip().upper() == "R"
    ]
    assert revised_in_fixture, "the fixture no longer covers the revision branch"

    rows, _ = observation_rows(cycles, "HIMS", DEFAULT_LAG_BUSINESS_DAYS)
    flagged = {
        row["ts"][:10] for row in rows
        if row["series_id"].endswith(":revised") and row["value"] == 1.0
    }
    assert flagged == {c["settlementDate"][:10] for c in revised_in_fixture}


def test_an_unrevised_cycle_is_recorded_as_zero_not_as_absent():
    """Absent would be indistinguishable from "not ingested yet" (section 12)."""
    rows, _ = observation_rows(
        [{"settlementDate": "2026-08-31", "currentShortPositionQuantity": "100",
          "daysToCoverQuantity": "2", "revisionFlag": ""}],
        "X", DEFAULT_LAG_BUSINESS_DAYS,
    )
    revised = next(row for row in rows if row["series_id"].endswith(":revised"))
    assert revised["value"] == 0.0


def test_a_cycle_with_no_position_is_reported_not_zeroed():
    rows, problems = observation_rows(
        [{"settlementDate": "2026-08-31", "currentShortPositionQuantity": "",
          "daysToCoverQuantity": "", "revisionFlag": ""}],
        "X", DEFAULT_LAG_BUSINESS_DAYS,
    )
    assert rows == []
    assert "no short position" in problems[0]


def test_a_missing_days_to_cover_drops_only_that_series():
    """One absent field must not cost the short interest itself."""
    rows, problems = observation_rows(
        [{"settlementDate": "2026-08-31", "currentShortPositionQuantity": "5000",
          "daysToCoverQuantity": "", "revisionFlag": ""}],
        "X", DEFAULT_LAG_BUSINESS_DAYS,
    )
    series = {row["series_id"] for row in rows}
    assert "X:short_interest" in series
    assert "X:days_to_cover" not in series
    assert problems == []


# --- The ingester ---------------------------------------------------------------


def settings_with(tmp_path, watchlist) -> Settings:
    base = load_settings()
    raw = json.loads(json.dumps(base.raw, default=str))
    raw["universe"]["tracked"] = []
    raw["universe"]["watchlist"] = watchlist
    raw["sources"]["finra"]["cache_dir"] = str(tmp_path / "cache")
    return Settings(raw=raw, db_path=tmp_path / "t.db", log_level="INFO",
                    secrets={}, public_mode=False)


def test_the_ingester_reads_the_cache_and_honors_the_contract(tmp_path, cycles):
    settings = settings_with(tmp_path, [{"ticker": "HIMS", "cik": "0001773751"}])
    ingester = ShortInterestIngester(settings)
    ingester._url = "http://127.0.0.1:1"      # any network attempt fails loudly
    cache = tmp_path / "cache" / "short_interest_HIMS.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(cycles), encoding="utf-8")

    frame = ingester.fetch()

    assert not frame.empty
    assert list(frame.columns) == ["source", "series_id", "ts", "ts_release", "value"]
    assert (frame["source"] == "finra").all()
    assert ingester.partial_failures() == []


def test_market_reference_etfs_are_not_asked_about(tmp_path):
    """Short interest on a broad ETF is hedging and creation mechanics, not a view."""
    settings = settings_with(tmp_path, [{"ticker": "HIMS", "cik": "0001773751"}])
    assert ShortInterestIngester.tickers_for(settings) == ["HIMS"]
    assert "SPY" not in ShortInterestIngester.tickers_for(settings)


def test_without_a_company_the_ingester_is_unavailable(tmp_path):
    settings = settings_with(tmp_path, [])
    assert ShortInterestIngester.is_available(settings) is False
    assert ShortInterestIngester(settings).fetch().empty


def test_the_rows_upsert_idempotently(tmp_path, cycles):
    from db import loader

    settings = settings_with(tmp_path, [{"ticker": "HIMS", "cik": "0001773751"}])
    ingester = ShortInterestIngester(settings)
    ingester._url = "http://127.0.0.1:1"
    cache = tmp_path / "cache" / "short_interest_HIMS.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(cycles), encoding="utf-8")

    conn = loader.init_db(tmp_path / "si.db")
    try:
        for _ in range(2):
            loader.upsert_observations(conn, ingester.fetch())
        count = conn.execute(
            "SELECT COUNT(*) FROM observations WHERE source = 'finra'"
        ).fetchone()[0]
        assert count == len(ingester.fetch())
    finally:
        conn.close()
