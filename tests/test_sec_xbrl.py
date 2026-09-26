"""SEC XBRL ingestion, against a frozen slice of Microsoft's real companyfacts.

The acceptance criterion of phase 2 is "for a known company, the reconstructed series
match its reported 10-K". That is
:func:`test_reconstructed_annual_revenue_matches_the_reported_10k`, and it is why the
fixture is a real document rather than a hand-written one: the traps this module has to
survive — a company that changed revenue concepts mid-history, year-to-date cumulatives
sharing an end date with quarters, the same period re-reported by four later filings — are
all in Microsoft's real file and none of them would have been invented.

The fixture keeps Microsoft's pre-2011 filings (the ``Revenues`` era) and everything filed
from 2023 on, which is what makes the concept switch testable at all.

The cases the slice cannot contain — a tag nobody configured, a fact in the wrong unit, a
download failure — are synthetic and marked as such. Network is never touched.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import pandas as pd
import pytest

from core.config import Settings, load_settings
from ingest.sec_xbrl import (
    SecXbrlIngester,
    concept_for,
    extract_metric,
)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "sec_companyfacts_msft.json"
CIK = "0000789019"


@pytest.fixture(scope="module")
def facts():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["facts"]


@pytest.fixture(scope="module")
def sec_config():
    cfg = load_settings().source("sec")
    return cfg["concepts"], cfg["duration_windows"]


def records_for(facts, sec_config, metric: str) -> pd.DataFrame:
    concepts, windows = sec_config
    records, _stats = extract_metric(facts, metric, concepts[metric], CIK, windows)
    return pd.DataFrame(records)


# --- The acceptance criterion -------------------------------------------------


def test_reconstructed_annual_revenue_matches_the_reported_10k(facts, sec_config):
    """Microsoft's own reported figures, to the dollar.

    FY2024 revenue was 245.122 billion. If the concept preference, the duration window or
    the unit filter were wrong, this is where it would show — those are the three ways to
    build a revenue series that looks plausible and is not.
    """
    frame = records_for(facts, sec_config, "revenue")
    annual = frame[frame["series_id"] == f"{CIK}:revenue:fy"]
    latest = annual.sort_values("ts_release").groupby("ts", as_index=False).last()
    by_year = dict(zip(latest["ts"], latest["value"]))

    assert by_year["2023-06-30"] == 211_915_000_000
    assert by_year["2024-06-30"] == 245_122_000_000
    assert by_year["2025-06-30"] == 281_724_000_000


def _fiscal_2024(facts, sec_config) -> tuple[pd.DataFrame, float]:
    frame = records_for(facts, sec_config, "revenue")
    latest = frame.sort_values("ts_release").groupby(["series_id", "ts"], as_index=False).last()
    quarters = latest[
        (latest["series_id"] == f"{CIK}:revenue:q")
        & (latest["ts"] > "2023-06-30") & (latest["ts"] <= "2024-06-30")
    ]
    annual = latest[
        (latest["series_id"] == f"{CIK}:revenue:fy") & (latest["ts"] == "2024-06-30")
    ]["value"].iloc[0]
    return quarters, float(annual)


def test_a_fiscal_year_only_ever_reports_three_quarters(facts, sec_config):
    """An accounting convention worth stating: **no company files a Q4 10-Q**.

    The fourth quarter exists only inside the 10-K, which reports the full year. So a
    company-year has three reported quarters and one annual figure, and Q4 has to be
    *derived* as FY − Q1 − Q2 − Q3. That derivation belongs in transform/, explicitly and
    once; doing it here would bury an assumption inside what is supposed to be raw
    ingestion. A panel that plots "quarterly revenue" straight from XBRL silently drops
    every fourth bar.
    """
    quarters, _annual = _fiscal_2024(facts, sec_config)
    assert len(quarters) == 3
    assert list(quarters["ts"]) == ["2023-09-30", "2023-12-31", "2024-03-31"]


def test_the_reported_quarters_never_exceed_the_year(facts, sec_config):
    """The arithmetic guard against a year-to-date cumulative slipping into ``:q``.

    companyfacts mixes 3-, 6-, 9- and 12-month figures under one concept, all sharing an
    end date. If a 9-month cumulative had been filed as a quarter, this sum would blow
    past the annual figure — and a revenue chart would still look perfectly fine.
    """
    quarters, annual = _fiscal_2024(facts, sec_config)
    reported = quarters["value"].sum()
    implied_q4 = annual - reported

    assert reported < annual, "the quarterly series contains a cumulative"
    assert 0.2 <= implied_q4 / annual <= 0.35, "the implied fourth quarter is not plausible"


# --- Point-in-time and restatements (sections 9.4, 9.6) -----------------------


def test_every_value_carries_the_date_it_became_public(facts, sec_config):
    frame = records_for(facts, sec_config, "revenue")
    assert frame["ts_release"].notna().all()
    assert (frame["ts_release"] >= frame["ts"]).all(), "a fact cannot be filed before it ends"


def test_the_filing_lag_is_real_and_visible(facts, sec_config):
    """Microsoft's FY ends 30 June and the 10-K lands a month later.

    Assigning that figure to "June" in a backtest hands it a month of the future. The gap
    is the whole reason ``ts_release`` exists (section 9.4).
    """
    frame = records_for(facts, sec_config, "revenue")
    fy2024 = frame[
        (frame["series_id"] == f"{CIK}:revenue:fy") & (frame["ts"] == "2024-06-30")
    ]
    first_filed = fy2024["ts_release"].min()
    lag = (dt.date.fromisoformat(first_filed) - dt.date(2024, 6, 30)).days
    assert 20 <= lag <= 90, f"unexpected filing lag of {lag} days"


def test_a_period_reported_again_keeps_every_version(facts, sec_config):
    """Section 9.6: later filings re-report old periods; all versions must survive.

    The panel shows the latest; a backtest uses the one in force on its simulated date.
    Overwriting would destroy the ability to ask what was knowable back then.
    """
    frame = records_for(facts, sec_config, "revenue")
    fy2024 = frame[
        (frame["series_id"] == f"{CIK}:revenue:fy") & (frame["ts"] == "2024-06-30")
    ]
    assert fy2024["ts_release"].nunique() >= 2, "only one vintage survived"


# --- Taxonomy normalization (section 8, phase 2 point 2) ----------------------


def test_the_concept_behind_each_number_is_recorded_and_decodable(facts, sec_config):
    """Microsoft changed revenue tags with ASC 606; the series must say so.

    Without this record, a step in the series is indistinguishable from a company that
    changed tags — and that is exactly the question an analyst asks first (section 9.8).
    """
    concepts, _ = sec_config
    frame = records_for(facts, sec_config, "revenue")
    provenance = frame[frame["series_id"] == f"{CIK}:revenue:fy:{'src'}"].set_index("ts")

    old = concept_for(concepts["revenue"], provenance.loc["2010-06-30", "value"].max())
    new = concept_for(concepts["revenue"], provenance.loc["2024-06-30", "value"].max())
    assert old == "Revenues"
    assert new == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_every_value_has_a_provenance_row(facts, sec_config):
    frame = records_for(facts, sec_config, "revenue")
    values = frame[~frame["series_id"].str.endswith(":src")]
    sources = frame[frame["series_id"].str.endswith(":src")]
    assert len(values) == len(sources)


def test_a_later_preference_only_fills_the_gaps_the_first_one_left(sec_config):
    """Synthetic: one continuous series, not two half ones."""
    _concepts, windows = sec_config
    spec = {"kind": "duration", "unit": "USD", "tags": ["Preferred", "Fallback"]}
    facts = {"us-gaap": {
        "Preferred": {"units": {"USD": [
            {"start": "2025-01-01", "end": "2025-12-31", "val": 100, "filed": "2026-02-01"},
        ]}},
        "Fallback": {"units": {"USD": [
            # Same period: must lose to the preferred tag.
            {"start": "2025-01-01", "end": "2025-12-31", "val": 999, "filed": "2026-02-01"},
            # A period the preferred tag never covered: must be used.
            {"start": "2024-01-01", "end": "2024-12-31", "val": 80, "filed": "2025-02-01"},
        ]}},
    }}
    records, _ = extract_metric(facts, "revenue", spec, CIK, windows)
    frame = pd.DataFrame(records).set_index(["series_id", "ts"])

    assert frame.loc[(f"{CIK}:revenue:fy", "2025-12-31"), "value"] == 100
    assert frame.loc[(f"{CIK}:revenue:fy", "2024-12-31"), "value"] == 80
    assert frame.loc[(f"{CIK}:revenue:fy:src", "2024-12-31"), "value"] == 1


def test_an_unknown_tag_yields_no_series_rather_than_a_zero(sec_config):
    """Section 12: a metric nobody can resolve is empty, never estimated."""
    _concepts, windows = sec_config
    spec = {"kind": "duration", "unit": "USD", "tags": ["NotThisOne"]}
    facts = {"us-gaap": {"SomethingElse": {"units": {"USD": [
        {"start": "2025-01-01", "end": "2025-12-31", "val": 100, "filed": "2026-02-01"},
    ]}}}}
    records, stats = extract_metric(facts, "revenue", spec, CIK, windows)
    assert records == []
    assert stats["wrong_unit"] == 0


def test_a_fact_in_another_unit_is_counted_not_mixed_in(sec_config):
    """Mixing units in one series is the silent way to lie by a factor of 1.1."""
    _concepts, windows = sec_config
    spec = {"kind": "duration", "unit": "USD", "tags": ["Revenues"]}
    facts = {"us-gaap": {"Revenues": {"units": {
        "EUR": [{"start": "2025-01-01", "end": "2025-12-31", "val": 90, "filed": "2026-02-01"}],
        "USD": [{"start": "2025-01-01", "end": "2025-12-31", "val": 100, "filed": "2026-02-01"}],
    }}}}
    records, stats = extract_metric(facts, "revenue", spec, CIK, windows)
    values = [r["value"] for r in records if not r["series_id"].endswith(":src")]
    assert values == [100]
    assert stats["wrong_unit"] == 1


def test_year_to_date_cumulatives_are_kept_but_never_as_quarters(facts, sec_config):
    """Cumulatives are labelled, not discarded — and labelled is what keeps them safe.

    A 6- or 9-month figure filed into the quarterly series would be two or three times too
    large and would look entirely plausible. Kept under their own suffix they are useful
    instead of dangerous: they are the only way to recover the quarters a company never
    files on their own (RESEARCH.md section 2.21).
    """
    concepts, windows = sec_config
    records, _stats = extract_metric(facts, "revenue", concepts["revenue"], CIK, windows)
    periods = {row["series_id"].split(":")[-1] for row in records
               if not row["series_id"].endswith(":src")}

    assert {"ytd2", "ytd3"} <= periods, "the cumulatives vanished"

    quarters = {row["ts"] for row in records if row["series_id"] == f"{CIK}:revenue:q"}
    cumulative = {row["ts"] for row in records if row["series_id"] == f"{CIK}:revenue:ytd2"}
    for ts in quarters & cumulative:
        q = next(r["value"] for r in records
                 if r["series_id"] == f"{CIK}:revenue:q" and r["ts"] == ts)
        ytd = next(r["value"] for r in records
                   if r["series_id"] == f"{CIK}:revenue:ytd2" and r["ts"] == ts)
        assert q != ytd, f"a cumulative was filed as a quarter on {ts}"


def test_a_duration_matching_no_window_is_still_discarded_and_counted(sec_config):
    """Synthetic: a 45-day stub period belongs to no bucket and must not be guessed at."""
    _concepts, windows = sec_config
    facts = {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2025-01-01", "end": "2025-02-15", "val": 1.0, "filed": "2025-03-01"},
    ]}}}}
    records, stats = extract_metric(
        facts, "revenue", {"tags": ["Revenues"], "unit": "USD"}, CIK, windows
    )

    assert records == []
    assert stats["unclassified_period"] == 1


def test_balance_sheet_items_are_instants_with_no_period_suffix(facts, sec_config):
    frame = records_for(facts, sec_config, "assets")
    series = set(frame["series_id"])
    assert series == {f"{CIK}:assets", f"{CIK}:assets:src"}


# --- The ingester -------------------------------------------------------------


def settings_with(tmp_path, tracked, secrets=None) -> Settings:
    base = load_settings()
    raw = json.loads(json.dumps(base.raw, default=str))
    raw["universe"]["tracked"] = tracked
    raw["sources"]["sec"]["cache_dir"] = str(tmp_path / "cache")
    return Settings(raw=raw, db_path=tmp_path / "t.db", log_level="INFO",
                    secrets=secrets or {}, public_mode=False)


def test_a_thesis_card_without_a_cik_is_skipped_not_guessed(tmp_path):
    """Section 9.3: the CIK is the key. A missing one is not a ticker to look up here."""
    settings = settings_with(tmp_path, [{"ticker": "MSFT"}, {"ticker": "T", "cik": "732717"}])
    assert SecXbrlIngester.companies_for(settings) == [("0000732717", "T")]


def test_availability_needs_both_a_user_agent_and_a_company(tmp_path):
    card = [{"ticker": "MSFT", "cik": CIK}]
    assert not SecXbrlIngester.is_available(settings_with(tmp_path, card))
    assert not SecXbrlIngester.is_available(
        settings_with(tmp_path, [], {"SEC_USER_AGENT": "Nombre correo@x.com"})
    )
    assert SecXbrlIngester.is_available(
        settings_with(tmp_path, card, {"SEC_USER_AGENT": "Nombre correo@x.com"})
    )


def test_a_fresh_cache_file_is_used_without_touching_the_network(tmp_path):
    """Section 4.4: companyfacts only changes when a filing lands. Re-downloading it
    daily for twenty companies is rude to a public service and slow for no gain."""
    settings = settings_with(tmp_path, [{"ticker": "MSFT", "cik": CIK}],
                             {"SEC_USER_AGENT": "Nombre correo@x.com"})
    ingester = SecXbrlIngester(settings)
    cache = tmp_path / "cache" / f"companyfacts_CIK{CIK}.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    # An unusable base URL: any attempt to reach the network fails the test loudly.
    ingester._base_url = "http://127.0.0.1:1"
    frame = ingester.fetch()

    assert not frame.empty
    assert f"{CIK}:revenue:fy" in set(frame["series_id"])
    assert ingester.partial_failures() == []


def test_an_expired_cache_is_used_on_failure_and_reported(tmp_path):
    """Stale audited data beats no data — as long as nobody is told it is fresh."""
    import os

    settings = settings_with(tmp_path, [{"ticker": "MSFT", "cik": CIK}],
                             {"SEC_USER_AGENT": "Nombre correo@x.com"})
    ingester = SecXbrlIngester(settings)
    ingester._base_url = "http://127.0.0.1:1"
    ingester._retry_kwargs = {"attempts": 1, "base_delay_s": 0, "max_delay_s": 0}

    cache = tmp_path / "cache" / f"companyfacts_CIK{CIK}.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    old = dt.datetime.now().timestamp() - 60 * 60 * 24 * 30
    os.utime(cache, (old, old))

    frame = ingester.fetch()

    assert not frame.empty, "the expired copy should still have been used"
    assert any("expired cache" in f for f in ingester.partial_failures())


def test_no_tracked_company_means_an_empty_frame_not_a_crash(tmp_path):
    settings = settings_with(tmp_path, [], {"SEC_USER_AGENT": "Nombre correo@x.com"})
    frame = SecXbrlIngester(settings).fetch()
    assert frame.empty
    assert list(frame.columns) == ["source", "series_id", "ts", "ts_release", "value"]


def test_a_held_position_without_a_card_is_fetched_too(tmp_path):
    """The panel values what is held; TMUS and UBER had no fundamentals until 2026-09-25."""
    from core import config
    from db import loader
    from ingest.sec_xbrl import SecXbrlIngester

    conn = loader.init_db(tmp_path / "held.db")
    loader.upsert_observations(conn, pd.DataFrame([
        {"source": "ibkr", "series_id": "NAV:total", "ts": "2026-09-15",
         "ts_release": "2026-09-15", "value": 100.0},
        {"source": "ibkr", "series_id": "TMUS:position_qty", "ts": "2026-09-15",
         "ts_release": "2026-09-15", "value": 1.0}]))
    loader.upsert_companies(conn, [{"cik": "0001283699", "ticker": "TMUS", "name": "T-Mobile",
                                    "sector": None, "thesis_category": None,
                                    "first_seen": "2026-09-15", "status": "active"}])
    ingester = SecXbrlIngester(config.load_settings())
    ingester.attach_database(conn)
    assert ("0001283699", "TMUS") in ingester._companies
    conn.close()
