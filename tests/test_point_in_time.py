"""Point-in-time discipline — a CI-blocking test (CLAUDE.md sections 9.4, 10).

The rule these tests defend: a series reconstructed for a past date may contain nothing
published after that date. Filtering by ``ts`` instead of ``ts_release`` produces a series
that looks right, plots right, and is wrong by exactly the publication lag — measured on
2026-08-28 against the live FRED API as ~45 days for CPI and ~59 for core PCE.
"""

from __future__ import annotations

import pandas as pd
import pytest

from transform import macro


def _obs(series_id, ts, ts_release, value):
    return {"source": "fred", "series_id": series_id, "ts": ts,
            "ts_release": ts_release, "value": value}


@pytest.fixture()
def cpi() -> pd.DataFrame:
    """Three CPI prints with the real publication lag: reference month, filed ~6 weeks on."""
    return pd.DataFrame([
        _obs("CPIAUCSL", "2026-04-01", "2026-05-12", 320.1),
        _obs("CPIAUCSL", "2026-05-01", "2026-06-10", 321.4),
        _obs("CPIAUCSL", "2026-06-01", "2026-07-15", 322.9),
    ])


def test_a_datum_is_invisible_before_it_is_published(cpi):
    """On 2026-07-01 the June CPI does not exist yet, though its ts says June."""
    reading = macro.latest(cpi, "CPIAUCSL", "2026-07-01")
    assert reading.ts == "2026-05-01", "the June print leaked before its publication date"
    assert reading.value == 321.4


def test_the_datum_appears_on_its_release_date(cpi):
    """And it becomes visible exactly on ts_release, not before and not after."""
    assert macro.latest(cpi, "CPIAUCSL", "2026-07-14").ts == "2026-05-01"
    assert macro.latest(cpi, "CPIAUCSL", "2026-07-15").ts == "2026-06-01"


def test_no_row_published_after_the_simulated_date_survives(cpi):
    """The general invariant, stated directly: nothing in the slice postdates as_of."""
    for as_of in ("2026-05-01", "2026-06-15", "2026-07-20", "2026-12-31"):
        known = macro.point_in_time(cpi, as_of)
        assert (known["ts_release"] <= as_of).all(), f"look-ahead leaked at {as_of}"


def test_nothing_known_yet_is_a_visible_hole_not_a_zero(cpi):
    """Before the first publication the reading is None — never 0.0 (section 12)."""
    reading = macro.latest(cpi, "CPIAUCSL", "2026-01-01")
    assert reading.value is None
    assert reading.ts is None
    assert reading.staleness_days is None


def test_an_unknown_series_returns_a_hole_not_an_exception(cpi):
    """Transforms report missing input as None, they do not raise (section 10)."""
    reading = macro.latest(cpi, "NOT_A_SERIES", "2026-08-01")
    assert reading.value is None


# --- Restatements: every version is kept, the vigent one is chosen per date -----------


@pytest.fixture()
def restated() -> pd.DataFrame:
    """One reference quarter published twice: an original print and a later revision."""
    return pd.DataFrame([
        _obs("PAYEMS", "2026-03-01", "2026-04-03", 159_000.0),   # first print
        _obs("PAYEMS", "2026-03-01", "2026-05-08", 157_200.0),   # revised down
    ])


def test_the_original_print_stands_until_the_revision_is_published(restated):
    """Backtesting March in April must use the number April had, not May's correction."""
    assert macro.latest(restated, "PAYEMS", "2026-04-20").value == 159_000.0


def test_the_revision_supersedes_it_from_its_own_release_date(restated):
    """From May onwards the revised figure is the vigent one for that same reference date."""
    assert macro.latest(restated, "PAYEMS", "2026-05-20").value == 157_200.0


def test_both_versions_survive_the_slice(restated):
    """The slice collapses to one row per (series_id, ts) — the newest *knowable* one."""
    known = macro.point_in_time(restated, "2026-06-01")
    assert len(known) == 1
    assert known.iloc[0]["value"] == 157_200.0


# --- Staleness and changes ------------------------------------------------------------


def test_staleness_is_reported_so_a_stale_print_is_not_read_as_today(cpi):
    """A six-week-old CPI is fine; presenting it as today's inflation is not."""
    reading = macro.latest(cpi, "CPIAUCSL", "2026-08-01")
    assert reading.ts == "2026-06-01"
    assert reading.staleness_days == 61


def test_change_is_none_when_the_history_does_not_reach_back(cpi):
    """No comparison point means None — never a change measured against the first value."""
    assert macro.change_over(cpi, "CPIAUCSL", "2026-05-20", lookback_days=365) is None


def test_change_uses_the_value_that_was_public_at_both_ends(cpi):
    """Both sides of the comparison are taken point-in-time, not just the current one."""
    # On 2026-07-20 the public value is June's (322.9); 60 days earlier, on 2026-05-21,
    # it was April's (320.1) — the May print did not exist until 2026-06-10.
    change = macro.change_over(cpi, "CPIAUCSL", "2026-07-20", lookback_days=60)
    assert change == pytest.approx(322.9 - 320.1)


def test_snapshot_keeps_a_row_for_a_series_with_no_data(cpi):
    """A missing series shows as a hole in the table, it does not vanish from it."""
    readings = macro.snapshot(cpi, ["CPIAUCSL", "NFCI"], "2026-08-01")
    assert [r.series_id for r in readings] == ["CPIAUCSL", "NFCI"]
    assert readings[1].value is None


def test_empty_input_does_not_raise():
    """The nothing-ingested-yet path returns empty, it does not crash the panel."""
    empty = pd.DataFrame(columns=["source", "series_id", "ts", "ts_release", "value"])
    assert macro.point_in_time(empty, "2026-08-01").empty
    assert macro.latest(empty, "CPIAUCSL", "2026-08-01").value is None
