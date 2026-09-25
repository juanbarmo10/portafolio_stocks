"""The macro calendar: official dates, in UTC, and no ghosts (section 8 phase 4).

The FRED payload is a captured response (release 10, CPI, asked on 2026-09-24); it carries
no key.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import pandas as pd

from ingest.macro_calendar import (
    current_calendar,
    fomc_calendar_problem,
    fomc_events,
    release_events,
    release_instant,
)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "fred_release_dates_cpi.json"
TODAY = dt.date(2026, 9, 24)


def cpi(horizon_days=60, today=TODAY):
    return release_events(
        json.loads(FIXTURE.read_text(encoding="utf-8")), release_id=10, label="CPI",
        time_et="08:30", today=today, horizon_days=horizon_days,
    )


def test_the_captured_schedule_becomes_events_within_the_horizon():
    rows = cpi()
    assert [r["ts"][:10] for r in rows] == ["2026-10-14", "2026-11-10"]
    assert all(r["is_estimated"] == 0 for r in rows), "an agency schedule is not an estimate"
    assert rows[0]["event_id"] == "fred:10:2026-10-14", "stable id, so a re-run updates"


def test_the_release_hour_is_stored_in_utc_across_daylight_saving():
    """08:30 New York is 12:30 UTC in October and 13:30 after the clocks change on
    1 November. A date without the hour could not answer "within 48 hours"."""
    first, second = cpi()
    assert first["ts"] == "2026-10-14T12:30:00+00:00"
    assert second["ts"] == "2026-11-10T13:30:00+00:00"
    assert release_instant(dt.date(2026, 12, 9), "14:00") == "2026-12-09T19:00:00+00:00"


def test_past_dates_and_dates_beyond_the_horizon_are_left_out():
    assert cpi(horizon_days=10) == []
    assert [r["ts"][:10] for r in cpi(today=dt.date(2026, 10, 15))] == ["2026-11-10", "2026-12-10"]


def test_fomc_dates_accept_yaml_dates_and_text():
    rows = fomc_events([dt.date(2026, 10, 28), "2026-12-09", "2026-09-16"], time_et="14:00",
                       today=TODAY, horizon_days=90, source_url="https://fed")
    assert [r["event_id"] for r in rows] == ["fomc:2026-10-28", "fomc:2026-12-09"]
    assert rows[0]["category"] == "fomc"


def test_an_exhausted_fomc_list_is_reported_a_horizon_ahead():
    """Said while there is still time to add next year, not on the day it runs out."""
    assert fomc_calendar_problem(["2026-12-09"], TODAY, 60) is None
    assert "ends on 2026-12-09" in fomc_calendar_problem(["2026-12-09"], dt.date(2026, 10, 20), 60)
    assert fomc_calendar_problem([], TODAY, 60) == "no FOMC dates configured"


def test_a_rescheduled_release_leaves_no_ghost():
    """The loader never deletes. A date the newest calendar no longer lists was moved,
    and it must stop counting — otherwise the alert fires for a release that isn't coming."""
    old = {"event_id": "fred:10:2026-10-14", "category": "macro", "ts": "2026-10-14T12:30:00+00:00",
           "payload": json.dumps({"calendar_as_of": "2026-09-20", "release_id": 10})}
    moved = {"event_id": "fred:10:2026-10-21", "category": "macro", "ts": "2026-10-21T12:30:00+00:00",
             "payload": json.dumps({"calendar_as_of": "2026-09-24", "release_id": 10})}
    fomc = {"event_id": "fomc:2026-10-28", "category": "fomc", "ts": "2026-10-28T18:00:00+00:00",
            "payload": json.dumps({"calendar_as_of": "2026-09-22"})}
    earnings = {"event_id": "x", "category": "earnings", "ts": "2026-10-01", "payload": "{}"}

    kept = current_calendar(pd.DataFrame([old, moved, fomc, earnings]))
    assert set(kept["event_id"]) == {"fred:10:2026-10-21", "fomc:2026-10-28"}, (
        "each calendar is judged against its own newest run, and earnings are not calendar rows"
    )


def test_one_release_failing_today_keeps_the_others_dates():
    """PCE failed to download today, so its rows are still from yesterday's run. They are
    still the newest PCE calendar there is, and still count."""
    cpi_today = {"event_id": "fred:10:2026-10-14", "category": "macro", "ts": "2026-10-14",
                 "payload": json.dumps({"calendar_as_of": "2026-09-24", "release_id": 10})}
    pce_yesterday = {"event_id": "fred:54:2026-09-30", "category": "macro", "ts": "2026-09-30",
                     "payload": json.dumps({"calendar_as_of": "2026-09-23", "release_id": 54})}
    kept = current_calendar(pd.DataFrame([cpi_today, pce_yesterday]))
    assert len(kept) == 2
