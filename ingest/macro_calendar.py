"""Macro calendar: the upcoming high-impact releases and FOMC decisions (sections 2, 8 phase 4).

Level 1 of the checklist asks one question before every contribution: *is there CPI, PCE,
payrolls or an FOMC decision in the next few days?* A release of that weight can move the
whole market on the day, and buying the afternoon before it is a bet nobody decided to
make. The phase-4 alert "high-impact macro release in < 48 h → do not execute the
contribution today" reads what this module writes to ``events``.

Two sources, both official, neither scraped:

- **CPI, PCE and payrolls** come from FRED's ``release/dates`` endpoint, which publishes the
  statistical agencies' own schedule, future dates included. Verified 2026-09-24: CPI
  (release 10), PCE (54) and the Employment Situation (50) return their next dates.
- **FOMC decisions come from config**, copied from the Federal Reserve's calendar page and
  dated when verified. FRED has no schedule for them — release 101, *FOMC Press Release*,
  answers every day of the year, and 326 is only the four projection meetings. The Fed
  fixes its meetings a year ahead, so a list in config is sturdier than scraping the page
  (section 4.3); when the list runs out the ingester says so through the exit code.

**Times are stored in UTC** (section 10). The rule counts hours, and "the 14th" is not
precise enough for that: CPI lands at 08:30 New York time, an FOMC statement at 14:00. The
hour lives in config per release; zoneinfo turns it into UTC including daylight saving.

**A rescheduled release must not leave a ghost.** The loader never deletes, so if a date
moves — a government shutdown moved several in 2025 — the old row would stay in ``events``
and keep firing. Every row carries the date of the calendar that produced it
(``calendar_as_of`` in the payload); readers trust only the most recent calendar. See
:func:`current_calendar`.

fetch() -> empty observations; fetch_tables() -> {"events": [...]}
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from core.config import Settings
from core.logging_setup import get_logger
from ingest.base import Ingester, empty_observations, retry

log = get_logger(__name__)

SOURCE = "macro_calendar"
RELEASE_DATES_URL = "{base}/release/dates"
NEW_YORK = ZoneInfo("America/New_York")


def release_instant(day: dt.date, time_et: str) -> str:
    """ISO8601 UTC instant of a release published at ``time_et`` New York time."""
    hour, minute = (int(part) for part in str(time_et).split(":"))
    local = dt.datetime.combine(day, dt.time(hour, minute), tzinfo=NEW_YORK)
    return local.astimezone(dt.timezone.utc).isoformat()


def _as_date(value: Any) -> dt.date:
    """A YAML date arrives as ``date`` or as text, depending on quoting."""
    return value if isinstance(value, dt.date) else dt.date.fromisoformat(str(value))


def release_events(
    payload: Mapping[str, Any],
    *,
    release_id: int,
    label: str,
    time_et: str,
    today: dt.date,
    horizon_days: int,
) -> list[dict[str, Any]]:
    """Event rows from one FRED ``release/dates`` payload, within ``[today, today+horizon]``.

    The id is ``fred:<release>:<date>``, so a re-run updates the row instead of adding one.
    """
    last = today + dt.timedelta(days=horizon_days)
    rows: list[dict[str, Any]] = []
    for entry in payload.get("release_dates", []):
        try:
            day = dt.date.fromisoformat(str(entry.get("date")))
        except ValueError:
            continue
        if not today <= day <= last:
            continue
        rows.append(_event(f"fred:{release_id}:{day}", "macro", label, day, time_et, today, {
            "source": "FRED release/dates", "release_id": release_id,
        }))
    return rows


def fomc_events(
    dates: Sequence[Any], *, time_et: str, today: dt.date, horizon_days: int, source_url: str
) -> list[dict[str, Any]]:
    """Event rows for the configured FOMC decision days within the horizon."""
    last = today + dt.timedelta(days=horizon_days)
    return [
        _event(f"fomc:{day}", "fomc", "FOMC (decisión de tipos)", day, time_et, today,
               {"source": source_url})
        for day in sorted(_as_date(d) for d in dates)
        if today <= day <= last
    ]


def _event(
    event_id: str, category: str, label: str, day: dt.date, time_et: str, today: dt.date,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "category": category,
        "cik": None,
        "ts": release_instant(day, time_et),
        # Both schedules are official announcements, not estimates.
        "is_estimated": 0,
        "label": label,
        "payload": json.dumps({**extra, "time_et": time_et, "calendar_as_of": today.isoformat()}),
    }


def fomc_calendar_problem(dates: Sequence[Any], today: dt.date, horizon_days: int) -> str | None:
    """Why the configured FOMC list can no longer be trusted, or ``None``.

    Raised a horizon ahead of the last date, not on the day it runs out: by then the alert
    would already have gone quiet on a meeting nobody wrote down.
    """
    if not dates:
        return "no FOMC dates configured"
    last = max(_as_date(d) for d in dates)
    if last < today + dt.timedelta(days=horizon_days):
        return f"the configured FOMC calendar ends on {last}; add next year's dates"
    return None


def current_calendar(events: pd.DataFrame) -> pd.DataFrame:
    """The macro and FOMC rows written by the most recent run of *their own* calendar.

    A row from an older run whose date the newer run no longer lists was rescheduled or
    removed; it is dropped here rather than deleted from the table. Judged per calendar —
    each FRED release, and the FOMC list — so a day on which one release failed to
    download does not throw away the others' valid dates.
    """
    if events.empty:
        return events
    rows = events[events["category"].isin(["macro", "fomc"])].copy()
    if rows.empty:
        return rows
    payloads = rows["payload"].map(lambda raw: json.loads(raw or "{}"))
    rows["_as_of"] = payloads.map(lambda p: p.get("calendar_as_of", ""))
    rows["_calendar"] = [
        f"{category}:{p.get('release_id', '')}" for category, p in zip(rows["category"], payloads)
    ]
    newest = rows.groupby("_calendar")["_as_of"].transform("max")
    return rows[rows["_as_of"] == newest].drop(columns=["_as_of", "_calendar"])


class MacroCalendarIngester(Ingester):
    """Upcoming CPI, PCE, payrolls (FRED) and FOMC decisions (config) → ``events``."""

    source = SOURCE

    def __init__(self, settings: Settings) -> None:
        fred = settings.source("fred")
        cfg = settings.source("macro_calendar")
        self._base_url = str(fred.get("base_url", "https://api.stlouisfed.org/fred")).rstrip("/")
        self._api_key = settings.secret("FRED_API_KEY")
        self._timeout = int(fred.get("request_timeout_s", 30))
        self._horizon = int(cfg.get("horizon_days", 60))
        self._releases = list(cfg.get("releases", []))
        self._fomc = dict(cfg.get("fomc", {}))
        self._today = dt.datetime.now(dt.timezone.utc).date()
        self._events: list[dict[str, Any]] = []
        self._failures: list[str] = []

    @staticmethod
    def is_available(settings: Settings) -> bool:
        """Needs the FRED key, like the rest of the macro block."""
        return bool(settings.secret("FRED_API_KEY"))

    def _release_dates(self, release_id: int) -> dict[str, Any]:
        params = {
            "release_id": release_id,
            "api_key": self._api_key,
            "file_type": "json",
            # Without this FRED lists only dates that already have data: the past.
            "include_release_dates_with_no_data": "true",
            "realtime_start": self._today.isoformat(),
            "sort_order": "asc",
            "limit": 24,
        }

        def call() -> dict[str, Any]:
            response = requests.get(
                RELEASE_DATES_URL.format(base=self._base_url), params=params,
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json()

        return retry(call, exceptions=(requests.RequestException,))

    def fetch(self) -> pd.DataFrame:
        events: list[dict[str, Any]] = []
        for release in self._releases:
            release_id, label = int(release["release_id"]), str(release["label"])
            try:
                payload = self._release_dates(release_id)
            except Exception as exc:
                # The key never reaches the log: requests puts it in the URL, and the
                # exception text carries the URL. Only the release id is named.
                log.error("FRED release %d (%s) failed: %s", release_id, label,
                          type(exc).__name__, extra={"source": SOURCE})
                self._failures.append(f"release {release_id} ({label})")
                continue
            rows = release_events(
                payload, release_id=release_id, label=label,
                time_et=str(release.get("time_et", "08:30")), today=self._today,
                horizon_days=self._horizon,
            )
            if not rows:
                # FRED always lists a few months ahead for these three; none at all means
                # the endpoint changed or the release was discontinued — not "quiet month".
                self._failures.append(f"release {release_id} ({label}): no upcoming dates")
            events.extend(rows)

        dates = list(self._fomc.get("decision_dates", []))
        events.extend(fomc_events(
            dates, time_et=str(self._fomc.get("time_et", "14:00")), today=self._today,
            horizon_days=self._horizon, source_url=str(self._fomc.get("source_url", "")),
        ))
        problem = fomc_calendar_problem(dates, self._today, self._horizon)
        if problem:
            self._failures.append(f"FOMC: {problem} ({self._fomc.get('source_url', '')})")

        log.info("Macro calendar: %d upcoming event(s) within %d days: %s", len(events),
                 self._horizon, ", ".join(f"{e['label']} {e['ts'][:10]}" for e in events),
                 extra={"source": SOURCE})
        self._events = events
        return empty_observations()

    def fetch_tables(self) -> dict[str, list[dict[str, Any]]]:
        return {"events": self._events} if self._events else {}

    def partial_failures(self) -> list[str]:
        return list(self._failures)
