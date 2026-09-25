"""Level 1 of the checklist — "is there risk appetite?" (CLAUDE.md sections 2, 8 phase 1).

Pure transformations over the long-format ``observations`` frame: no network, no database
handle, no hidden state (section 10). A missing input yields ``None``, never an estimate
and never an exception (section 12).

**The point-in-time rule lives here.** Every reading is computed from the rows knowable at
a given date, selected by ``ts_release`` and never by ``ts`` (section 9.4). The FRED lags
measured on 2026-08-28 make the size of the trap concrete: the CPI print for a month is
not public for ~45 days, core PCE for ~59, and payrolls for ~35. Filtering by ``ts`` would
hand a backtest a month and a half of information nobody had — and the bias is systematic,
always in favor of the simulated strategy, never against it.

This module reports levels, changes and staleness. It does **not** score, rank or combine
them into a verdict: that is ``transform/regime.py``, with fixed equal weights and rules
written in config before any result was seen (section 9.7).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import pandas as pd

# Rows whose publication date is unknown are stored with this sentinel rather than NULL,
# so that it stays usable inside the primary key (RESEARCH.md section 1.1). It sorts
# before any ISO8601 date, which is the right semantics for "known all along".
TS_RELEASE_UNKNOWN = ""


@dataclass(frozen=True)
class Reading:
    """One macro series as it was knowable on a given date.

    Attributes:
        series_id: FRED code, e.g. ``"BAMLH0A0HYM2"``.
        value: Latest value publicly known at ``as_of``; ``None`` if nothing was.
        ts: Reference date of that value (the period it describes).
        ts_release: Date that value was first published.
        staleness_days: Days between ``ts`` and ``as_of``. A weekly or monthly series is
            legitimately stale; the panel shows this so a six-week-old CPI print is not
            read as today's inflation.
        change: Change in level versus ``lookback_days`` earlier, or ``None`` when there
            is no comparable observation. Absolute change in the series' own units — a
            percentage change on a spread already expressed in percent would be a second
            unit nobody asked for.
        lookback_days: Window the change was measured over.
    """

    series_id: str
    value: float | None
    ts: str | None
    ts_release: str | None
    staleness_days: int | None
    change: float | None = None
    lookback_days: int | None = None


def _as_date(value: Any) -> dt.date:
    """Coerce an ISO8601 string, date or datetime to a plain date."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def point_in_time(df: pd.DataFrame, as_of: Any) -> pd.DataFrame:
    """Restrict ``df`` to what was publicly known on ``as_of``.

    Two filters, and both matter:

    1. Drop every row published after ``as_of`` (``ts_release > as_of``). This is the
       look-ahead guard of section 9.4.
    2. Among the surviving versions of the same ``(series_id, ts)``, keep the one with
       the latest ``ts_release``. A restatement supersedes the original *from its
       publication date onwards*, and every version is kept in the database precisely so
       that this choice can be made per date instead of once and forever (section 9.6).

    Args:
        df: Observations frame with at least ``series_id``, ``ts``, ``ts_release``.
        as_of: Simulation date (ISO string, date or datetime).

    Returns:
        A copy holding one row per ``(series_id, ts)``. Empty if nothing was known yet.
    """
    if df.empty:
        return df.copy()

    cutoff = _as_date(as_of).isoformat()
    known = df[df["ts_release"].fillna(TS_RELEASE_UNKNOWN).str[:10] <= cutoff]
    if known.empty:
        return known.copy()

    ordered = known.sort_values(["series_id", "ts", "ts_release"])
    return ordered.drop_duplicates(subset=["series_id", "ts"], keep="last").reset_index(drop=True)


def latest(df: pd.DataFrame, series_id: str, as_of: Any) -> Reading:
    """Most recent value of one series that was public on ``as_of``.

    Args:
        df: Observations frame (any number of series).
        series_id: Series to read.
        as_of: Simulation date.

    Returns:
        A :class:`Reading`. When the series is absent or nothing had been published yet,
        every field is ``None`` — a visible hole, not a zero (section 12).
    """
    known = point_in_time(df[df["series_id"] == series_id], as_of)
    known = known[known["value"].notna()]
    if known.empty:
        return Reading(series_id=series_id, value=None, ts=None, ts_release=None,
                       staleness_days=None)

    row = known.sort_values("ts").iloc[-1]
    ts = _as_date(row["ts"])
    return Reading(
        series_id=series_id,
        value=float(row["value"]),
        ts=ts.isoformat(),
        ts_release=str(row["ts_release"])[:10] or None,
        staleness_days=(_as_date(as_of) - ts).days,
    )


def change_over(df: pd.DataFrame, series_id: str, as_of: Any, lookback_days: int) -> float | None:
    """Change in level between now and roughly ``lookback_days`` ago.

    The comparison point is the last observation on or before ``as_of - lookback_days``,
    both sides taken point-in-time. Using the nearest *earlier* point rather than the
    nearest point in either direction keeps the window from silently reaching forward on
    a series with gaps.

    Args:
        df: Observations frame.
        series_id: Series to read.
        as_of: Simulation date.
        lookback_days: Calendar days back. Calendar, not trading, days — the series here
            have different frequencies (daily, weekly, monthly) and a trading-day count
            would mean a different span for each.

    Returns:
        ``current - past``, or ``None`` if either side is unavailable. A series whose
        history does not reach back that far returns ``None`` rather than comparing
        against its own first value, which would report a change that never happened.
    """
    current = latest(df, series_id, as_of)
    if current.value is None:
        return None

    past_date = _as_date(as_of) - dt.timedelta(days=lookback_days)
    past = latest(df, series_id, past_date)
    if past.value is None or past.ts == current.ts:
        return None
    return current.value - past.value


def snapshot(
    df: pd.DataFrame,
    series_ids: list[str],
    as_of: Any,
    lookback_days: int = 30,
) -> list[Reading]:
    """Level-1 table: every configured series as it stood on ``as_of``.

    Args:
        df: Observations frame covering the requested series.
        series_ids: Series to report, in display order.
        as_of: Simulation date.
        lookback_days: Window for the change column.

    Returns:
        One :class:`Reading` per requested series, in the order given. A series with no
        data still gets a row, so the panel shows the hole instead of hiding the series.
    """
    readings: list[Reading] = []
    for series_id in series_ids:
        reading = latest(df, series_id, as_of)
        readings.append(
            Reading(
                series_id=reading.series_id,
                value=reading.value,
                ts=reading.ts,
                ts_release=reading.ts_release,
                staleness_days=reading.staleness_days,
                change=change_over(df, series_id, as_of, lookback_days),
                lookback_days=lookback_days,
            )
        )
    return readings


def to_frame(readings: list[Reading]) -> pd.DataFrame:
    """Render readings as a display frame, keeping ``None`` as ``None``."""
    return pd.DataFrame([
        {
            "series_id": r.series_id,
            "value": r.value,
            "change": r.change,
            "ts": r.ts,
            "ts_release": r.ts_release,
            "staleness_days": r.staleness_days,
        }
        for r in readings
    ])
