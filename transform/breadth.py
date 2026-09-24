"""Market breadth — level 2, "is the market healthy or is it a narrow rally?" (section 2).

A cap-weighted index can rise while most of its members fall, if the giants carry it. A
rally carried by a handful is fragile. Breadth measures how many members take part:
here, the share of index members trading above their own 200-session average — roughly a
trading year, the classic long-term trend reference (THEORY.md §2.1).

**Every value travels with its coverage.** The membership on each date is known
(``ingest/universe.py``), but free price data does not include the companies that have
since left: measured on 2026-09-23, 63 % of the members at the 2009 low can be priced,
86 % from early 2020. The missing ones are acquired, bankrupt or renamed — not a random
sample — so a breadth reading from the priced subset overstates health, most of all in
crises. ``coverage`` says how much of the index each reading actually rests on, and a
reading below the configured threshold is flagged, never shown as if it were whole
(section 9.5: the limitation goes on screen, not in a code comment).

**Split adjustment, and why it is not look-ahead.** Breadth cannot run on raw closes: a
4:1 split cuts the raw price by 75 % and would put a healthy company "below its average"
for 200 sessions. Closes are therefore expressed in today's share units (every split
applied). That uses splits *after* a date to compute that date's reading, which looks like
the section 9.1 trap and is not: a later split multiplies every bar in the window by the
same constant, and ``close / average`` is invariant to a constant. Splits *inside* the
window — the ones that would distort a raw comparison — are exactly the ones this corrects.
``test_breadth.py`` pins that invariance as a test.

Pure functions over stored frames: no network, no database handle (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from transform.adjustments import adjustment_factor, as_date, as_frame

DEFAULT_WINDOW = 200
DEFAULT_THRESHOLD = 0.85

# A session whose priced count collapses below this share of the surrounding median is a
# hole in the source, not survivorship (see source_holes).
HOLE_FLOOR = 0.5
HOLE_WINDOW = 21


@dataclass(frozen=True)
class BreadthReading:
    """One date's breadth, with everything needed to judge whether to trust it.

    Attributes:
        date: The session.
        members: Index members that day.
        priced: Members with a close that day — the survivorship-relevant count.
        computable: Priced members with a full averaging window behind them. A recent IPO
            is priced but not yet computable; that is not survivorship, just youth.
        above: Computable members closing above their average.
        breadth: ``above / computable``, or ``None`` with nothing computable.
        coverage: ``priced / members``. How much of the index the reading rests on.
        meets_threshold: Whether ``coverage`` reaches the configured minimum. Below it the
            reading exists but must not be presented as a statement about the index.
        source_hole: The source returned this session nearly empty while the sessions
            around it were whole. Not survivorship — a defect in the data for one day —
            so ``breadth`` is ``None`` rather than a figure over a random subset.
    """

    date: str
    members: int
    priced: int
    computable: int
    above: int
    breadth: float | None
    coverage: float | None
    meets_threshold: bool
    source_hole: bool = False


def membership_mask(
    intervals: Sequence[Mapping[str, Any]] | pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: Sequence[str],
) -> pd.DataFrame:
    """Boolean frame ``dates × tickers``: was the ticker a member on that date?

    Intervals are ``[start_date, end_date)``, end exclusive and ``None`` while current.
    """
    frame = as_frame(intervals)
    if frame.empty or len(dates) == 0:
        return pd.DataFrame(False, index=dates, columns=list(tickers))

    # Only intervals that touch the date range matter; a stay that ended in 2005 adds an
    # all-False column and nothing else.
    first, last = dates.min(), dates.max()
    ends = pd.to_datetime(frame["end_date"].map(lambda v: as_date(v)), errors="coerce")
    starts = pd.to_datetime(frame["start_date"].map(lambda v: as_date(v)))
    frame = frame[(starts <= last) & (ends.isna() | (ends > first))]

    # Every column created at once: members without prices still count (they are the
    # gap), and adding them one by one fragments the frame into hundreds of blocks.
    columns = list(dict.fromkeys([*tickers, *frame["ticker"]]))
    mask = pd.DataFrame(False, index=dates, columns=columns)
    for row in frame.to_dict("records"):
        ticker = row["ticker"]
        start = pd.Timestamp(as_date(row["start_date"]))
        end = row.get("end_date")
        window = dates >= start
        if end is not None and not pd.isna(end):
            window &= dates < pd.Timestamp(as_date(end))
        mask.loc[window, ticker] = True
    return mask


def adjusted_closes(
    closes: pd.DataFrame, splits: Mapping[str, Mapping[Any, float]]
) -> pd.DataFrame:
    """Raw closes (``dates × tickers``) expressed in today's share units.

    See the module docstring for why every split, including later ones, is applied: the
    breadth ratio is invariant to the constant a later split introduces, and the splits
    inside the window are the ones that have to be corrected.
    """
    out = closes.copy()
    for ticker in out.columns:
        ticker_splits = splits.get(ticker) or {}
        if not ticker_splits:
            continue
        factors = pd.Series(
            [adjustment_factor(day, ticker_splits) for day in out.index],
            index=out.index,
        )
        out[ticker] = out[ticker] / factors
    return out


def source_holes(priced: pd.Series, *, window: int = HOLE_WINDOW,
                 floor: float = HOLE_FLOOR) -> pd.Series:
    """Sessions where the priced count collapses against the sessions around it.

    Two different things lower coverage, and the panel has to tell them apart:

    - **Survivorship**: members that have since left, missing from free price data.
      Structural and slow — it changes by a few members a year.
    - **A hole in the source**: the provider returns one session nearly empty. Measured on
      2026-09-23 — Yahoo served 2026-09-22 for JPM, AAPL and MSFT but not for XOM or KO,
      and still did hours later, so it is not always transient.

    Treated as survivorship, one bad day reads as "12 % coverage" and splits the usable
    series in two. The same arithmetic as ``ingest.universe.incomplete_sessions``, applied
    to the stored data rather than to a download — kept as two small functions so that
    ingest does not import from transform.
    """
    typical = priced.rolling(window, center=True, min_periods=5).median()
    return priced < floor * typical


def breadth_series(
    intervals: Sequence[Mapping[str, Any]] | pd.DataFrame,
    closes: pd.DataFrame,
    splits: Mapping[str, Mapping[Any, float]] | None = None,
    *,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    start: Any = None,
) -> list[BreadthReading]:
    """Share of members above their ``window``-session average, date by date.

    Args:
        intervals: Membership intervals, as stored in ``universe_membership``.
        closes: Raw closes, ``dates × tickers`` (a wide frame; see :func:`wide_closes`).
        splits: ``{ticker: {ex_date: ratio}}`` for the adjustment.
        window: Averaging window in sessions.
        threshold: Minimum coverage for a reading to count as describing the index.
        start: First date to report. Earlier dates are still used to warm up averages.

    Returns:
        One :class:`BreadthReading` per session on or after ``start``.
    """
    if closes is None or closes.empty:
        return []
    closes = closes.sort_index()
    adjusted = adjusted_closes(closes, splits or {})
    average = adjusted.rolling(window, min_periods=window).mean()

    mask = membership_mask(intervals, closes.index, list(closes.columns))
    # Members with no price column at all still count as members: they are the gap.
    adjusted = adjusted.reindex(columns=mask.columns)
    average = average.reindex(columns=mask.columns)

    priced = mask & adjusted.notna()
    computable = priced & average.notna()
    above = computable & (adjusted > average)

    counts = pd.DataFrame({
        "members": mask.sum(axis=1),
        "priced": priced.sum(axis=1),
        "computable": computable.sum(axis=1),
        "above": above.sum(axis=1),
    }).astype(int)
    counts["hole"] = source_holes(counts["priced"])
    if start is not None:
        counts = counts[counts.index >= pd.Timestamp(as_date(start))]

    readings: list[BreadthReading] = []
    for day, row in counts.iterrows():
        # Plain Python types, as the dataclass declares: numpy scalars leak otherwise
        # (np.False_ fails an ``is False`` check and does not serialize to JSON).
        members, priced_n = int(row.members), int(row.priced)
        computable_n, above_n = int(row.computable), int(row.above)
        coverage = priced_n / members if members else None
        hole = bool(row.hole)
        readings.append(BreadthReading(
            date=day.date().isoformat(),
            members=members,
            priced=priced_n,
            computable=computable_n,
            above=above_n,
            # Over a hole the priced subset is whatever the source happened to return;
            # a percentage of that is a number about nothing (section 12).
            breadth=None if hole else (above_n / computable_n if computable_n else None),
            coverage=coverage,
            meets_threshold=bool(not hole and coverage is not None and coverage >= threshold),
            source_hole=hole,
        ))
    return readings


def wide_closes(observations: pd.DataFrame, suffix: str = "close_raw") -> pd.DataFrame:
    """Long observations -> ``dates × tickers`` raw closes, one row per session."""
    if observations is None or observations.empty:
        return pd.DataFrame()
    rows = observations[observations["series_id"].str.endswith(f":{suffix}")]
    if rows.empty:
        return pd.DataFrame()
    frame = rows.assign(
        ticker=rows["series_id"].str.rsplit(":", n=1).str[0],
        day=pd.to_datetime(rows["ts"].str[:10]),
    )
    return frame.pivot_table(index="day", columns="ticker", values="value", aggfunc="last")


def splits_by_ticker(
    actions: Sequence[Mapping[str, Any]] | pd.DataFrame, source: str = "yfinance"
) -> dict[str, dict[Any, float]]:
    """``corporate_actions`` rows -> ``{ticker: {ex_date: ratio}}`` for one provider."""
    frame = as_frame(actions)
    out: dict[str, dict[Any, float]] = {}
    if frame.empty:
        return out
    rows = frame[(frame["kind"] == "split") & (frame["source"] == source)]
    for row in rows.to_dict("records"):
        if row.get("ratio") is None or pd.isna(row["ratio"]):
            continue
        out.setdefault(row["ticker"], {})[as_date(row["ex_date"])] = float(row["ratio"])
    return out


def usable_from(readings: Sequence[BreadthReading]) -> str | None:
    """First date from which every later reading meets the coverage threshold.

    The honest start of the series: not the first date that happens to clear the bar, but
    the date after which it never drops below again. Holes in the source are skipped — they
    neither start nor break the run, because they are not about coverage.
    """
    start = None
    for reading in readings:
        if reading.source_hole:
            continue            # a bad day in the source says nothing about coverage
        if reading.meets_threshold:
            start = start or reading.date
        else:
            start = None
    return start
