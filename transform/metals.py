"""Gold, silver and their miners, as context (CLAUDE.md §15.5, point 14).

A metal pays no coupon and has no cash flow, so it has no valuation in the sense of question
3; what the panel can say about it is **context** (question 1: real rates, the dollar) and
**behaviour** (question 4: did it cushion the portfolio when the market fell?). Whether to
hold any is the user's written policy; nothing here recommends a weight.

- **Gold/silver ratio**: ounces of silver one ounce of gold buys, from the nearest COMEX
  futures (``GC=F``, ``SI=F``). The ETFs would give the same shape in another scale (each
  share holds a fraction of an ounce that shrinks with fees). A high ratio means silver is
  cheap *relative to gold*, which says nothing about whether either is cheap.
- **Behaviour in the market's falls**: every decline of SPY (total return) of at least
  ``DRAWDOWN_THRESHOLD`` from a peak, and what each asset did from that peak to the trough.
  10 % is the convention for a "correction"; it is a definition, not a tuned parameter.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

DRAWDOWN_THRESHOLD = 0.10


def ratio(gold: pd.Series, silver: pd.Series) -> pd.Series:
    """Gold price over silver price on the days both have a positive close."""
    both = pd.concat({"g": gold, "s": silver}, axis=1).dropna()
    both = both[(both["g"] > 0) & (both["s"] > 0)]
    return (both["g"] / both["s"]).rename("ratio")


def percentile_of_last(series: pd.Series) -> float | None:
    """Share of the series' history at or below its last value (0 … 1)."""
    clean = series.dropna()
    if len(clean) < 2:
        return None
    return float((clean <= clean.iloc[-1]).mean())


@dataclass(frozen=True)
class Episode:
    peak: pd.Timestamp
    trough: pd.Timestamp
    depth: float          # negative: the fall from peak to trough
    recovered: bool       # the index made a new high afterwards


def drawdown_episodes(index: pd.Series, threshold: float = DRAWDOWN_THRESHOLD) -> list[Episode]:
    """Non-overlapping falls of at least ``threshold`` from a running peak.

    An episode runs from a peak to the lowest point before the index regains that peak; the
    last one may still be open (``recovered=False``).
    """
    clean = index.dropna()
    out: list[Episode] = []
    if clean.empty:
        return out
    peak_day, peak = clean.index[0], float(clean.iloc[0])
    trough_day, trough = peak_day, peak
    for day, value in clean.items():
        value = float(value)
        if value >= peak:
            if trough / peak - 1 <= -threshold:
                out.append(Episode(peak_day, trough_day, trough / peak - 1, True))
            peak_day, peak, trough_day, trough = day, value, day, value
        elif value < trough:
            trough_day, trough = day, value
    if trough / peak - 1 <= -threshold:
        out.append(Episode(peak_day, trough_day, trough / peak - 1, False))
    return out


def _at(series: pd.Series, day: pd.Timestamp) -> float | None:
    known = series[series.index <= day].dropna()
    if known.empty or (day - known.index[-1]).days > 7:
        return None
    return float(known.iloc[-1])


def cushion(episodes: list[Episode], assets: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Each episode with every asset's return from its peak to its trough. An asset without
    a close within 7 days of either date is ``None`` for that episode (it did not exist, or
    the source has a hole), never measured over another span."""
    rows = []
    for e in episodes:
        row = {"peak": e.peak.date().isoformat(), "trough": e.trough.date().isoformat(),
               "depth": e.depth, "recovered": e.recovered}
        for name, series in assets.items():
            start, end = _at(series, e.peak), _at(series, e.trough)
            row[name] = None if start is None or end is None or start <= 0 else end / start - 1
        rows.append(row)
    return pd.DataFrame(rows)
