"""Where the price stands before buying: trend, recent moves, and the two study conditions
(🏢 Empresa, 📋 Desplegar; CLAUDE.md §15.5, points 8 and 12).

Description first, gate only if earned: the conditions are the ones of ``knife_study``
(``settings.yaml``), evaluated on the split-adjusted closes known at the date. Whether either
becomes a **gate** ("do not buy unless a written decision says so") depends on the study's
verdict, recorded in ``panel.entry_gate`` after its single run — never assumed.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class EntryContext:
    """The price's position at ``day``. ``None`` = not enough history."""

    day: str
    price: float | None
    sma: float | None
    distance_to_sma: float | None      # price / average − 1
    return_1m: float | None
    return_3m: float | None
    return_12m: float | None
    drawdown_52w: float | None         # from the highest close of the last 365 days
    knife: bool | None
    overextended: bool | None


def _return(series: pd.Series, day: pd.Timestamp, days: int) -> float | None:
    before = series[series.index <= day - pd.Timedelta(days=days)]
    if before.empty or before.iloc[-1] <= 0:
        return None
    return float(series.iloc[-1] / before.iloc[-1] - 1)


def assess(closes: pd.Series, as_of: Any, conditions: Mapping[str, Any]) -> EntryContext | None:
    """See :class:`EntryContext`. ``closes``: split-adjusted closes by date."""
    day = pd.Timestamp(as_of)
    series = closes[closes.index <= day].dropna()
    if series.empty:
        return None
    knife_cfg, over_cfg = conditions["knife"], conditions["overextended"]
    window = int(knife_cfg["sma_sessions"])
    sma = float(series.tail(window).mean()) if len(series) >= window else None
    price = float(series.iloc[-1])
    last_year = series[series.index > day - pd.Timedelta(days=365)]
    r3 = _return(series, day, int(knife_cfg["return_days"]))
    return EntryContext(
        day=series.index[-1].date().isoformat(), price=price, sma=sma,
        distance_to_sma=None if sma is None else price / sma - 1,
        return_1m=_return(series, day, 30), return_3m=r3,
        return_12m=_return(series, day, 365),
        drawdown_52w=float(price / last_year.max() - 1) if len(last_year) else None,
        knife=(None if sma is None or r3 is None
               else bool(price < sma and r3 <= float(knife_cfg["max_return"]))),
        overextended=(None if sma is None
                      else bool(price >= float(over_cfg["min_ratio_to_sma"]) * sma)),
    )
