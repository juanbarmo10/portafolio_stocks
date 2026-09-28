"""Fundamental inflection: growth that accelerates or slows, a gross margin that turns
(🏢 Empresa, Telegram; CLAUDE.md §15.5, point 5).

A growth stock moves on the **change** of its growth, not on its level: the market already
prices the current rate. This module reads the latest quarter, as knowable at a date (the
quarterly table of ``transform.quarterly``, point-in-time), and names two events:

- **Acceleration / deceleration**: this quarter's year-on-year revenue growth minus the
  previous quarter's, beyond ``acceleration`` (a fraction: 0.10 = 10 points).
- **Gross margin turn**: this quarter's gross margin against the same quarter a year earlier
  (seasonality cancels), beyond ``gross_margin`` (0.03 = 3 points). A growing business whose
  gross margin falls is usually buying the growth with price.

Thresholds are written in config before looking at any data (§9.7) and are the user's to
change. The event fires on the **filing**, never on the price (§12): the alert keys it by
quarter, and :func:`is_new` says whether that quarter became visible recently.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from transform import quarterly as qt


@dataclass(frozen=True)
class Inflection:
    """The latest quarter's changes, and the events they cross. ``None`` = unknown."""

    quarter: str
    growth: float | None
    previous_growth: float | None
    acceleration: float | None
    gross_margin: float | None
    gross_margin_year_ago: float | None
    gross_margin_change: float | None
    signals: list[tuple[str, str]] = field(default_factory=list)   # (kind, Spanish text)


def _pp(value: float) -> str:
    return f"{value * 100:+.1f}".replace(".", ",") + " pp"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}".replace(".", ",") + " %"


def assess(observations: pd.DataFrame, cik: str, as_of: Any, *,
           acceleration: float = 0.10, gross_margin: float = 0.03) -> Inflection | None:
    """See :class:`Inflection`. ``None`` when the company has no quarterly revenue."""
    table = qt.quarter_table(observations, cik, as_of, quarters=8)
    if table.empty:
        return None
    last = table.iloc[-1]
    prev = table.iloc[-2] if len(table) > 1 else None
    growth = last["revenue_growth"]
    growth = None if growth is None or pd.isna(growth) else float(growth)
    previous = None if prev is None or pd.isna(prev["revenue_growth"]) \
        else float(prev["revenue_growth"])
    accel = None if growth is None or previous is None else growth - previous

    end = pd.Timestamp(last["quarter_end"])
    year_ago = table[[abs((end - pd.Timestamp(d)).days - 365) <= 15
                      for d in table["quarter_end"]]]
    gm = last["gross_margin"]
    gm = None if gm is None or pd.isna(gm) else float(gm)
    gm_ago = None if year_ago.empty or pd.isna(year_ago.iloc[-1]["gross_margin"]) \
        else float(year_ago.iloc[-1]["gross_margin"])
    gm_change = None if gm is None or gm_ago is None else gm - gm_ago

    signals = []
    if accel is not None and abs(accel) >= acceleration:
        kind = "accelerates" if accel > 0 else "slows"
        verb = "ACELERA" if accel > 0 else "FRENA"
        signals.append((kind, f"El crecimiento {verb}: {_pct(growth)} interanual frente a "
                              f"{_pct(previous)} el trimestre anterior ({_pp(accel)})."))
    if gm_change is not None and abs(gm_change) >= gross_margin:
        kind = "margin_up" if gm_change > 0 else "margin_down"
        verb = "sube" if gm_change > 0 else "baja"
        signals.append((kind, f"El margen bruto {verb}: {_pct(gm)} frente a {_pct(gm_ago)} un "
                              f"año antes ({_pp(gm_change)})."))
    return Inflection(quarter=str(last["quarter_end"]), growth=growth,
                      previous_growth=previous, acceleration=accel, gross_margin=gm,
                      gross_margin_year_ago=gm_ago, gross_margin_change=gm_change,
                      signals=signals)


def is_new(observations: pd.DataFrame, cik: str, as_of: Any, lookback_days: int) -> bool:
    """Whether the latest quarter at ``as_of`` was not yet visible ``lookback_days`` earlier:
    news, as opposed to a quarter the user has had months to read."""
    now = qt.quarter_table(observations, cik, as_of, quarters=1)
    if now.empty:
        return False
    earlier = (pd.Timestamp(as_of) - pd.Timedelta(days=lookback_days)).date().isoformat()
    before = qt.quarter_table(observations, cik, earlier, quarters=1)
    return before.empty or str(before.iloc[-1]["quarter_end"]) < str(now.iloc[-1]["quarter_end"])
