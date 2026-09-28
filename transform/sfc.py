"""The Colombian supervisor's view of a lender (CLAUDE.md §15.5, point 11).

Built on ``ingest/sfc.py`` (the CUIF of the Superintendencia Financiera). Point-in-time as
``transform/bcb``: a month is visible from its ``ts_release``, and the latest version at the
date wins.

Conventions of the source handled here:

- **The result (``590000``) accumulates from January.** The annualized return on equity is
  ``result × 12 / month / equity`` — an approximation, labelled so on the page. Months are
  never summed.
- **Loan quality by risk category** (A normal, B acceptable, C appreciable, D significant, E
  uncollectible — the SFC's own scale). The share in C, D and E is the supervisor's
  *calidad por calificación*, not the 90-day delinquency a company reports.
- **Net loans** (``140000``) are after impairment; the categories add up the **gross** book.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from transform.bcb import series

DEPOSITS = ("deposits_current", "deposits_term", "deposits_savings")


@dataclass(frozen=True)
class Snapshot:
    """One entity at its latest month known at ``as_of``. ``None`` = unknown."""

    entity: str
    name: str
    month: str | None
    total_assets: float | None
    loans_net: float | None
    loans_growth: float | None
    deposits: float | None
    deposits_growth: float | None
    loans_to_deposits: float | None
    risk_share: float | None          # (C + D + E) / (A … E), gross book
    net_income_ytd: float | None
    roe_annualized: float | None
    equity: float | None


def _year_before(values: pd.Series, day: pd.Timestamp) -> float | None:
    earlier = (day - pd.DateOffset(years=1)) + pd.offsets.MonthEnd(0)
    return float(values[earlier]) if earlier in values.index else None


def _growth(values: pd.Series, day: pd.Timestamp) -> float | None:
    if day not in values.index:
        return None
    before = _year_before(values, day)
    return None if not before or before <= 0 else float(values[day]) / before - 1


def assess(observations: pd.DataFrame, entity: str, name: str, as_of: Any) -> Snapshot | None:
    """See :class:`Snapshot`. ``None`` when nothing had been published by ``as_of``."""
    def s(key: str) -> pd.Series:
        return series(observations, f"{entity}:{key}", as_of)

    assets = s("total_assets")
    if assets.empty:
        return None
    month = assets.index[-1]

    def at(values: pd.Series) -> float | None:
        return float(values[month]) if month in values.index else None

    parts = [s(k) for k in DEPOSITS]
    deposits_line = pd.concat(parts, axis=1).sum(axis=1, min_count=1) if any(
        not p.empty for p in parts) else pd.Series(dtype=float)
    loans = s("loans_net")
    categories = {c: s(f"loans_cat_{c}") for c in "abcde"}
    gross = [at(v) for v in categories.values()]
    risky = [at(categories[c]) for c in "cde"]
    gross_total = sum(v for v in gross if v is not None) if any(v is not None for v in gross) \
        else None
    risk_share = (None if not gross_total else
                  sum(v for v in risky if v is not None) / gross_total)
    income, equity = at(s("net_income_ytd")), at(s("equity"))
    deposits = at(deposits_line)
    loans_now = at(loans)
    return Snapshot(
        entity=entity, name=name, month=month.date().isoformat(),
        total_assets=at(assets), loans_net=loans_now, loans_growth=_growth(loans, month),
        deposits=deposits, deposits_growth=_growth(deposits_line, month),
        loans_to_deposits=None if not deposits or loans_now is None else loans_now / deposits,
        risk_share=risk_share, net_income_ytd=income,
        roe_annualized=(None if income is None or not equity or equity <= 0
                        else income * 12 / month.month / equity),
        equity=equity,
    )


def compare(observations: pd.DataFrame, entities: Sequence[Mapping[str, Any]],
            as_of: Any) -> list[Snapshot]:
    """:func:`assess` for each configured entity (``{tipo, codigo, name}``), in order; the
    ones with nothing published are left out."""
    out = []
    for e in entities:
        # Same key as ingest.sfc.entity_key: "{tipo}-{codigo}".
        snap = assess(observations, f"{int(e['tipo'])}-{int(e['codigo'])}",
                      str(e.get("name")), as_of)
        if snap is not None:
            out.append(snap)
    return out
