"""The Brazilian supervisor's view of a bank the SEC cannot read (RESEARCH.md §2.43).

Built on ``ingest/bcb.py`` (IF.data, SGS) and FRED's BRL/USD. Everything is point-in-time:
a quarter is visible from its ``ts_release`` — observed (first seen) or derived (late on
purpose) — and the latest version at the date wins (section 9.6).

Two conventions of the source handled here, where derivations belong:

- **Profit accumulates over the semester** (measured on Nu: 2025 Q1 2,87 → S1 5,97 billion
  reais). The quarter is ``Q1 = S(Mar)``, ``Q2 = S(Jun) − S(Mar)``, ``Q3 = S(Sep)``,
  ``Q4 = S(Dec) − S(Sep)``, dated by the later of the two releases — the same rule as the
  derived quarters of §9.12 and §9.14.
- **The credit portfolio changed basis in 2025** (Res. 4.966, IFRS 9-like): "Carteira de
  Crédito Classificada" until 2024, "Carteira de Crédito" after. They are two series; growth
  is only computed within one basis, never across the break (§9.8).

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from transform.macro import point_in_time


def series(observations: pd.DataFrame, series_id: str, as_of: Any) -> pd.Series:
    """One series as knowable on ``as_of``, indexed by reference date."""
    rows = observations[observations["series_id"] == series_id] if not observations.empty \
        else observations
    if rows.empty:
        return pd.Series(dtype=float)
    known = point_in_time(rows, as_of)
    if known.empty:
        return pd.Series(dtype=float)
    out = pd.Series(known["value"].astype(float).to_numpy(),
                    index=pd.to_datetime(known["ts"].astype(str).str[:10]))
    return out[~out.index.duplicated(keep="last")].sort_index()


def quarterly_from_semester(semester: pd.Series) -> pd.Series:
    """Quarterly values from semester-to-date ones (module docstring). A Q2 or Q4 whose Q1 or
    Q3 is missing is left out, never taken as the whole semester."""
    out = {}
    for day, value in semester.items():
        if day.month in (3, 9):
            out[day] = value
        elif day.month in (6, 12):
            first = pd.Timestamp(year=day.year, month=day.month - 3, day=1) + pd.offsets.MonthEnd(0)
            if first in semester.index:
                out[day] = value - semester[first]
    return pd.Series(out, dtype=float).sort_index()


def _year_before(values: pd.Series, day: pd.Timestamp) -> float | None:
    earlier = day - pd.DateOffset(years=1)
    earlier = pd.Timestamp(earlier) + pd.offsets.MonthEnd(0)
    return float(values[earlier]) if earlier in values.index else None


@dataclass(frozen=True)
class Supervisory:
    """The latest quarter known at ``as_of``, and its history.

    Attributes:
        quarter: Reference date of the latest quarter.
        observed: Whether its publication date was observed (first seen) or derived.
        credit_portfolio / credit_basis: The latest portfolio and which basis it is on.
        credit_growth: Year over year, within the same basis only.
        problem_share / delinquent_share: ``problem_assets`` and ``delinquent`` over the
            total exposure (Res. 4.966 report, from 2025).
        net_income_quarter: Derived from the semester accumulation.
        roe_annualized: ``4 × quarterly profit / equity`` — approximate, labelled so.
        history: ``[date, credit, basis, problem_share, net_income_q]`` for the chart.
    """

    quarter: str | None
    observed: bool | None
    credit_portfolio: float | None
    credit_basis: str | None
    credit_growth: float | None
    credit_clients: float | None
    basel: float | None
    cet1: float | None
    problem_share: float | None
    delinquent_share: float | None
    net_income_quarter: float | None
    roe_annualized: float | None
    equity: float | None
    history: pd.DataFrame


def assess(observations: pd.DataFrame, code: str, as_of: Any) -> Supervisory | None:
    """See :class:`Supervisory`. ``None`` when nothing had been published by ``as_of``."""
    def s(key: str) -> pd.Series:
        return series(observations, f"{code}:{key}", as_of)

    total = s("total_assets")
    if total.empty:
        return None
    quarter = total.index[-1]
    new_basis, old_basis = s("credit_portfolio"), s("credit_portfolio_classified")
    if quarter in new_basis.index:
        credit, basis = float(new_basis[quarter]), "Res. 4.966 (desde 2025)"
        before = _year_before(new_basis, quarter)
    elif quarter in old_basis.index:
        credit, basis = float(old_basis[quarter]), "clasificada (hasta 2024)"
        before = _year_before(old_basis, quarter)
    else:
        credit, basis, before = None, None, None
    growth = None if credit is None or not before or before <= 0 else credit / before - 1

    def at(values: pd.Series) -> float | None:
        return float(values[quarter]) if quarter in values.index else None

    exposure, problem, delinquent = s("exposure_total"), s("problem_assets"), s("delinquent")
    def share(part: pd.Series) -> float | None:
        return None if at(part) is None or not at(exposure) else at(part) / at(exposure)

    income_q = quarterly_from_semester(s("net_income_semester"))
    equity = at(s("equity"))
    net_q = float(income_q[quarter]) if quarter in income_q.index else None
    observed = s("release_observed")

    credit_line = pd.concat([old_basis.rename("credit").to_frame().assign(basis="clasificada"),
                             new_basis.rename("credit").to_frame().assign(basis="Res. 4.966")])
    history = credit_line.join((problem / exposure).rename("problem_share"), how="outer") \
        .join(income_q.rename("net_income_q"), how="outer").sort_index() \
        .rename_axis("date").reset_index()
    return Supervisory(
        quarter=quarter.date().isoformat(),
        observed=None if quarter not in observed.index else bool(observed[quarter]),
        credit_portfolio=credit, credit_basis=basis, credit_growth=growth,
        credit_clients=at(s("credit_clients")), basel=at(s("basel_ratio")),
        cet1=at(s("cet1_ratio")), problem_share=share(problem),
        delinquent_share=share(delinquent), net_income_quarter=net_q,
        roe_annualized=None if net_q is None or not equity or equity <= 0
        else 4 * net_q / equity,
        equity=equity, history=history,
    )
