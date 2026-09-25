"""What it costs to get a contribution from pesos into a position (CLAUDE.md §2, level 4).

The route is COP → dollars bought at an intermediary → transfer → IBKR → one or more orders.
Two kinds of cost ride on it, and they call for opposite habits:

- **Proportional** (the exchange spread against the TRM): the same share of every peso,
  whatever the size or the frequency. Batching does nothing for it.
- **Fixed** (a fee per transfer, a minimum commission per order): a larger share of a small
  contribution. Batching — fewer, larger transfers and orders — is the only lever.

Nothing here assumes a fee. The end-to-end cost of each deposit is **measured**: the pesos
the user paid (entered in ``settings.local.yaml``, the only figure the statement does not
have) against the dollars IBKR credited times the TRM of that day. With deposits of
different sizes, a fixed component shows up as a larger percentage on the smaller ones, and
a straight line through the points separates the two — labelled as a fit over N deposits,
never passed off as the intermediary's price list.

The commission per order is the median of the account's own executions, not the published
schedule.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def trm_on(trm: pd.DataFrame, day: Any) -> tuple[float | None, str | None]:
    """The TRM in force on ``day``: the last one on or before it. ``(value, its date)``."""
    if trm is None or trm.empty:
        return None, None
    days = trm["ts"].astype(str).str[:10]
    rows = trm[days <= str(day)[:10]]
    if rows.empty:
        return None, None
    last = rows.assign(day=days).sort_values("day").iloc[-1]
    return float(last["value"]), str(last["day"])


def deposit_costs(deposits: pd.DataFrame, paid: Sequence[Mapping[str, Any]],
                  trm: pd.DataFrame) -> pd.DataFrame:
    """Each IBKR deposit with its end-to-end cost when the pesos paid are known.

    Args:
        deposits: ``cash_transactions`` rows of kind ``deposit`` (``ts``, ``amount`` USD).
        paid: From config, ``[{date, cop_paid, bought_on?}]`` — ``date`` is the IBKR credit
            date; ``bought_on`` the day the dollars were bought, whose TRM is the fair one
            (defaults to the credit date).
        trm: ``TRM:COP_USD`` observations.

    Returns:
        ``[date, usd_received, cop_paid, trm, trm_date, cost_cop, cost_pct]``; the cost
        columns are ``None`` for a deposit whose pesos were not entered.
    """
    columns = ["date", "usd_received", "cop_paid", "trm", "trm_date", "cost_cop", "cost_pct"]
    if deposits is None or deposits.empty:
        return pd.DataFrame(columns=columns)
    by_date = {str(p["date"])[:10]: p for p in paid or [] if p.get("date")}
    rows = []
    for row in deposits.sort_values("ts").to_dict("records"):
        day = str(row["ts"])[:10]
        usd = float(row["amount"])
        entry = by_date.get(day, {})
        cop = entry.get("cop_paid")
        rate, rate_day = trm_on(trm, entry.get("bought_on") or day)
        cost_cop = cost_pct = None
        if cop and rate:
            cost_cop = float(cop) - usd * rate
            cost_pct = cost_cop / float(cop)
        rows.append({"date": day, "usd_received": usd,
                     "cop_paid": float(cop) if cop else None, "trm": rate,
                     "trm_date": rate_day, "cost_cop": cost_cop, "cost_pct": cost_pct})
    return pd.DataFrame(rows, columns=columns)


@dataclass(frozen=True)
class CostSplit:
    """A straight line through the measured deposits: ``cost = fixed + rate × amount``.

    Attributes:
        fixed_cop: Cost per transfer that does not depend on the amount.
        proportional: Share of every peso (spread plus any percentage fee).
        deposits: How many deposits the line is fitted on — the reader's measure of trust.
    """

    fixed_cop: float
    proportional: float
    deposits: int


def split_costs(costs: pd.DataFrame, *, min_deposits: int = 3) -> CostSplit | None:
    """Least squares over the deposits with a measured cost; ``None`` below ``min_deposits``
    or when every deposit has the same size (then the two parts cannot be told apart)."""
    known = costs.dropna(subset=["cost_cop", "cop_paid"])
    if len(known) < min_deposits or known["cop_paid"].nunique() < 2:
        return None
    slope, intercept = np.polyfit(known["cop_paid"].astype(float),
                                  known["cost_cop"].astype(float), 1)
    return CostSplit(fixed_cop=float(intercept), proportional=float(slope),
                     deposits=len(known))


def order_fee(trades: pd.DataFrame) -> float | None:
    """Median commission per execution in the account (positive USD)."""
    if trades is None or trades.empty or "commission" not in trades:
        return None
    fees = pd.to_numeric(trades["commission"], errors="coerce").abs()
    fees = fees[fees > 0]
    return float(fees.median()) if len(fees) else None


def batching(monthly_usd: float, *, every_months: Sequence[int], fixed_usd: float | None,
             proportional: float | None, fee_per_order: float | None,
             orders: int = 1) -> pd.DataFrame:
    """Cost of moving ``monthly_usd`` a month, transferring every k months.

    Per transfer: ``fixed + proportional × amount + orders × fee``, as a share of the
    amount. Each unknown input leaves the columns it feeds ``None`` (section 12).

    What batching costs is not here: money waiting in pesos is money not yet invested.
    The page states that trade-off in words; putting a number on it needs an expected
    return, which is the user's assumption, not a measurement.
    """
    rows = []
    for k in every_months:
        amount = monthly_usd * k
        fixed = None if fixed_usd is None else fixed_usd / amount
        spread = proportional
        commission = None if fee_per_order is None else orders * fee_per_order / amount
        parts = [fixed, spread, commission]
        rows.append({"every_months": k, "amount_usd": amount, "fixed": fixed,
                     "proportional": spread, "commission": commission,
                     "total": None if any(p is None for p in parts) else sum(parts),
                     "fixed_and_commission_usd_year": (
                         None if fixed_usd is None or fee_per_order is None
                         else (fixed_usd + orders * fee_per_order) * 12 / k)})
    return pd.DataFrame(rows)
