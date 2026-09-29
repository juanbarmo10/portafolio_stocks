"""Insider buying and selling of one company, by month (🏢 Empresa; user request).

Context, not a signal (RESEARCH.md §2.42, §2.62: purchases did not precede better returns).
Built on ``ingest/insider_activity`` and point-in-time: a transaction is visible from the day
its Form 4 was filed, not from the day it happened.

Reading it well needs one distinction the chart keeps: **plan** sales (Rule 10b5-1, set up
months before) say little — executives sell stock they receive as pay, on a schedule —
while **discretionary** sales and, above all, **purchases** are decisions taken now.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

KINDS = {"buy": "Compras", "sell:discretionary": "Ventas discrecionales",
         "sell:plan": "Ventas con plan 10b5-1", "sell:unknown": "Ventas (sin dato de plan)"}


@dataclass(frozen=True)
class Activity:
    """``monthly``: ``[month, kind, label, value]`` (sales negative); the last 12 months:
    ``buy_12m``, ``sell_12m``, ``discretionary_12m``, ``buyers_12m`` (insider-days buying)."""

    monthly: pd.DataFrame
    buy_12m: float
    sell_12m: float
    discretionary_12m: float
    buyers_12m: int
    last_filed: str | None


def assess(observations: pd.DataFrame, cik: str, as_of: Any, months: int = 36) -> Activity | None:
    """See :class:`Activity`. ``None`` when the company has no P or S transaction filed."""
    if observations is None or observations.empty:
        return None
    day = pd.Timestamp(as_of).date().isoformat()
    rows = observations[observations["series_id"].astype(str).str.startswith(f"{cik}:insider:")
                        & (observations["ts_release"].astype(str) <= day)]
    if rows.empty:
        return None
    parts = rows["series_id"].astype(str).str.split(":")
    rows = rows.assign(kind=parts.str[2] + parts.map(lambda p: f":{p[3]}" if len(p) > 3
                                                     and p[3] != "owners" else ""),
                       owners=parts.map(lambda p: p[-1] == "owners"),
                       when=pd.to_datetime(rows["ts"].astype(str).str[:10]))
    values, counts = rows[~rows["owners"]], rows[rows["owners"]]
    start = pd.Timestamp(day) - pd.DateOffset(months=months)
    window = values[values["when"] >= start]
    monthly = (window.assign(month=window["when"].dt.to_period("M").dt.to_timestamp())
               .groupby(["month", "kind"])["value"].sum().reset_index())
    monthly["value"] = monthly.apply(lambda r: -r["value"] if r["kind"].startswith("sell")
                                     else r["value"], axis=1)
    monthly["label"] = monthly["kind"].map(KINDS)
    year = values[values["when"] >= pd.Timestamp(day) - pd.DateOffset(months=12)]
    buyers = counts[(counts["when"] >= pd.Timestamp(day) - pd.DateOffset(months=12))
                    & (counts["kind"] == "buy")]
    return Activity(
        monthly=monthly[["month", "kind", "label", "value"]],
        buy_12m=float(year.loc[year["kind"] == "buy", "value"].sum()),
        sell_12m=float(year.loc[year["kind"].str.startswith("sell"), "value"].sum()),
        discretionary_12m=float(year.loc[year["kind"] == "sell:discretionary", "value"].sum()),
        buyers_12m=int(buyers["value"].sum()),
        last_filed=str(rows["ts_release"].max()),
    )
