"""Summaries for the tax conversation — ESTIMADO, never a settled tax (CLAUDE.md section 11).

**What this module does not do, on purpose.** It computes no tax. It asserts no rate, no
threshold and no treatment. A holding period that changes how a gain is treated exists only
if the owner writes it in ``settings.local.yaml`` after asking an accountant; until then the
answer is ``None`` and the page says "no calculable". The same with every other parameter.

What it does compute is mechanical and true whatever the tax rules turn out to be: the local-
currency value of what is held, how much of the result came from the currency, the dividends
and the withholding actually charged (observed, section 11), and how much sits in assets
with US *situs* — shown as a question for an adviser, not as an exposure to any tax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from fiscal.lots import fx_asof


@dataclass(frozen=True)
class Attribution:
    """Unrealized result in local currency, split into what the assets and the currency did."""

    cost_local: float
    value_local: float
    result_local: float
    asset_part_local: float
    fx_part_local: float
    result_usd: float
    fx_now: float
    fx_cost_average: float        # implied: frozen local cost / USD cost


def attribution(valued: pd.DataFrame) -> Attribution | None:
    """The portfolio-level split. ``None`` if any lot lacks a rate or a price: a partial sum
    presented as the total would be a wrong number that looks right."""
    if valued is None or valued.empty:
        return None
    needed = ["cost_local", "value_local", "asset_part_local", "fx_part_local", "value_usd"]
    if valued[needed].isna().any().any():
        return None
    cost_usd = float(valued["cost_usd"].sum())
    cost_local = float(pd.to_numeric(valued["cost_local"]).sum())
    value_usd = float(valued["value_usd"].sum())
    value_local = float(valued["value_local"].sum())
    return Attribution(
        cost_local=cost_local, value_local=value_local,
        result_local=value_local - cost_local,
        asset_part_local=float(valued["asset_part_local"].sum()),
        fx_part_local=float(valued["fx_part_local"].sum()),
        result_usd=value_usd - cost_usd,
        fx_now=value_local / value_usd if value_usd else float("nan"),
        fx_cost_average=cost_local / cost_usd if cost_usd else float("nan"),
    )


def holding_threshold(lots: pd.DataFrame, today: Any, months: int | None) -> pd.DataFrame:
    """For each lot, the date it completes ``months`` held and the days left.

    ``months`` comes from the owner's config, never from here. ``None`` → both columns
    ``None``: the threshold is unknown, and an unknown threshold is not "already met".
    """
    out = lots.copy()
    if months is None or out.empty:
        out["threshold_date"] = None
        out["days_to_threshold"] = None
        return out
    today = pd.Timestamp(str(today)[:10])
    dates = [pd.Timestamp(str(ts)[:10]) + pd.DateOffset(months=int(months)) for ts in out["ts"]]
    out["threshold_date"] = [d.date().isoformat() for d in dates]
    out["days_to_threshold"] = [max(0, (d - today).days) for d in dates]
    return out


def classify_disposals(disposals: pd.DataFrame, months: int | None) -> pd.DataFrame:
    """Mark each disposal as held at least ``months`` or not; ``None`` without the parameter."""
    out = disposals.copy()
    if months is None or out.empty:
        out["held_past_threshold"] = None
        return out
    out["held_past_threshold"] = [
        pd.Timestamp(str(sold)[:10])
        >= pd.Timestamp(str(bought)[:10]) + pd.DateOffset(months=int(months))
        for bought, sold in zip(out["acquired_ts"], out["ts"])
    ]
    return out


def dividends_local(cash: pd.DataFrame, fx: pd.Series) -> pd.DataFrame:
    """Each payment date: gross dividend, withholding actually charged, and net, in USD and in
    local currency at that date's rate. Withholding is the observed figure (section 11)."""
    columns = ["date", "ticker", "gross_usd", "withholding_usd", "net_usd", "fx",
               "gross_local", "withholding_local", "net_local"]
    if cash is None or cash.empty:
        return pd.DataFrame(columns=columns)
    rows = cash[cash["kind"].isin(["dividend", "withholding_tax"])].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)
    rows["date"] = rows["ts"].str[:10]
    rows["ticker"] = rows["ticker"].fillna("—")
    table = rows.pivot_table(index=["date", "ticker"], columns="kind", values="amount",
                             aggfunc="sum", fill_value=0.0).reset_index()
    gross = table["dividend"] if "dividend" in table else 0.0
    withholding = table["withholding_tax"] if "withholding_tax" in table else 0.0
    out = pd.DataFrame({"date": table["date"], "ticker": table["ticker"],
                        "gross_usd": gross, "withholding_usd": withholding})
    out["net_usd"] = out["gross_usd"] + out["withholding_usd"]
    out["fx"] = [fx_asof(fx, d) for d in out["date"]]
    rate = pd.to_numeric(out["fx"], errors="coerce")
    for part in ("gross", "withholding", "net"):
        out[f"{part}_local"] = out[f"{part}_usd"] * rate
    return out[columns].sort_values("date").reset_index(drop=True)


def us_situs_value(valued_positions: pd.DataFrame, securities: pd.DataFrame) -> dict[str, Any]:
    """Market value of the positions whose issuer is in the United States.

    Only the figure, and only as a **question for an adviser** (RESEARCH.md section 3,
    question 2). Whether any estate tax applies, from what amount and to whom, is not
    answered here. A position with no issuer country is counted apart, never assumed US.
    """
    if valued_positions is None or valued_positions.empty:
        return {"us_usd": 0.0, "unknown_usd": 0.0, "unknown": []}
    country = {}
    if securities is not None and not securities.empty:
        country = dict(zip(securities["ticker"], securities["issuer_country"]))
    us = unknown = 0.0
    unknown_tickers = []
    for row in valued_positions.to_dict("records"):
        value = row.get("market_value")
        if value is None or pd.isna(value):
            continue
        where = country.get(row["ticker"])
        if where == "US":
            us += float(value)
        elif not where:
            unknown += float(value)
            unknown_tickers.append(row["ticker"])
    return {"us_usd": us, "unknown_usd": unknown, "unknown": unknown_tickers}
