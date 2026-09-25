"""Lots and disposals in the owner's local currency (CLAUDE.md sections 9.9, 11). ESTIMADO.

Built on :func:`transform.portfolio.fifo_lots` — the FIFO engine already verified against
IBKR's own to below a millionth of a dollar — so there is one matching of sales to lots in
the project, not two. This module only converts.

**The convention, and why it is only a convention.** Each lot's cost is converted at the
official rate of its **purchase date** and frozen; each sale at the rate of its **sale
date**. That is the economic truth of what was paid and received in the local currency.
Whether the tax authority uses the same dates is question 4 of RESEARCH.md section 3, for the
accountant; the panel says so next to every figure and does not answer it.

**Why it matters even before any tax question.** With everything in USD and a life in the
local currency, the portfolio is a long USD position nobody decided (section 9.9). A loss in
dollars can be a gain in the local currency, and the reverse. The split below makes that
visible:

    asset part = (proceeds_usd − cost_usd) × rate_at_purchase     (what the asset did)
    fx part    =  proceeds_usd × (rate_at_sale − rate_at_purchase) (what the currency did)
    asset + fx = proceeds_usd × rate_at_sale − cost_usd × rate_at_purchase = local result

Pure functions. A date without a known rate gives ``None``, never a guessed rate (section 12).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def fx_series(observations: pd.DataFrame, series_id: str) -> pd.Series:
    """The stored rate as an ascending series indexed by its validity date."""
    if observations is None or observations.empty:
        return pd.Series(dtype=float)
    rows = observations[observations["series_id"] == series_id]
    if rows.empty:
        return pd.Series(dtype=float)
    series = pd.Series(rows["value"].astype(float).to_numpy(),
                       index=pd.to_datetime(rows["ts"].str[:10]))
    return series[~series.index.duplicated(keep="last")].sort_index()


def fx_asof(fx: pd.Series, when: Any) -> float | None:
    """The rate valid on ``when``'s date: the latest one starting on or before it.

    ``None`` before the first stored rate — a trade older than the history is not given the
    first rate available, which would be a rate from another day.
    """
    if fx.empty or when is None:
        return None
    day = pd.Timestamp(str(when)[:10])
    position = fx.index.searchsorted(day, side="right") - 1
    return None if position < 0 else float(fx.iloc[position])


def local_lots(open_lots: pd.DataFrame, fx: pd.Series) -> pd.DataFrame:
    """Open lots with their cost frozen in local currency at the purchase date's rate."""
    columns = ["ticker", "ts", "quantity", "cost_usd", "fx_cost", "cost_local"]
    if open_lots is None or open_lots.empty:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame({
        "ticker": open_lots["ticker"],
        "ts": open_lots["ts"],
        "quantity": open_lots["quantity"].astype(float),
        "cost_usd": open_lots["cost"].astype(float),   # commission included
    })
    out["fx_cost"] = [fx_asof(fx, ts) for ts in out["ts"]]
    out["cost_local"] = [None if r is None else c * r
                         for c, r in zip(out["cost_usd"], out["fx_cost"])]
    return out[columns].reset_index(drop=True)


def local_disposals(disposals: pd.DataFrame, fx: pd.Series) -> pd.DataFrame:
    """Each FIFO disposal in local currency, with its result split into asset and FX parts.

    Commissions follow their own leg: the buy commission raises the cost (at the purchase
    rate), the sell commission lowers the proceeds (at the sale rate).
    """
    columns = ["ticker", "ts", "acquired_ts", "holding_days", "quantity", "cost_usd",
               "proceeds_usd", "result_usd", "fx_cost", "fx_sale", "cost_local",
               "proceeds_local", "result_local", "asset_part_local", "fx_part_local"]
    if disposals is None or disposals.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for d in disposals.to_dict("records"):
        cost = float(d["cost"]) - float(d.get("buy_commission") or 0.0)
        proceeds = float(d["proceeds"]) + float(d.get("sell_commission") or 0.0)
        r_cost, r_sale = fx_asof(fx, d["acquired_ts"]), fx_asof(fx, d["ts"])
        known = r_cost is not None and r_sale is not None
        rows.append({
            "ticker": d["ticker"], "ts": d["ts"], "acquired_ts": d["acquired_ts"],
            "holding_days": d["holding_days"], "quantity": float(d["quantity"]),
            "cost_usd": cost, "proceeds_usd": proceeds, "result_usd": proceeds - cost,
            "fx_cost": r_cost, "fx_sale": r_sale,
            "cost_local": cost * r_cost if r_cost is not None else None,
            "proceeds_local": proceeds * r_sale if r_sale is not None else None,
            "result_local": proceeds * r_sale - cost * r_cost if known else None,
            "asset_part_local": (proceeds - cost) * r_cost if known else None,
            "fx_part_local": proceeds * (r_sale - r_cost) if known else None,
        })
    return pd.DataFrame(rows, columns=columns)


def valued_lots(lots: pd.DataFrame, prices: dict[str, float], fx_now: float | None) -> pd.DataFrame:
    """Open lots at today's price and rate, with the unrealized result split the same way."""
    out = lots.copy()
    if out.empty:
        return out.assign(value_usd=[], value_local=[], result_usd=[], result_local=[],
                          asset_part_local=[], fx_part_local=[])
    price = out["ticker"].map(prices).astype(float)
    out["value_usd"] = out["quantity"] * price
    out["result_usd"] = out["value_usd"] - out["cost_usd"]
    rate = pd.to_numeric(out["fx_cost"], errors="coerce")
    now = np.nan if fx_now is None else float(fx_now)
    out["value_local"] = out["value_usd"] * now
    out["result_local"] = out["value_local"] - pd.to_numeric(out["cost_local"], errors="coerce")
    out["asset_part_local"] = out["result_usd"] * rate
    out["fx_part_local"] = out["value_usd"] * (now - rate)
    return out
