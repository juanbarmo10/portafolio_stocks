"""The behaviour mirror: what the plan says against what the account did (CLAUDE.md §15.4).

Section 2 names the project's first risk: it is behavioural, not technical. The panel had the
pieces — the journal, the performance against SPY, the cost of cash — scattered on three
pages, and none of them put the pattern in front of the investor. This module measures it,
over a window, from the account alone:

- **Turnover**: ``min(purchases, sales) / average NAV`` — the standard definition (it
  leaves out money that arrives and is simply invested). An investor who contributes and
  never sells turns over ~0.
- **Holding period** of what was sold (FIFO lots, weighted by proceeds), against the
  horizon the plan writes down (``min_holding_days``).
- **Disposition effect**: sales at a gain against sales at a loss, and open positions at a
  loss. Selling winners early and keeping losers is the most documented retail bias (Shefrin
  and Statman, 1985; Odean, 1998). With a handful of sales it is a pattern to look at, not a
  statistic.
- **What selling cost**: each sale's value times what the sold stock did afterwards (total
  return, to ``as_of``), and the same money in SPY. Positive = the shares kept rising after
  they were sold.
- **Cash**: average and current share of the account, and **commissions** over average NAV.

Nothing here judges or recommends: the page puts each figure next to what the plan says.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from transform import portfolio


@dataclass(frozen=True)
class Mirror:
    """The account's behaviour over ``[start, end]``. See the module docstring."""

    start: str
    end: str
    average_nav: float | None
    purchases: float
    sales: float
    turnover: float | None
    holding_median_days: float | None
    sold_before_horizon: float | None       # share of sold value held < min_holding_days
    min_holding_days: int
    sells_at_gain: int
    sells_at_loss: int
    open_at_gain: int
    open_at_loss: int
    after_sale: float | None                # what the sold shares did since, in USD
    after_sale_spy: float | None            # the same money in SPY
    cash_share_average: float | None
    cash_share_now: float | None
    commission_drag: float | None


def _series(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    out = pd.Series(frame["value"].astype(float).to_numpy(),
                    index=pd.to_datetime(frame["ts"].astype(str).str[:10]))
    return out[~out.index.duplicated(keep="last")].sort_index()


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float | None:
    if not len(values) or weights.sum() <= 0:
        return None
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def _since(series: pd.Series | None, day: pd.Timestamp, end: pd.Timestamp) -> float | None:
    if series is None or series.empty:
        return None
    start_v = series[series.index <= day]
    end_v = series[series.index <= end]
    if start_v.empty or end_v.empty or start_v.iloc[-1] <= 0:
        return None
    return float(end_v.iloc[-1] / start_v.iloc[-1] - 1)


def mirror(trades: pd.DataFrame, nav: pd.DataFrame, nav_cash: pd.DataFrame,
           valued: pd.DataFrame, indices: Mapping[str, pd.Series], as_of: Any, *,
           window_days: int = 365, min_holding_days: int = 90) -> Mirror:
    """See :class:`Mirror`. ``indices``: total-return series per sold ticker and ``SPY``."""
    end = pd.Timestamp(as_of).normalize()
    start = end - pd.Timedelta(days=window_days)
    total = _series(nav)
    total = total[(total.index > start) & (total.index <= end)]
    average_nav = float(total.mean()) if len(total) else None

    frame = trades.copy() if trades is not None else pd.DataFrame()
    purchases = sales = 0.0
    if not frame.empty:
        frame["day"] = pd.to_datetime(frame["ts"].astype(str).str[:10])
        frame = frame[(frame["day"] > start) & (frame["day"] <= end)]
        value = frame["quantity"].astype(float) * frame["price"].astype(float)
        side = frame["side"].str.lower()
        purchases, sales = float(value[side == "buy"].sum()), float(value[side == "sell"].sum())
        commissions = float(pd.to_numeric(frame["commission"], errors="coerce").fillna(0).sum())
    else:
        commissions = 0.0

    # Holding periods and gain/loss of what was sold, from the FIFO engine (all trades, so a
    # sale in the window is matched against a purchase before it).
    _lots, disposals, _unmatched = portfolio.fifo_lots(trades) if trades is not None \
        and not trades.empty else (None, pd.DataFrame(), [])
    held_median = before = None
    at_gain = at_loss = 0
    after = after_spy = None
    if not disposals.empty:
        disposals = disposals.assign(day=pd.to_datetime(disposals["ts"].astype(str).str[:10]))
        disposals = disposals[(disposals["day"] > start) & (disposals["day"] <= end)]
    if not disposals.empty:
        days = disposals["holding_days"].astype(float).to_numpy()
        weights = disposals["proceeds"].astype(float).to_numpy()
        held_median = _weighted_median(days, weights)
        before = float(weights[days < min_holding_days].sum() / weights.sum()) \
            if weights.sum() > 0 else None
        # One sale = one ticker on one day, however many lots it closed.
        per_sale = disposals.groupby(["day", "ticker"]).agg(
            pnl=("realized_pnl", "sum"), proceeds=("proceeds", "sum")).reset_index()
        at_gain = int((per_sale["pnl"] > 0).sum())
        at_loss = int((per_sale["pnl"] < 0).sum())
        moves = [(_since(indices.get(r.ticker), r.day, end),
                  _since(indices.get("SPY"), r.day, end), r.proceeds)
                 for r in per_sale.itertuples()]
        if all(m is not None for m, _, _ in moves):
            after = float(sum(m * v for m, _, v in moves))
        if all(s is not None for _, s, _ in moves):
            after_spy = float(sum(s * v for _, s, v in moves))

    open_gain = open_loss = 0
    if valued is not None and not valued.empty and "unrealized_pnl" in valued:
        pnl = pd.to_numeric(valued["unrealized_pnl"], errors="coerce").dropna()
        open_gain, open_loss = int((pnl > 0).sum()), int((pnl < 0).sum())

    cash = _series(nav_cash)
    share = (cash / _series(nav).where(_series(nav) > 0)).dropna()
    share = share[(share.index > start) & (share.index <= end)]
    return Mirror(
        start=start.date().isoformat(), end=end.date().isoformat(), average_nav=average_nav,
        purchases=purchases, sales=sales,
        turnover=min(purchases, sales) / average_nav if average_nav else None,
        holding_median_days=held_median, sold_before_horizon=before,
        min_holding_days=min_holding_days, sells_at_gain=at_gain, sells_at_loss=at_loss,
        open_at_gain=open_gain, open_at_loss=open_loss,
        after_sale=after, after_sale_spy=after_spy,
        cash_share_average=float(share.mean()) if len(share) else None,
        cash_share_now=float(share.iloc[-1]) if len(share) else None,
        commission_drag=abs(commissions) / average_nav if average_nav else None,
    )
