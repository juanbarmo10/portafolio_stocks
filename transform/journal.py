"""The decision journal (CLAUDE.md §15.1.10): judge the decision, not only the outcome.

A good result from a bad decision teaches the wrong lesson, and so does a bad result from a
good one. The only defence is to have written down, **at the time**, why and what was
expected — and to read it back later against what happened.

Two halves, deliberately separate:

- **What the panel knows by itself**, from the account and point-in-time data: each
  decision (the IBKR executions of one ticker, one side, one day), what the traffic light
  said that day, and what the stock has done since against SPY. Nothing here is typed by
  hand (section 5.1: the account is never written by hand).
- **What only the user can write**, in ``settings.local.yaml`` under ``journal``: why, what
  was expected and by when, and the pre-mortem — "if this goes wrong, it will have been
  because…". An entry is matched to its decision by ticker and date (within
  ``tolerance_days``, because the note may be written the evening before).

A decision without an entry is flagged, not hidden: a position that exists before its
written reason is exactly what section 5.1 warns about.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

DECISION_COLUMNS = ["date", "ticker", "side", "quantity", "value", "price", "commission",
                    "executions"]


def decisions(trades: pd.DataFrame) -> pd.DataFrame:
    """The account's decisions: executions grouped by day, ticker and side.

    IBKR splits one order into several executions (fractional shares, partial fills); what
    was decided is the day's buy or sell of a ticker. ``price`` is the value-weighted
    average.
    """
    if trades is None or trades.empty:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    frame = trades.assign(
        date=trades["ts"].astype(str).str[:10],
        side=trades["side"].str.lower(),
        value=trades["quantity"].astype(float) * trades["price"].astype(float),
        commission=pd.to_numeric(trades["commission"], errors="coerce").fillna(0.0),
    )
    grouped = frame.groupby(["date", "ticker", "side"]).agg(
        quantity=("quantity", "sum"), value=("value", "sum"),
        commission=("commission", "sum"), executions=("trade_id", "count"),
    ).reset_index()
    grouped["price"] = grouped["value"] / grouped["quantity"].where(grouped["quantity"] > 0)
    return grouped[DECISION_COLUMNS].sort_values("date", ascending=False).reset_index(drop=True)


def match_entries(decisions_frame: pd.DataFrame, entries: Sequence[Mapping[str, Any]],
                  tolerance_days: int = 3) -> pd.DataFrame:
    """Attach each written entry to its decision: same ticker, date within the tolerance.

    Adds ``entry`` (the mapping, or ``None``). Entries that match no decision are returned
    as rows with ``side = 'sin ejecutar'`` — a plan written and not (yet) carried out is
    worth reading back too.
    """
    out = decisions_frame.copy()
    out["entry"] = None
    used: set[int] = set()
    for idx, row in out.iterrows():
        day = pd.Timestamp(row["date"])
        for i, entry in enumerate(entries or []):
            if i in used or str(entry.get("ticker", "")).upper() != str(row["ticker"]).upper():
                continue
            when = pd.Timestamp(str(entry.get("date"))[:10])
            if abs((when - day).days) <= tolerance_days:
                out.at[idx, "entry"] = dict(entry)
                used.add(i)
                break
    unmatched = [
        {"date": str(e.get("date"))[:10], "ticker": str(e.get("ticker", "")).upper(),
         "side": "sin ejecutar", "quantity": None, "value": None, "price": None,
         "commission": None, "executions": 0, "entry": dict(e)}
        for i, e in enumerate(entries or []) if i not in used
    ]
    if unmatched:
        out = pd.concat([out, pd.DataFrame(unmatched)], ignore_index=True)
    return out.sort_values("date", ascending=False).reset_index(drop=True)


def since(series: pd.Series, day: Any, until: Any) -> float | None:
    """Return of ``series`` from the close of ``day`` (or the last before it) to ``until``."""
    if series is None or series.empty:
        return None
    start = series[series.index <= pd.Timestamp(day)]
    end = series[series.index <= pd.Timestamp(until)]
    if start.empty or end.empty or start.iloc[-1] <= 0:
        return None
    return float(end.iloc[-1] / start.iloc[-1] - 1)


def verdict_on(frame: pd.DataFrame | None, day: Any) -> str | None:
    """The traffic light's verdict in force on ``day`` (the last one on or before it)."""
    if frame is None or frame.empty:
        return None
    rows = frame[frame.index <= pd.Timestamp(day)]
    return None if rows.empty else str(rows["verdict"].iloc[-1])


def review_due(entry: Mapping[str, Any] | None, today: Any) -> bool:
    """Whether the entry's own ``review_after`` date has come."""
    if not entry or not entry.get("review_after"):
        return False
    return pd.Timestamp(str(entry["review_after"])[:10]) <= pd.Timestamp(today)
