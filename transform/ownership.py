"""Who owns and who trades the stock: institutions (13F) and retail participation (FINRA).

User request, 2026-09-29: "how the whales and retail move". Two views on purpose, never one
chart (different frequencies, units and meaning):

- **Institutional** (``ingest/institutional``): shares held by 13F managers at each quarter
  end, point-in-time (each manager's latest version filed by the date), how many managers,
  the share of the company that represents, and who added or cut the most in the last
  quarter. Holdings, not trades: the change between quarters is net institutional buying.
  A quarter less than ``complete_days`` old is still being filed (deadline: 45 days) and is
  marked so — its total would read as selling that has not happened.
- **Retail participation** (``ingest/otc_volume``): each week's off-exchange, non-ATS volume
  over total volume. It measures **how much** retail trades, never whether it buys or sells.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from transform.macro import point_in_time


@dataclass(frozen=True)
class Institutional:
    """``quarters``: ``[quarter, shares, holders, share_of_company, change, complete]``;
    ``movers``: ``[filer_cik, name, before, after, change]`` of the latest complete quarter."""

    quarters: pd.DataFrame
    movers: pd.DataFrame
    latest: str | None
    notes: list[str] = field(default_factory=list)


def institutional(observations: pd.DataFrame, cik: str, as_of: Any,
                  shares_outstanding: pd.Series | None = None,
                  names: Mapping[str, str] | None = None,
                  complete_days: int = 50) -> Institutional | None:
    """See :class:`Institutional`. ``shares_outstanding``: basic shares by date, for the share
    of the company (``None`` for a company with no SEC figures, such as NU)."""
    if observations is None or observations.empty:
        return None
    rows = observations[observations["series_id"].astype(str).str.startswith(f"{cik}:inst:")]
    if rows.empty:
        return None
    known = point_in_time(rows, as_of)
    if known.empty:
        return None
    known = known.assign(filer=known["series_id"].str.split(":").str[2],
                         quarter=known["ts"].astype(str).str[:10],
                         value=known["value"].astype(float))
    day = pd.Timestamp(as_of)
    table = []
    for quarter, group in known.groupby("quarter"):
        held = group[group["value"] > 0]
        outstanding = None
        if shares_outstanding is not None and not shares_outstanding.empty:
            before = shares_outstanding[shares_outstanding.index <= pd.Timestamp(quarter)]
            outstanding = float(before.iloc[-1]) if len(before) else None
        total = float(held["value"].sum())
        table.append({"quarter": quarter, "shares": total, "holders": int(len(held)),
                      "share_of_company": total / outstanding if outstanding else None,
                      "complete": (day - pd.Timestamp(quarter)).days >= complete_days})
    quarters = pd.DataFrame(table).sort_values("quarter").reset_index(drop=True)
    quarters["change"] = quarters["shares"].diff()
    done = quarters[quarters["complete"]]
    movers = pd.DataFrame(columns=["filer_cik", "name", "before", "after", "change",
                                   "entity_change"])
    notes = []
    if len(done) >= 2:
        last, prev = done.iloc[-1]["quarter"], done.iloc[-2]["quarter"]
        after = known[known["quarter"] == last].set_index("filer")["value"]
        before = known[known["quarter"] == prev].set_index("filer")["value"]
        both = pd.concat([before.rename("before"), after.rename("after")], axis=1).fillna(0.0)
        both["change"] = both["after"] - both["before"]
        both = both[both["change"] != 0].rename_axis("filer_cik").reset_index()
        both["name"] = both["filer_cik"].map(lambda c: (names or {}).get(c, c))
        both["entity_change"] = entity_changes(both)
        movers = both[["filer_cik", "name", "before", "after", "change", "entity_change"]]
    if not quarters.empty and not bool(quarters.iloc[-1]["complete"]):
        notes.append(f"El trimestre al {quarters.iloc[-1]['quarter']} todavía se está "
                     "presentando (plazo: 45 días): su total está incompleto.")
    return Institutional(quarters=quarters, movers=movers,
                         latest=str(done.iloc[-1]["quarter"]) if len(done) else None,
                         notes=notes)


def _first_word(name: str) -> str:
    words = str(name).upper().replace(",", " ").split()
    return words[0] if words else ""


def entity_changes(movers: pd.DataFrame, tolerance: float = 0.2) -> pd.Series:
    """Rows that look like one manager moving its holding to another filing entity: a full
    exit and a new position whose sizes are within ``tolerance`` and whose names start with
    the same word (Pershing Square, 2026). Flagged, never merged: the name is a clue, not an
    identity (section 9.3)."""
    flags = pd.Series(False, index=movers.index)
    exits = movers[(movers["after"] == 0) & (movers["before"] > 0)]
    entries = movers[(movers["before"] == 0) & (movers["after"] > 0)]
    for i, out in exits.iterrows():
        for j, inn in entries.iterrows():
            if (_first_word(out["name"]) == _first_word(inn["name"])
                    and abs(inn["after"] - out["before"]) <= tolerance * out["before"]):
                flags[i] = flags[j] = True
    return flags


def retail_participation(finra: pd.DataFrame, volume: pd.Series, ticker: str,
                         as_of: Any) -> pd.DataFrame:
    """``[week, otc_share, ats_share, total]``: each week's off-exchange non-ATS and ATS
    volume over the total volume of its sessions (``volume``: daily shares, by date).
    Point-in-time on FINRA's publication date."""
    columns = ["week", "otc_share", "ats_share", "total"]
    if finra is None or finra.empty or volume is None or volume.empty:
        return pd.DataFrame(columns=columns)
    rows = finra[finra["series_id"].isin([f"{ticker}:otc_nonats:w", f"{ticker}:ats:w"])]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    known = point_in_time(rows, as_of)
    wide = known.assign(kind=known["series_id"].str.split(":").str[1],
                        week=pd.to_datetime(known["ts"].astype(str).str[:10])) \
        .pivot_table(index="week", columns="kind", values="value", aggfunc="last")
    out = []
    for week, r in wide.iterrows():
        total = float(volume[(volume.index >= week)
                             & (volume.index < week + pd.Timedelta(days=7))].sum())
        if total <= 0:
            continue
        out.append({"week": week, "otc_share": r.get("otc_nonats", float("nan")) / total,
                    "ats_share": r.get("ats", float("nan")) / total, "total": total})
    return pd.DataFrame(out, columns=columns)
