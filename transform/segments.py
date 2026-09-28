"""Business segments: where the growth is (🏢 Empresa, CLAUDE.md §15.5, point 15).

Built on ``ingest/segments.py`` (the XBRL of each filing). Point-in-time: every figure is
visible from the filing that carried it, and the latest version at the date wins.

Derivations that belong here, as for the fundamentals (sections 9.12, 9.14):

- **Quarters hidden in cumulatives**: ``Q2 = YTD2 − Q1``, ``Q3 = YTD3 − YTD2`` and
  ``Q4 = FY − YTD3`` (Uber reports its fourth quarter only inside the year), each dated by
  the later of the two filings and only when both periods belong to the same fiscal year.
  A quarter the company filed on its own always wins.
- **One profit measure per series.** A company's segment measure is its own and can change:
  Uber moved from *Segment Adjusted EBITDA* to *Segment Operating Income* in 2026. The measure
  used is the one of the **latest** quarter; a year-on-year change is computed only within it,
  and the older measure is named, never spliced (section 9.8).
- **Restated comparatives**: the same segment, concept and quarter carried with another value
  by a later filing. The latest wins; how many there were is said.

A company that tags a single segment (the 2024 rule makes single-segment filers tag theirs)
is reported as such: its "segment" is the company.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from transform.macro import point_in_time

QUARTER_DAYS = (80, 100)       # between the end of a cumulative and the next one
YEAR_DAYS = (350, 380)


@dataclass(frozen=True)
class SegmentView:
    """Segments of one company as knowable at ``as_of``.

    Attributes:
        members: Segment names, largest revenue first.
        quarter: End of the latest quarter with segment revenue.
        table: ``[member, revenue, revenue_growth, share, profit, margin, profit_growth]``
            at that quarter.
        history: ``[member, date, revenue, derived]`` — the quarterly revenue series.
        profit_concept / older_concepts: The profit measure of the latest quarter, and the
            ones used before it (a change of measure).
        restated: How many segment figures a later filing carried with another value.
    """

    members: list[str]
    quarter: str | None
    table: pd.DataFrame
    history: pd.DataFrame
    profit_concept: str | None
    older_concepts: list[str] = field(default_factory=list)
    restated: int = 0


def _parse(observations: pd.DataFrame, cik: str) -> pd.DataFrame:
    rows = observations[observations["series_id"].astype(str).str.startswith(f"{cik}:")]
    if rows.empty:
        return pd.DataFrame(columns=["member", "concept", "label", "ts", "ts_release", "value"])
    parts = rows["series_id"].astype(str).str.split(":", expand=True)
    return rows.assign(member=parts[1], concept=parts[2], label=parts[3],
                       ts=pd.to_datetime(rows["ts"].astype(str).str[:10]))


def _restated(frame: pd.DataFrame) -> int:
    """Figures carried by more than one filing with values more than 0,5 % apart."""
    if frame.empty:
        return 0
    spread = frame.groupby(["member", "concept", "label", "ts"])["value"].agg(["min", "max"])
    scale = spread[["min", "max"]].abs().max(axis=1).clip(lower=1.0)
    return int(((spread["max"] - spread["min"]) / scale > 0.005).sum())


def quarterly(known: pd.DataFrame) -> pd.DataFrame:
    """``[member, concept, ts, ts_release, value, derived]`` quarterly rows from the latest
    versions: filed quarters plus those derived from cumulatives (module docstring)."""
    out = []
    for (member, concept), group in known.groupby(["member", "concept"]):
        by_label = {label: g.set_index("ts") for label, g in group.groupby("label")}
        quarters = by_label.get("q", pd.DataFrame(columns=["value", "ts_release"]))
        rows = {ts: (float(r["value"]), str(r["ts_release"]), False)
                for ts, r in quarters.iterrows()}
        # (longer, shorter): the quarter ending with `longer` is longer − shorter.
        for longer, shorter in (("ytd2", "q"), ("ytd3", "ytd2"), ("fy", "ytd3")):
            if longer not in by_label or shorter not in by_label:
                continue
            short = by_label[shorter]
            for end, r in by_label[longer].iterrows():
                if end in rows:
                    continue
                earlier = [d for d in short.index
                           if QUARTER_DAYS[0] <= (end - d).days <= QUARTER_DAYS[1]]
                if len(earlier) != 1:
                    continue
                s = short.loc[earlier[0]]
                rows[end] = (float(r["value"]) - float(s["value"]),
                             max(str(r["ts_release"]), str(s["ts_release"])), True)
        out += [{"member": member, "concept": concept, "ts": ts, "ts_release": rel,
                 "value": v, "derived": d} for ts, (v, rel, d) in rows.items()]
    return pd.DataFrame(out, columns=["member", "concept", "ts", "ts_release", "value",
                                      "derived"])


def _year_ago(series: pd.Series, day: pd.Timestamp) -> float | None:
    earlier = [d for d in series.index if YEAR_DAYS[0] <= (day - d).days <= YEAR_DAYS[1]]
    return float(series[earlier[-1]]) if earlier else None


def _growth(now: float | None, before: float | None) -> float | None:
    return None if now is None or before is None or before <= 0 else now / before - 1


def assess(observations: pd.DataFrame, cik: str, as_of: Any,
           revenue_concepts: Sequence[str], profit_concepts: Sequence[str]) -> SegmentView | None:
    """See :class:`SegmentView`. ``None`` when the company tagged no segment revenue."""
    frame = _parse(observations, cik)
    if frame.empty:
        return None
    visible = frame[frame["ts_release"].astype(str) <= pd.Timestamp(as_of).date().isoformat()]
    restated = _restated(visible)
    known = point_in_time(visible.assign(series_id=visible["member"] + "|" + visible["concept"]
                                         + "|" + visible["label"],
                                         ts=visible["ts"].dt.date.astype(str)), as_of)
    if known.empty:
        return None
    parts = known["series_id"].str.split("|", expand=True)
    known = known.assign(member=parts[0], concept=parts[1], label=parts[2],
                         ts=pd.to_datetime(known["ts"]))
    q = quarterly(known)
    revenue = q[q["concept"].isin(revenue_concepts)]
    if revenue.empty:
        return None
    # One revenue concept per segment: the first by preference that it filed.
    rank = {c: i for i, c in enumerate(revenue_concepts)}
    revenue = revenue.assign(rank=revenue["concept"].map(rank)).sort_values("rank") \
        .drop_duplicates(["member", "ts"], keep="first")
    quarter = revenue["ts"].max()
    latest = revenue[revenue["ts"] == quarter]
    total = latest["value"].sum()
    profits = q[q["concept"].isin(profit_concepts)]
    current = profits[profits["ts"] == quarter]
    concept = None
    for c in profit_concepts:
        if (current["concept"] == c).any():
            concept = c
            break
    older = sorted(set(profits.loc[profits["concept"] != concept, "concept"])) if concept else []
    table = []
    for member in latest.sort_values("value", ascending=False)["member"]:
        series = revenue[revenue["member"] == member].set_index("ts")["value"].sort_index()
        now = float(series[quarter])
        profit_series = (profits[(profits["member"] == member) & (profits["concept"] == concept)]
                         .set_index("ts")["value"].sort_index() if concept else pd.Series())
        profit = float(profit_series[quarter]) if quarter in profit_series.index else None
        table.append({
            "member": member, "revenue": now,
            "revenue_growth": _growth(now, _year_ago(series, quarter)),
            "share": now / total if total > 0 else None,
            "profit": profit,
            "margin": None if profit is None or now <= 0 else profit / now,
            "profit_growth": (None if profit is None else
                              _growth(profit, _year_ago(profit_series, quarter))),
        })
    history = revenue[["member", "ts", "value", "derived"]].rename(
        columns={"ts": "date", "value": "revenue"}).sort_values(["date", "member"])
    return SegmentView(
        members=[r["member"] for r in table], quarter=quarter.date().isoformat(),
        table=pd.DataFrame(table), history=history.reset_index(drop=True),
        profit_concept=concept, older_concepts=older, restated=restated,
    )


def labels(profit_labels: Mapping[str, str], concept: str | None) -> str:
    """The on-screen name of a profit concept, or the concept itself when unknown."""
    return "" if concept is None else str(profit_labels.get(concept, concept))
