"""A company against the peers the user chose (🏢 Empresa, CLAUDE.md §15.5, point 11).

The comparison by SIC code (``transform.screen.peer_comparison``) fails on companies whose
code says little about who they compete with: HIMS is "offices of doctors" (8011), NU a
foreign IFRS filer with no SEC figures, UBER a catch-all code. Here the user writes the group (``universe.peers``) and each company is read
from the growth screen (``transform.growth_screen``: its latest reported quarter against the
same one a year earlier, in USD).

A peer the growth screen cannot read — a foreign filer reporting in US GAAP but only once a
year and in its own currency, such as DiDi — is read from its fiscal years instead
(``ingest/peers``, :func:`annual_table`). Every metric shown is a ratio, so the currency
cancels; the row says its period ("FY2025") so a year is never mistaken for a quarter.

The group's **median** comes with the table: without it, "gross margin 64 %" says nothing.
Nothing here says which company is better — more growth and more dilution do not read in the
same direction (section 12).

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from transform.growth_screen import COLUMNS
from transform.macro import point_in_time
from transform.screen import _ratio

# Shown, in this order, with the median of the group.
METRICS = ["revenue_growth", "acceleration", "gross_margin", "operating_margin",
           "incremental_margin", "fcf_margin", "sbc_over_revenue", "dilution", "rule_of_40",
           "price_to_sales", "market_cap"]
YEAR_DAYS = (330, 400)   # a previous fiscal year ends ~365 days before


def _latest(observations: pd.DataFrame, as_of: Any) -> dict[str, dict[str, pd.Series]]:
    """``{cik: {metric: series by fiscal-year end}}``, point-in-time at ``as_of``."""
    out: dict[str, dict[str, pd.Series]] = {}
    if observations is None or observations.empty:
        return out
    known = point_in_time(observations, as_of)
    for sid, rows in known.groupby("series_id"):
        cik, metric, _ = str(sid).split(":")
        values = pd.Series(rows["value"].astype(float).to_numpy(),
                           index=pd.to_datetime(rows["ts"].astype(str).str[:10]))
        out.setdefault(cik, {})[metric] = values[~values.index.duplicated(keep="last")] \
            .sort_index()
    return out


def annual_table(observations: pd.DataFrame, registry: Mapping[str, tuple[str, str]],
                 as_of: Any) -> pd.DataFrame:
    """Growth-screen-shaped rows (``transform.growth_screen.COLUMNS``) from fiscal years:
    the latest year against the one before. No price and no quarter, so acceleration,
    market cap and P/sales stay unknown."""
    rows = []
    for cik, metrics in _latest(observations, as_of).items():
        revenue = metrics.get("revenue", pd.Series(dtype=float))
        if revenue.empty:
            continue
        end = revenue.index[-1]
        earlier = [d for d in revenue.index
                   if YEAR_DAYS[0] <= (end - d).days <= YEAR_DAYS[1]]
        prev = earlier[-1] if earlier else None

        def at(metric: str, day: pd.Timestamp | None) -> float | None:
            values = metrics.get(metric)
            return None if values is None or day is None or day not in values.index \
                else float(values[day])

        growth = None if prev is None else _ratio(at("revenue", end), at("revenue", prev))
        growth = None if growth is None else growth - 1
        ebit, ebit_prev = at("operating_income", end), at("operating_income", prev)
        rev, rev_prev = at("revenue", end), at("revenue", prev)
        ocf, capex = at("operating_cash_flow", end), at("capex", end)
        fcf = None if ocf is None or capex is None else ocf - capex
        fcf_margin = _ratio(fcf, rev)
        shares = _ratio(at("basic_shares", end), at("basic_shares", prev))
        ticker, name = registry.get(cik, (None, None))
        row = dict.fromkeys(COLUMNS)
        row.update({
            "cik": cik, "ticker": ticker, "name": name, "quarter": f"FY{end.year}",
            "quarter_end": end.date().isoformat(), "stale": False, "revenue_q": rev,
            "revenue_growth": growth,
            "gross_margin": _ratio(at("gross_profit", end), rev),
            "operating_margin": _ratio(ebit, rev),
            "incremental_margin": (None if ebit is None or ebit_prev is None
                                   or rev is None or rev_prev is None
                                   else _ratio(ebit - ebit_prev, rev - rev_prev)),
            "dilution": None if shares is None else shares - 1, "share_jump": False,
            "revenue_annual": rev, "fcf_annual": fcf, "fcf_margin": fcf_margin,
            "sbc_over_revenue": _ratio(at("sbc", end), rev),
            "rule_of_40": None if growth is None or fcf_margin is None else growth + fcf_margin,
            "cap_suspect": False,
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def peer_table(growth: pd.DataFrame, annual: pd.DataFrame, ticker: str,
               peers: Sequence[str]) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """``(rows, median)``: the company and its peers, in the order written, and the median of
    the **peers** (the company excluded) for each of :data:`METRICS`.

    ``rows`` has the growth-screen columns plus ``role`` (``empresa`` / ``par``) and
    ``found``; a ticker read by neither source keeps a row with ``found = False``, so a peer
    that silently vanished would be seen as missing rather than simply absent.
    """
    frames = [f for f in (growth, annual) if f is not None and not f.empty]
    pool = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    pool = pool.drop_duplicates(subset=["ticker"], keep="first")   # the quarter wins
    by_ticker = {str(t): r for t, r in zip(pool["ticker"], pool.to_dict("records"))}
    rows = []
    for i, t in enumerate([ticker, *peers]):
        found = t in by_ticker
        row = dict(by_ticker[t]) if found else {**dict.fromkeys(COLUMNS), "ticker": t}
        rows.append({**row, "role": "empresa" if i == 0 else "par", "found": found})
    table = pd.DataFrame(rows)
    # A bank's operating cash flow carries the loans it grants: its "free cash flow" is not
    # an industrial's (the annual screen sets financials apart for the same reason). SoFi
    # read −643 % of revenue. Unknown, not a number that means something else.
    if "financial" in table:
        banks = table["financial"].astype("boolean").fillna(False).astype(bool)
        table.loc[banks, ["fcf_margin", "rule_of_40"]] = None
    group = table[(table["role"] == "par") & table["found"]]
    median = {}
    for metric in METRICS:
        values = pd.to_numeric(group[metric], errors="coerce").dropna() if metric in group \
            else pd.Series(dtype=float)
        median[metric] = float(values.median()) if len(values) else None
    return table, median
