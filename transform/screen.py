"""The quarterly screen: quality, shareholder value accrual and price, across ~500 companies.

Feeds question 3 of section 2 from the other end: not "is this thesis alive?" but "which
companies deserve a thesis?". Its output is a list of candidates to **study** on the company
page — never a buy list. It runs once a quarter on audited annual figures (``ingest/screen``)
and nothing on it is fresher than the last 10-K, on purpose (sections 2, 12).

Every ratio follows the project's rules:

- **No value over a non-positive base** (section 12, THEORY §4): growth over negative
  revenue, a multiple over negative EBIT and cash conversion over a loss are ``None``, not
  a number that reads as cheap or as the worst case.
- **Dilution is the net measure of value returned** (section 2): gross buybacks are left
  out on purpose, because SBC can cancel them.
- **Market cap uses today's raw close and the filed share count brought to today's share
  units** — a split after the fiscal year would otherwise divide the cap by its ratio
  (section 9.1, same rule as ``transform.valuation``).
- **Unknown is neither pass nor fail.** A filter keeps the companies that pass, drops the
  ones that fail, and counts the ones it cannot judge separately, so the page can say how
  many were left out for lack of data instead of hiding them.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from transform.valuation import split_factor_between

PERIOD = re.compile(r"^CY(\d{4})$")


def latest_year(frames: pd.DataFrame) -> int | None:
    """The newest ``CY{year}`` flow period in the screen's observations."""
    if frames.empty:
        return None
    years = [int(m.group(1)) for sid in frames["series_id"].astype(str)
             if (m := PERIOD.match(sid.rsplit(":", 1)[-1]))]
    return max(years) if years else None


def _pivot(frames: pd.DataFrame, year: int) -> pd.DataFrame:
    """``cik`` × ``{metric}`` (this year), ``{metric}_prev`` (the year before) and the
    fiscal end of this year's revenue and share count."""
    parts = frames["series_id"].astype(str).str.split(":", n=2, expand=True)
    frame = frames.assign(cik=parts[0], metric=parts[1], period=parts[2])
    current = {f"CY{year}": "", f"CY{year}Q4I": "", f"CY{year - 1}": "_prev"}
    frame = frame[frame["period"].isin(current)]
    frame = frame.assign(column=frame["metric"] + frame["period"].map(current))
    wide = frame.pivot_table(index="cik", columns="column", values="value", aggfunc="last")
    ends = frame[frame["column"] == "diluted_shares"].set_index("cik")["ts"].astype(str)
    return wide.assign(shares_end=ends.str[:10].reindex(wide.index))


def _ratio(numerator: Any, denominator: Any, *, positive_base: bool = True) -> float | None:
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator):
        return None
    if denominator == 0 or (positive_base and denominator < 0):
        return None
    return float(numerator) / float(denominator)


def _value(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    return None if value is None or pd.isna(value) else float(value)


def last_closes(closes: pd.DataFrame) -> dict[str, tuple[float, str]]:
    """``{ticker: (last raw close, its date)}`` from ``[series_id, ts, value]`` rows."""
    if closes.empty:
        return {}
    frame = closes.assign(ticker=closes["series_id"].astype(str).str.split(":").str[0],
                          day=closes["ts"].astype(str).str[:10])
    frame = frame.dropna(subset=["value"]).sort_values("day").groupby("ticker").tail(1)
    return {r.ticker: (float(r.value), r.day) for r in frame.itertuples()}


def screen_table(frames: pd.DataFrame, closes: pd.DataFrame, actions: Any,
                 registry: Mapping[str, tuple[str, str | None]]) -> pd.DataFrame:
    """One row per company with its metrics for the newest fiscal year in ``frames``.

    Args:
        frames: The ``sec_frames`` observations.
        closes: Recent raw closes, ``[series_id, ts, value]``.
        actions: ``corporate_actions`` rows, for the split between filing and price.
        registry: ``{cik: (ticker, name)}``.
    """
    year = latest_year(frames)
    if year is None:
        return pd.DataFrame()
    wide = _pivot(frames, year)
    prices = last_closes(closes)
    rows = []
    for cik, row in wide.to_dict("index").items():
        ticker, name = registry.get(cik, (None, None))
        revenue = _value(row, "revenue")
        ocf, capex = _value(row, "operating_cash_flow"), _value(row, "capex")
        fcf = None if ocf is None or capex is None else ocf - capex
        sbc, shares = _value(row, "sbc"), _value(row, "diluted_shares")
        net_income, ebit = _value(row, "net_income"), _value(row, "operating_income")
        debt, cash = _value(row, "long_term_debt"), _value(row, "cash")

        # The membership list writes class shares with a dot (BRK.B), the price source with
        # a dash (BRK-B).
        price, price_day = (prices.get(ticker) or prices.get(ticker.replace(".", "-"))
                            or (None, None)) if ticker else (None, None)
        market_cap = None
        if price is not None and shares is not None and row.get("shares_end"):
            factor = split_factor_between(actions, ticker, row["shares_end"], price_day)
            market_cap = price * shares * factor
        # No long-term debt concept filed is read as none — the convention of
        # transform.valuation, flagged in its own column; missing cash leaves EV unknown.
        enterprise_value = (None if market_cap is None or cash is None
                            else market_cap + (debt or 0.0) - cash)
        rows.append({
            "cik": cik, "ticker": ticker, "name": name, "fiscal_year": year,
            "revenue": revenue,
            "revenue_growth": (None if (g := _ratio(revenue, _value(row, "revenue_prev")))
                               is None else g - 1),
            "operating_margin": _ratio(ebit, revenue),
            "fcf": fcf,
            "fcf_margin": _ratio(fcf, revenue),
            "cash_conversion": _ratio(fcf, net_income),
            "sbc_over_revenue": _ratio(sbc, revenue),
            "dilution": (None if (d := _ratio(shares, _value(row, "diluted_shares_prev")))
                         is None else d - 1),
            "market_cap": market_cap,
            "fcf_yield": _ratio(fcf, market_cap),
            "fcf_after_sbc_yield": (None if fcf is None or sbc is None
                                    else _ratio(fcf - sbc, market_cap)),
            "ev_ebit": _ratio(enterprise_value, ebit),
            "ev_sales": _ratio(enterprise_value, revenue),
            "debt_missing": debt is None,
            "price_date": price_day,
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Filtered:
    """The result of the filters: the companies kept, and how many each rule could not judge."""

    table: pd.DataFrame
    failed: int
    unknown: dict[str, int]


# column → (direction, parameter name in config `screen.defaults`)
FILTERS = {
    "dilution": ("max", "max_dilution"),
    "sbc_over_revenue": ("max", "max_sbc_over_revenue"),
    "fcf_margin": ("min", "min_fcf_margin"),
    "market_cap": ("min", "min_market_cap"),
}


def apply_filters(table: pd.DataFrame, limits: Mapping[str, float | None], *,
                  keep_unknown: bool = False) -> Filtered:
    """Keep the rows that pass every limit set (``None`` = limit off).

    A row a rule cannot judge (its value is missing) is neither kept nor counted as failed
    unless ``keep_unknown``: it goes to ``unknown[column]`` (if it fails no other rule), so
    the page can say how many the data, not the rules, left out.
    """
    if table.empty:
        return Filtered(table, 0, {})
    keep = pd.Series(True, index=table.index)
    failed = pd.Series(False, index=table.index)
    missing_by: dict[str, pd.Series] = {}
    for column, (direction, name) in FILTERS.items():
        limit = limits.get(name)
        if limit is None:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        missing = values.isna()
        passes = values <= float(limit) if direction == "max" else values >= float(limit)
        failed |= ~missing & ~passes
        missing_by[column] = missing
        keep &= passes | (missing & keep_unknown)
    # Unknown to a rule and failing no other: the ones the data, not the rules, left out.
    unknown = {c: int((m & ~failed).sum()) for c, m in missing_by.items()}
    return Filtered(table[keep], int(failed.sum()), unknown)


# --- Peers (§15.1.4, 2026-09-25) ------------------------------------------------------------

MIN_PEERS = 6
# (column, label, reading) — the reading says which way is "more", never which is better:
# a high SBC share can be a young company's choice, a low EV/sales a business in decline.
PEER_METRICS = [
    ("revenue_growth", "Crecimiento de ingresos"),
    ("operating_margin", "Margen operativo"),
    ("fcf_margin", "Margen FCF"),
    ("cash_conversion", "FCF / beneficio"),
    ("sbc_over_revenue", "SBC / ingresos"),
    ("dilution", "Dilución"),
    ("ev_sales", "VE / ventas"),
    ("fcf_after_sbc_yield", "Rent. FCF − SBC"),
]


@dataclass(frozen=True)
class PeerComparison:
    """A company against the screen companies sharing its SIC code.

    Attributes:
        code / digits: The SIC prefix used — 4 digits when that gives ``MIN_PEERS``, else 3,
            else 2. A broader group is still a group, but a looser one, and says so.
        peers: Tickers in the group (the company excluded).
        rows: ``[metric, label, value, median, percentile, n]`` — ``percentile`` is the share
            of peers with a lower value (``None`` when the company's value is missing).
    """

    code: str | None
    digits: int
    description: str | None
    peers: list[str]
    rows: pd.DataFrame


def peer_comparison(table: pd.DataFrame, cik: str, sics: Mapping[str, str | None],
                    descriptions: Mapping[str, str | None] | None = None) -> PeerComparison | None:
    """See :class:`PeerComparison`. ``None`` when the company is not in the screen or has no
    SIC, or no prefix gives ``MIN_PEERS`` peers."""
    if table.empty or cik not in set(table["cik"]) or not sics.get(cik):
        return None
    own = str(sics[cik])
    for digits in (4, 3, 2):
        prefix = own[:digits]
        members = [c for c in table["cik"] if c != cik and str(sics.get(c) or "")[:digits] == prefix]
        if len(members) >= MIN_PEERS:
            break
    else:
        return None
    group = table[table["cik"].isin(members)]
    me = table[table["cik"] == cik].iloc[0]
    rows = []
    for column, label in PEER_METRICS:
        values = pd.to_numeric(group[column], errors="coerce").dropna()
        mine = me[column]
        mine = None if mine is None or pd.isna(mine) else float(mine)
        rows.append({"metric": column, "label": label, "value": mine,
                     "median": float(values.median()) if len(values) else None,
                     "percentile": (float((values < mine).mean()) if mine is not None
                                    and len(values) else None),
                     "n": int(len(values))})
    return PeerComparison(code=prefix, digits=digits,
                          description=(descriptions or {}).get(cik) if digits == 4 else None,
                          peers=sorted(str(t) for t in group["ticker"].dropna()),
                          rows=pd.DataFrame(rows))
