"""The growth screen: growth, its acceleration and its quality, across every SEC filer.

The second mode of 🔎 Cribado (CLAUDE.md §15.5, point 2). Its question is not "which
companies are good value?" but "which companies are growing, and is the growth any good?".
Like the annual screen, its output is a list to **study**, never to buy (sections 2, 12),
and nothing on it is fresher than each company's last 10-Q.

**Each company is read at its own latest reported quarter**, among the three most recent
calendar quarters (``ingest/growth_screen``), against the same quarter a year earlier. A
year-on-year comparison is the only one that survives seasonality; and a single calendar
quarter would not do, because the one holding each company's fiscal Q4 is missing
(section 9.12).

Metrics, and the traps each one avoids:

- **Revenue growth** year on year, and its **acceleration**: this quarter's growth minus the
  previous reported quarter's. Acceleration is what a growth investor pays for; a high but
  slowing growth rate is the more common and less valuable case.
- **Gross margin** and its change over the year: a business that grows by cutting prices
  shows it here before anywhere else.
- **Incremental operating margin**: extra operating income over extra revenue. Positive and
  above the current margin is operating leverage. ``None`` when revenue did not grow (a
  ratio over a non-positive base, section 12).
- **Dilution** over the **basic** share count (section 9.15: in a loss period diluted equals
  basic, so the diluted count jumps without a share being issued). A change above
  ``jump_threshold`` is an IPO, a SPAC or a merger, not dilution: ``None`` and flagged.
- **FCF margin and SBC over revenue** from the latest **annual** frame: a 3-month cash flow
  statement barely exists in ``frames`` (section 9.14). The **rule of 40** is therefore this
  quarter's growth plus the annual FCF margin — a mix of periods, said on the page.
- **Runway**: cash and short-term investments at the quarter's end over the annual cash
  burn, in years; ``None`` for a company that generates cash.
- **Market cap**: last close × basic shares of the quarter (diluted when no total basic count
  is filed, as with two share classes) × the splits after the quarter's end (section 9.1).
  **P/sales** on annual revenue. A daily dollar volume above the market cap means a share
  count filed in the wrong scale: the cap is left unknown and flagged.

Unknown is neither pass nor fail (``transform.screen.apply_filters``). Pure functions; no
network, no database (section 10).
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import pandas as pd

from transform.screen import _ratio, is_financial

QUARTER = re.compile(r"^CY(\d{4})Q([1-4])$")
YEAR = re.compile(r"^CY(\d{4})$")

# column → (direction, parameter name in `screen.growth.defaults`)
FILTERS = {
    "market_cap": ("min", "min_market_cap"),
    "revenue_q": ("min", "min_revenue_q"),
    "dollar_volume": ("min", "min_dollar_volume"),
    "revenue_growth": ("min", "min_revenue_growth"),
    "gross_margin": ("min", "min_gross_margin"),
    "dilution": ("max", "max_dilution"),
}

COLUMNS = ["cik", "ticker", "name", "quarter", "quarter_end", "stale", "revenue_q",
           "revenue_growth", "previous_growth", "acceleration", "gross_margin",
           "gross_margin_change", "operating_margin", "incremental_margin", "dilution",
           "share_jump", "revenue_annual", "fcf_annual", "fcf_margin", "sbc_over_revenue",
           "rule_of_40", "cash", "runway_years", "price", "price_date", "split_factor",
           "market_cap", "price_to_sales", "dollar_volume", "cap_suspect", "sic", "financial"]


def _year_ago(period: str) -> str:
    m = QUARTER.match(period)
    return f"CY{int(m.group(1)) - 1}Q{m.group(2)}" if m else ""


def _quarters(periods: set[str]) -> list[str]:
    """Quarter labels present, newest first."""
    found = [(int(m.group(1)), int(m.group(2)), p) for p in periods if (m := QUARTER.match(p))]
    return [p for _, _, p in sorted(found, reverse=True)]


def _wide(observations: pd.DataFrame) -> tuple[dict, dict]:
    """``{cik: {(metric, period): value}}`` and ``{cik: {(metric, period): end date}}``."""
    values: dict[str, dict[tuple[str, str], float]] = {}
    ends: dict[str, dict[tuple[str, str], str]] = {}
    if observations is None or observations.empty:
        return values, ends
    for sid, ts, value in zip(observations["series_id"].astype(str),
                              observations["ts"].astype(str), observations["value"]):
        parts = sid.split(":")
        if len(parts) != 3 or value is None or pd.isna(value):
            continue
        cik, metric, period = parts
        values.setdefault(cik, {})[(metric, period)] = float(value)
        ends.setdefault(cik, {})[(metric, period)] = ts[:10]
    return values, ends


def _prices(prices: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """``{cik: {close, day, dollar_volume, splits: {day: ratio}}}``."""
    out: dict[str, dict[str, Any]] = {}
    if prices is None or prices.empty:
        return out
    for sid, ts, value in zip(prices["series_id"].astype(str), prices["ts"].astype(str),
                              prices["value"]):
        cik, _, kind = sid.partition(":")
        row = out.setdefault(cik, {"splits": {}})
        if kind == "close":
            row["close"], row["day"] = float(value), ts[:10]
        elif kind == "dollar_volume":
            row["dollar_volume"] = float(value)
        elif kind == "split":
            row["splits"][ts[:10]] = float(value)
    return out


def _growth(now: float | None, before: float | None) -> float | None:
    ratio = _ratio(now, before)
    return None if ratio is None else ratio - 1


def growth_table(fundamentals: pd.DataFrame, prices: pd.DataFrame,
                 registry: Mapping[str, tuple[str, str]],
                 sics: Mapping[str, Any] | None = None, *,
                 jump_threshold: float = 0.5) -> pd.DataFrame:
    """One row per company with a recent reported quarter. See the module docstring.

    Args:
        fundamentals: ``sec_growth`` observations, ``{cik}:{metric}:{period}``.
        prices: ``yfinance_screen`` observations, ``{cik}:close|dollar_volume|split``.
        registry: ``{cik: (ticker, name)}``, frozen at ingest.
        sics: ``{cik: SIC code}``.
    """
    values, ends = _wide(fundamentals)
    all_periods = {p for per_cik in values.values() for (_, p) in per_cik}
    recent = _quarters(all_periods)[:3]
    newest = recent[0] if recent else None
    years = sorted((int(m.group(1)) for p in all_periods if (m := YEAR.match(p))), reverse=True)
    annual = f"CY{years[0]}" if years else None
    quotes = _prices(prices)
    rows = []
    for cik, v in values.items():
        reported = [p for p in recent if (v.get(("revenue", p)) or 0) > 0
                    and (v.get(("revenue", _year_ago(p))) or 0) > 0]
        if not reported:
            continue
        q, ago = reported[0], _year_ago(reported[0])
        revenue, revenue_ago = v[("revenue", q)], v[("revenue", ago)]
        growth = _growth(revenue, revenue_ago)
        previous = (_growth(v[("revenue", reported[1])], v[("revenue", _year_ago(reported[1]))])
                    if len(reported) > 1 else None)
        gross, gross_ago = v.get(("gross_profit", q)), v.get(("gross_profit", ago))
        gm, gm_ago = _ratio(gross, revenue), _ratio(gross_ago, revenue_ago)
        ebit, ebit_ago = v.get(("operating_income", q)), v.get(("operating_income", ago))
        incremental = (None if ebit is None or ebit_ago is None
                       else _ratio(ebit - ebit_ago, revenue - revenue_ago))
        shares, shares_ago = v.get(("basic_shares", q)), v.get(("basic_shares", ago))
        dilution = _growth(shares, shares_ago)
        jump = dilution is not None and abs(dilution) > jump_threshold
        rev_year = v.get(("revenue", annual)) if annual else None
        ocf, capex = v.get(("operating_cash_flow", annual)), v.get(("capex", annual))
        fcf = None if ocf is None or capex is None else ocf - capex
        fcf_margin = _ratio(fcf, rev_year)
        cash = v.get(("cash", f"{q}I"))
        cash_total = None if cash is None else cash + (v.get(("short_term_investments",
                                                              f"{q}I")) or 0.0)
        runway = (_ratio(cash_total, -fcf) if fcf is not None and fcf < 0
                  and cash_total is not None else None)

        quote = quotes.get(cik, {})
        quarter_end = ends[cik].get(("revenue", q))
        factor = 1.0
        for day, ratio in quote.get("splits", {}).items():
            if quarter_end and day > quarter_end and ratio > 0:
                factor *= ratio
        close = quote.get("close")
        volume = quote.get("dollar_volume")
        # Basic shares first; the diluted count when no total basic count is filed (two share
        # classes, Duolingo) or when the basic one is in the wrong scale (Amcor: 1,5 M basic
        # against 463,8 M diluted). A day's trading worth more than the whole company means a
        # count filed in the wrong scale — McDonald's tags 709,1 (millions), Iovance 450.189
        # (thousands). With no plausible count the cap is unknown and flagged, never rescaled
        # by guess (section 12). Dilution itself stays on basic shares (section 9.15).
        market_cap, cap_suspect = None, False
        if close is not None:
            for count in (shares, v.get(("diluted_shares", q))):
                if count is None:
                    continue
                cap = close * count * factor
                if volume is not None and volume > cap:
                    cap_suspect = True
                    continue
                market_cap, cap_suspect = cap, False
                break
        ticker, name = registry.get(cik, (None, None))
        sic = (sics or {}).get(cik)
        rows.append({
            "cik": cik, "ticker": ticker, "name": name, "quarter": q,
            "quarter_end": quarter_end, "stale": q != newest,
            "revenue_q": revenue, "revenue_growth": growth, "previous_growth": previous,
            "acceleration": None if growth is None or previous is None else growth - previous,
            "gross_margin": gm,
            "gross_margin_change": None if gm is None or gm_ago is None else gm - gm_ago,
            "operating_margin": _ratio(ebit, revenue, positive_base=True),
            "incremental_margin": incremental,
            "dilution": None if jump else dilution, "share_jump": jump,
            "revenue_annual": rev_year, "fcf_annual": fcf, "fcf_margin": fcf_margin,
            "sbc_over_revenue": _ratio(v.get(("sbc", annual)), rev_year),
            "rule_of_40": None if growth is None or fcf_margin is None else growth + fcf_margin,
            "cash": cash_total, "runway_years": runway,
            "price": close, "price_date": quote.get("day"), "split_factor": factor,
            "market_cap": market_cap, "price_to_sales": _ratio(market_cap, rev_year),
            "dollar_volume": volume, "cap_suspect": cap_suspect,
            "sic": sic, "financial": is_financial(sic),
        })
    return pd.DataFrame(rows, columns=COLUMNS)
