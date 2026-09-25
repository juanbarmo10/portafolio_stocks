"""Quarter by quarter, and the balance sheet (CLAUDE.md §15.1.3).

The TTM flattens exactly what an analyst reads first: the **trend** and the **operating
leverage** — whether costs grow slower than revenue. This module lays the last quarters side
by side, as the company filed them or as derived (Q4 from the 10-K, §9.12; Q2/Q3 from
year-to-date cumulatives, §9.14), and the balance sheet on the last filing.

Rules carried over:
- Every quarter comes from ``fundamentals.quarterly``, so it is point-in-time and its
  derivations are the tested ones. Nothing is derived here that is not derived there.
- **Growth is year over year, paired by date** (a quarter against the one ~365 days before,
  THEORY §7.5), not against the previous quarter: seasonality would make every Q4 look like
  an acceleration.
- **Gross margin only from a filed ``GrossProfit``** (see the config note); a cost line the
  company does not file is ``None``, never zero.
- The share count is left out on purpose: it is an average (§9.11), its Q4 cannot be
  derived, and dilution has its own section.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from transform import fundamentals as fun

FLOWS = ("revenue", "gross_profit", "research_development", "marketing", "general_admin",
         "selling_general_admin", "operating_income", "net_income", "operating_cash_flow",
         "capex", "sbc", "interest_expense")


def _by_quarter(observations: pd.DataFrame, cik: str, metric: str, as_of: Any) -> pd.Series:
    quarters, _ = fun.quarterly(observations, cik, metric, as_of)
    if quarters.empty:
        return pd.Series(dtype=float)
    series = pd.Series(quarters["value"].astype(float).to_numpy(),
                       index=pd.to_datetime(quarters["ts"].astype(str).str[:10]))
    return series[~series.index.duplicated(keep="last")].sort_index()


def _year_before(series: pd.Series, day: pd.Timestamp) -> float | None:
    """The value ~one year before ``day`` (``fundamentals.YEAR_DAYS``), paired by date."""
    low, high = fun.YEAR_DAYS
    window = series[(series.index >= day - pd.Timedelta(days=high))
                    & (series.index <= day - pd.Timedelta(days=low))]
    return None if window.empty else float(window.iloc[-1])


def quarter_table(observations: pd.DataFrame, cik: str, as_of: Any,
                  quarters: int = 8) -> pd.DataFrame:
    """The last ``quarters`` quarters, oldest first, one row per quarter end.

    Columns: every metric in :data:`FLOWS` (``None`` where not filed), ``fcf``, and the
    ratios ``revenue_growth`` (y/y), ``gross_margin``, ``rd_share``, ``marketing_share``,
    ``operating_margin``, ``net_margin``, ``fcf_margin``, ``sbc_share``.
    """
    data = {m: _by_quarter(observations, cik, m, as_of) for m in FLOWS}
    revenue = data["revenue"]
    if revenue.empty:
        return pd.DataFrame()
    ends = revenue.index[-quarters:]
    rows = []
    for day in ends:
        row: dict[str, Any] = {"quarter_end": day.date().isoformat()}
        for metric, series in data.items():
            row[metric] = float(series[day]) if day in series.index else None
        ocf, capex = row["operating_cash_flow"], row["capex"]
        row["fcf"] = None if ocf is None or capex is None else ocf - capex
        rev = row["revenue"]
        row["revenue_growth"] = fun.growth(rev, _year_before(revenue, day))
        for name, metric in (("gross_margin", "gross_profit"),
                             ("rd_share", "research_development"),
                             ("marketing_share", "marketing"),
                             ("operating_margin", "operating_income"),
                             ("net_margin", "net_income"),
                             ("fcf_margin", "fcf"),
                             ("sbc_share", "sbc")):
            row[name] = fun.ratio(row[metric], rev)
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Balance:
    """The balance sheet on the last filing known at ``as_of``.

    Attributes:
        liquidity: Cash and equivalents plus short-term investments (either may be absent;
            both absent → ``None``).
        debt: Current plus long-term debt, convertibles included (config note on
            ``long_term_debt``).
        net_cash: ``liquidity − debt``; positive means more cash than debt.
        interest_coverage: TTM operating income over TTM interest expense. ``None`` with
            no interest filed or a loss — a negative coverage reads as nothing useful.
        noncurrent_liabilities: ``liabilities − liabilities_current``: what the page shows
            when no debt concept is filed, so a company with debt under its own tags
            (ImmunityBio) does not read as debt-free.
        cash_burn_ttm: ``−FCF`` over the last four quarters when negative, else ``None``.
        runway_years: ``liquidity / cash_burn_ttm`` — how long the cash lasts at the last
            year's burn, before raising money (which, for a company burning cash, usually
            means dilution). ``None`` when the company does not burn cash. For a
            pre-revenue biotech it is the first number to read (THEORY §3.5.1).
    """

    as_of: str
    balance_date: str | None
    cash: float | None
    short_term_investments: float | None
    liquidity: float | None
    debt_current: float | None
    long_term_debt: float | None
    debt: float | None
    net_cash: float | None
    equity: float | None
    assets: float | None
    liabilities: float | None
    interest_coverage: float | None
    cash_burn_ttm: float | None
    runway_years: float | None
    noncurrent_liabilities: float | None


def balance(observations: pd.DataFrame, cik: str, as_of: Any) -> Balance:
    """See :class:`Balance`. Every line on the latest balance sheet only
    (``fundamentals.latest_instant``)."""
    reference = fun.known(observations, cik, fun.BALANCE_REFERENCE, "", as_of)
    last_day = str(reference["ts"].iloc[-1])[:10] if not reference.empty else None

    def instant(metric: str) -> float | None:
        return fun.latest_instant(observations, cik, metric, as_of)

    cash, investments = instant("cash"), instant("short_term_investments")
    current, long_term = instant("debt_current"), instant("long_term_debt")
    liquidity = (None if cash is None and investments is None
                 else (cash or 0.0) + (investments or 0.0))
    debt = (None if current is None and long_term is None
            else (current or 0.0) + (long_term or 0.0))
    ebit = fun.ttm_at(observations, cik, "operating_income", as_of)
    interest = fun.ttm_at(observations, cik, "interest_expense", as_of)
    coverage = (ebit / interest if ebit is not None and interest and interest > 0
                and ebit > 0 else None)
    fcf = fun.free_cash_flow(observations, cik, as_of)
    liabilities, current_liabilities = instant("liabilities"), instant("liabilities_current")
    burn = -fcf if fcf is not None and fcf < 0 else None
    return Balance(
        as_of=str(pd.Timestamp(as_of).date()), balance_date=last_day, cash=cash, short_term_investments=investments,
        liquidity=liquidity, debt_current=current, long_term_debt=long_term, debt=debt,
        net_cash=None if liquidity is None or debt is None else liquidity - debt,
        equity=instant("equity"), assets=instant("assets"),
        liabilities=liabilities, interest_coverage=coverage,
        cash_burn_ttm=burn,
        runway_years=None if burn is None or liquidity is None else liquidity / burn,
        noncurrent_liabilities=(None if liabilities is None or current_liabilities is None
                                else liabilities - current_liabilities),
    )
