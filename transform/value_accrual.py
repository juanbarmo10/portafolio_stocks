"""Shareholder value accrual — the governing question of the project (CLAUDE.md section 2).

The question is not "is the company doing well?" but **"how exactly does the shareholder
benefit if it does?"**. A company can grow revenue 30% a year and destroy value per share
by diluting 35% over the same period, and every metric on the income statement will look
excellent while it happens. This module makes that gap visible. It is the direct analogue
of the unlock view in the sibling crypto project: same risk, different vocabulary.

What it reports, all of it point-in-time (section 9.4):

- **Diluted share count and its year-on-year change** — the analogue of unlocks.
- **Stock-based compensation over revenue and over free cash flow.** SBC is a real cost
  paid in shares; over FCF it answers "how much of the cash the business generates is
  already owed to employees?".
- **Net buybacks** — repurchases minus issuance. A gross buyback figure flatters: it says
  nothing about whether the share count actually fell, because SBC pushes the other way.
- **Whether the buyback actually worked**, by comparing the cash spent against the change
  in share count. This is the number that catches a "shareholder-friendly" programme that
  only offsets dilution.
- **ROIC against a hurdle rate the user sets**, never one this module invents.

⚠️ **A weighted-average share count is not a flow.** It must never be summed across
quarters and its fourth quarter must never be derived by subtraction. Both operations
produce numbers that look like share counts: summing four quarters of Microsoft gives
~30 billion shares instead of ~7.5, and deriving Q4 as ``FY − Q1 − Q2 − Q3`` gives
**−14.9 billion**. The config marks it ``kind: average``, and :func:`share_count` is the
only sanctioned way to read it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from transform import fundamentals as fun

# Metrics that are a period average rather than something that accumulates. Summing or
# subtracting them is always wrong (see the module docstring).
AVERAGE_METRICS = frozenset({"diluted_shares", "basic_shares"})


@dataclass(frozen=True)
class ValueAccrual:
    """How much of the company's success reaches the shareholder, as knowable on a date.

    Attributes:
        cik: Company key.
        as_of: Date the reading was computed for.
        diluted_shares: Latest reported weighted-average diluted share count.
        dilution_yoy: Change in that count against a year earlier. **Positive means
            dilution**: more shares for the same company. Negative means the count shrank.
        sbc_ttm: Stock-based compensation over the trailing year.
        sbc_over_revenue: SBC as a share of revenue.
        sbc_over_fcf: SBC as a share of free cash flow — how much of the cash generated is
            already committed to employees.
        buybacks_ttm: Gross repurchases (cash out).
        issuance_ttm: Proceeds from issuing stock (cash in).
        net_buybacks_ttm: ``buybacks − issuance``. The gross figure alone flatters.
        buyback_per_share_removed: Net buyback cash divided by the fall in share count, or
            ``None`` when the count did not fall. When the company spent billions and the
            count did not move, that is SBC eating the buyback — and this reads ``None``
            with ``shares_removed`` at zero or negative, which is the honest way to say it.
        shares_removed: Fall in the diluted count over the year. Negative means it grew.
        effective_tax_rate: Tax expense over pre-tax income, **observed in the filings**,
            never assumed.
        nopat: Operating income after the observed tax rate.
        invested_capital: Equity plus long-term debt minus cash.
        roic: NOPAT over invested capital. An approximation, and labelled as one.
        hurdle_rate: The cost-of-capital proxy from config, or ``None`` if unset.
        roic_above_hurdle: ``roic - hurdle_rate``, or ``None`` when either is missing.
            Unknown is not "it passes".
    """

    cik: str
    as_of: str
    diluted_shares: float | None
    dilution_yoy: float | None
    sbc_ttm: float | None
    sbc_over_revenue: float | None
    sbc_over_fcf: float | None
    buybacks_ttm: float | None
    issuance_ttm: float | None
    net_buybacks_ttm: float | None
    buyback_per_share_removed: float | None
    shares_removed: float | None
    effective_tax_rate: float | None
    nopat: float | None
    invested_capital: float | None
    roic: float | None
    hurdle_rate: float | None
    roic_above_hurdle: float | None


def share_count(
    observations: pd.DataFrame, cik: str, as_of: Any, metric: str = "diluted_shares"
) -> float | None:
    """Latest weighted-average diluted share count knowable on ``as_of``.

    The annual figure is preferred, because it is the one the company itself weighted over
    the year. Falling back to the newest quarter is acceptable — it is a real reported
    average — but summing quarters or deriving a fourth is not, and this function is the
    only place that decides (see the module docstring).
    """
    annual = fun.known(observations, cik, metric, fun.ANNUAL, as_of)
    if not annual.empty:
        return float(annual["value"].iloc[-1])
    quarters = fun.known(observations, cik, metric, fun.QUARTER, as_of)
    if quarters.empty:
        return None
    return float(quarters["value"].iloc[-1])


def share_count_year_earlier(
    observations: pd.DataFrame, cik: str, as_of: Any, metric: str = "diluted_shares"
) -> float | None:
    """The same count one fiscal year back, for the year-on-year comparison.

    The annual series is consecutive, so one row back really is one year back. **The
    quarterly fallback is not**, and that is the trap: no company files a fourth-quarter
    10-Q (section 9.12), so a fiscal year arrives as *three* filed quarters and counting
    rows backwards overshoots. Measured on Microsoft's filed quarters: four rows back is
    365 days, five rows back is **455**. Comparing against a count from fifteen months ago
    raises no error and returns a perfectly plausible dilution figure, roughly a third too
    large — section 9.11 again, an ordinary operation applied to the wrong row.

    So the fallback matches by **date**: the filed quarter nearest to one year before the
    latest one, and only when it really is a year away. Otherwise ``None`` (section 12).
    """
    annual = fun.known(observations, cik, metric, fun.ANNUAL, as_of)
    if len(annual) >= 2:
        return float(annual["value"].iloc[-2])

    quarters = fun.known(observations, cik, metric, fun.QUARTER, as_of)
    if len(quarters) < 2:
        return None
    ends = pd.to_datetime(quarters["ts"])
    gaps = (ends.iloc[-1] - ends.iloc[:-1]).dt.days
    nearest = (gaps - 365).abs().idxmin()
    if not fun.YEAR_DAYS[0] <= gaps.loc[nearest] <= fun.YEAR_DAYS[1]:
        return None
    return float(quarters["value"].loc[nearest])


def dilution(observations: pd.DataFrame, cik: str, as_of: Any) -> float | None:
    """Year-on-year change in the diluted share count. **Positive means dilution.**

    The single most important number in this module. A business growing 30% a year while
    the share count grows 35% is shrinking on a per-share basis, and nothing on the income
    statement says so.
    """
    now = share_count(observations, cik, as_of)
    before = share_count_year_earlier(observations, cik, as_of)
    return fun.growth(now, before)


def effective_tax_rate(observations: pd.DataFrame, cik: str, as_of: Any) -> float | None:
    """Tax expense over pre-tax income, as filed.

    Observed, never assumed. A hardcoded "21% because that is the statutory rate" would be
    an invented number in a calculation that then looks precise (section 12), and the real
    rate of a multinational is rarely the statutory one.
    """
    tax = fun.ttm_at(observations, cik, "income_tax_expense", as_of)
    pretax = fun.ttm_at(observations, cik, "pretax_income", as_of)
    if tax is None or pretax is None or pretax <= 0:
        return None
    return tax / pretax


def invested_capital(observations: pd.DataFrame, cik: str, as_of: Any) -> float | None:
    """Equity plus long-term debt minus cash — the approximation this project uses.

    Deliberately crude and labelled as such. A precise invested capital needs operating
    lease assets, goodwill treatment decisions and working-capital adjustments that XBRL
    does not hand over uniformly; pretending otherwise would dress an estimate as a fact.
    """
    equity = fun.latest_instant(observations, cik, "equity", as_of)
    debt = fun.latest_instant(observations, cik, "long_term_debt", as_of)
    cash = fun.latest_instant(observations, cik, "cash", as_of)
    if equity is None:
        return None
    total = equity + (debt or 0.0) - (cash or 0.0)
    return total if total > 0 else None


def roic(
    observations: pd.DataFrame, cik: str, as_of: Any
) -> tuple[float | None, float | None]:
    """Approximate return on invested capital, and the NOPAT behind it.

    ``operating_income_TTM × (1 − effective tax rate) / invested capital``. Returns
    ``(roic, nopat)``, both ``None`` when any input is missing — never a partial figure
    computed with a stand-in tax rate.
    """
    operating = fun.ttm_at(observations, cik, "operating_income", as_of)
    tax_rate = effective_tax_rate(observations, cik, as_of)
    capital = invested_capital(observations, cik, as_of)
    if operating is None or tax_rate is None or capital is None:
        return None, None
    nopat = operating * (1.0 - tax_rate)
    return nopat / capital, nopat


def assess(
    observations: pd.DataFrame, cik: str, as_of: Any, hurdle_rate: float | None = None
) -> ValueAccrual:
    """The whole level-3 value-accrual reading for one company. See :class:`ValueAccrual`."""
    as_of_iso = pd.Timestamp(as_of).date().isoformat()

    revenue = fun.ttm_at(observations, cik, "revenue", as_of)
    fcf = fun.free_cash_flow(observations, cik, as_of)
    sbc = fun.ttm_at(observations, cik, "sbc", as_of)
    buybacks = fun.ttm_at(observations, cik, "buybacks", as_of)
    issuance = fun.ttm_at(observations, cik, "issuance", as_of)

    net_buybacks = None
    if buybacks is not None:
        net_buybacks = buybacks - (issuance or 0.0)

    shares_now = share_count(observations, cik, as_of)
    shares_before = share_count_year_earlier(observations, cik, as_of)
    shares_removed = (
        shares_before - shares_now
        if shares_now is not None and shares_before is not None else None
    )
    # Only meaningful when the count actually fell. If a company spent billions and the
    # count stayed flat, the honest reading is "no shares were removed", not a division
    # that would silently flip sign (section 2: the gross buyback flatters).
    per_share_removed = (
        net_buybacks / shares_removed
        if net_buybacks is not None and shares_removed is not None and shares_removed > 0
        else None
    )

    roic_value, nopat = roic(observations, cik, as_of)

    return ValueAccrual(
        cik=cik,
        as_of=as_of_iso,
        diluted_shares=shares_now,
        dilution_yoy=fun.growth(shares_now, shares_before),
        sbc_ttm=sbc,
        sbc_over_revenue=fun.ratio(sbc, revenue),
        sbc_over_fcf=fun.ratio(sbc, fcf),
        buybacks_ttm=buybacks,
        issuance_ttm=issuance,
        net_buybacks_ttm=net_buybacks,
        buyback_per_share_removed=per_share_removed,
        shares_removed=shares_removed,
        effective_tax_rate=effective_tax_rate(observations, cik, as_of),
        nopat=nopat,
        invested_capital=invested_capital(observations, cik, as_of),
        roic=roic_value,
        hurdle_rate=hurdle_rate,
        # Unknown is not "it clears the bar": without a hurdle in config this stays None
        # and the panel says "no calculable" (section 12).
        roic_above_hurdle=(
            roic_value - hurdle_rate
            if roic_value is not None and hurdle_rate is not None else None
        ),
    )
