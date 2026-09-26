"""Valuation — "at what price?", the half of level 3 that was missing (CLAUDE.md §2, §15.1).

The panel could say whether a business is good and whether the shareholder captures it, and
nothing about whether the **price** already pays for all of it. This module adds that, from
data already in the database and under the same rules as the rest:

- **Point-in-time.** A multiple on a date uses the close of that date and the fundamentals
  *filed* by then (section 9.4). The same function builds today's reading and the history.
- **Raw price, split-aware share count.** The close is the raw one (section 9.1). The share
  count comes from a filing dated earlier, so any split between the two dates is applied
  to the count — a raw price after a 4:1 split times a pre-split count would quarter the
  market cap.
- **The share count is a weighted average** of the newest filed period (section 9.11), not
  the cover-page count: multi-class companies (HIMS) do not publish that one in the SEC's
  non-dimensional facts. Labelled as what it is.
- **A multiple over a non-positive base is not shown** (``None``): P/E −30 reads as cheap
  when the company loses money. Yields are shown whatever their sign, because a negative
  FCF yield *is* readable — the company burns cash.
- **EV is approximate**: market cap + long-term debt (convertibles included) − cash and
  short-term investments (the same liquidity the balance sheet shows; until 2026-09-25 it
  subtracted cash alone, 0,23 billion off for HIMS). It leaves out the current portion of
  debt, leases and minority interests, and says so.
- **A missing debt concept is not "no debt" when the liabilities say otherwise.** A company
  may file its debt under its own tags (ImmunityBio: 1,6 billion of non-current liabilities,
  no standard debt concept). When no debt is filed and non-current liabilities exceed
  ``unidentified_debt_share`` of total assets, the EV is ``None`` — incomplete — instead of
  an EV that takes the debt as zero. Below it, the liabilities are the usual leases and
  deferred items, and zero debt is the honest reading (NAUT 15 %, DUOL 4 %, IBRX 254 %).

The reverse DCF takes no discount rate of its own. The rate is the investor's (section 12),
so it answers for a grid of rates and marks the user's ``hurdle_rate`` when set.

Pure functions over stored frames; no network (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from transform import fundamentals as fun
from transform.adjustments import as_date, as_frame

MULTIPLES = ("ev_sales", "ev_ebit", "pe", "p_fcf", "fcf_yield")
UNIDENTIFIED_DEBT_SHARE = 0.25   # config: panel.valuation.unidentified_debt_share


@dataclass(frozen=True)
class Valuation:
    """One company's price against its fundamentals, as knowable on ``as_of``.

    Attributes:
        price / price_date: Last raw close on or before ``as_of``.
        shares / shares_date: Weighted-average diluted count of the newest filed period,
            adjusted for any split between ``shares_date`` and ``price_date``.
        market_cap: ``price × shares``.
        debt / cash: Latest filed long-term debt (convertibles included) and cash.
        enterprise_value: ``market_cap + debt − cash`` — approximate (module docstring).
        revenue_ttm / operating_income_ttm / net_income_ttm / fcf_ttm / sbc_ttm: TTM bases.
        ev_sales / ev_ebit / pe / p_fcf: Multiples, ``None`` over a non-positive base.
        fcf_yield: ``fcf / market_cap``. The inverse of P/FCF, defined whatever the sign.
        fcf_after_sbc_yield: ``(fcf − sbc) / market_cap``. SBC is a real cost paid in
            shares; FCF adds it back. When SBC dwarfs FCF, this is the yield that matters.
        earnings_yield: ``net_income / market_cap``.
    """

    as_of: str
    price: float | None
    price_date: str | None
    shares: float | None
    shares_date: str | None
    market_cap: float | None
    debt: float | None
    cash: float | None
    enterprise_value: float | None
    short_term_investments: float | None
    debt_unidentified: bool
    revenue_ttm: float | None
    operating_income_ttm: float | None
    net_income_ttm: float | None
    fcf_ttm: float | None
    sbc_ttm: float | None
    ev_sales: float | None
    ev_ebit: float | None
    pe: float | None
    p_fcf: float | None
    fcf_yield: float | None
    fcf_after_sbc_yield: float | None
    earnings_yield: float | None


def price_at(prices: pd.DataFrame, ticker: str, as_of: Any) -> tuple[float | None, str | None]:
    """Last raw close of ``ticker`` on or before ``as_of`` (a close is public at the close)."""
    if prices is None or prices.empty:
        return None, None
    rows = prices[prices["series_id"] == f"{ticker}:close_raw"]
    if rows.empty:
        return None, None
    day = pd.Timestamp(as_date(as_of))
    dates = pd.to_datetime(rows["ts"].str[:10])
    visible = rows[dates <= day]
    if visible.empty:
        return None, None
    last = visible.assign(_d=dates[dates <= day]).sort_values("_d").iloc[-1]
    value = last["value"]
    return (None if pd.isna(value) else float(value)), str(last["ts"])[:10]


def latest_share_count(observations: pd.DataFrame, cik: str, as_of: Any
                       ) -> tuple[float | None, str | None]:
    """Newest filed weighted-average diluted count — quarter or year, whichever ends later.

    ``value_accrual.share_count`` prefers the annual figure, which is right for a year-on-
    year dilution and stale for a price: a company diluting 9 % a year has a year-average
    count well below today's.
    """
    candidates = []
    for period in (fun.QUARTER, fun.ANNUAL):
        rows = fun.known(observations, cik, "diluted_shares", period, as_of)
        if not rows.empty:
            candidates.append((str(rows["ts"].iloc[-1])[:10], float(rows["value"].iloc[-1])))
    if not candidates:
        return None, None
    day, value = max(candidates)
    return value, day


def split_factor_between(actions: Any, ticker: str, after: Any, until: Any) -> float:
    """Product of split ratios with ``after < ex_date <= until`` — to bring a share count
    filed at ``after`` to the share units in force at ``until`` (section 9.1)."""
    frame = as_frame(actions)
    if frame.empty:
        return 1.0
    start, end = as_date(after), as_date(until)
    factor = 1.0
    rows = frame[(frame["ticker"] == ticker) & (frame["kind"] == "split")]
    for row in rows.to_dict("records"):
        day = as_date(row.get("ex_date"))
        ratio = row.get("ratio")
        if day is None or ratio is None or pd.isna(ratio) or not ratio:
            continue
        if start < day <= end:
            factor *= float(ratio)
    return factor


def _positive_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """``numerator / denominator`` only over a positive base (see the module docstring)."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _yield(flow: float | None, market_cap: float | None) -> float | None:
    if flow is None or not market_cap or market_cap <= 0:
        return None
    return flow / market_cap


def fundamentals_at(observations: pd.DataFrame, cik: str, as_of: Any) -> dict[str, Any]:
    """The filed bases of a valuation — everything except the price — as known on ``as_of``.

    Split from the price on purpose: fundamentals only change when a filing lands, prices
    change every day. :func:`history` computes these once per filing date instead of once per
    point (13,5 s → 5,5 s for five monthly years of HIMS, same result to the digit).
    """
    as_of_iso = as_date(as_of).isoformat()
    shares, shares_date = latest_share_count(observations, cik, as_of_iso)
    return {
        "shares": shares, "shares_date": shares_date,
        "debt": fun.latest_instant(observations, cik, "long_term_debt", as_of_iso),
        "cash": fun.latest_instant(observations, cik, "cash", as_of_iso),
        "short_term_investments": fun.latest_instant(observations, cik,
                                                     "short_term_investments", as_of_iso),
        "assets": fun.latest_instant(observations, cik, "assets", as_of_iso),
        "liabilities": fun.latest_instant(observations, cik, "liabilities", as_of_iso),
        "liabilities_current": fun.latest_instant(observations, cik, "liabilities_current",
                                                  as_of_iso),
        "revenue": fun.ttm_at(observations, cik, "revenue", as_of_iso),
        "ebit": fun.ttm_at(observations, cik, "operating_income", as_of_iso),
        "net_income": fun.ttm_at(observations, cik, "net_income", as_of_iso),
        "fcf": fun.free_cash_flow(observations, cik, as_of_iso),
        "sbc": fun.ttm_at(observations, cik, "sbc", as_of_iso),
    }


def debt_unidentified(bases: Mapping[str, Any],
                      threshold: float = UNIDENTIFIED_DEBT_SHARE) -> bool:
    """No debt concept filed, yet non-current liabilities above ``threshold`` of assets: the
    debt exists under the company's own tags, and zero would be a plausible wrong number."""
    if bases.get("debt") is not None:
        return False
    total, current, assets = (bases.get("liabilities"), bases.get("liabilities_current"),
                              bases.get("assets"))
    if total is None or current is None or not assets or assets <= 0:
        return False
    return (total - current) / assets > threshold


def combine(bases: Mapping[str, Any], prices: pd.DataFrame, actions: Any, ticker: str,
            as_of: Any, *, unidentified_debt_share: float = UNIDENTIFIED_DEBT_SHARE
            ) -> Valuation:
    """A valuation from filed bases and the close of ``as_of``."""
    as_of_iso = as_date(as_of).isoformat()
    price, price_date = price_at(prices, ticker, as_of_iso)
    shares, shares_date = bases["shares"], bases["shares_date"]
    if shares is not None and shares_date and price_date:
        shares *= split_factor_between(actions, ticker, shares_date, price_date)
    market_cap = price * shares if price is not None and shares is not None else None
    debt, cash = bases["debt"], bases["cash"]
    investments = bases.get("short_term_investments")
    hidden_debt = debt_unidentified(bases, unidentified_debt_share)
    # No long-term debt concept filed is read as no long-term debt — a debt-free company does
    # not tag one — unless the liabilities say otherwise (module docstring). Missing cash
    # leaves the EV unknown.
    enterprise_value = (market_cap + (debt or 0.0) - cash - (investments or 0.0)
                        if market_cap is not None and cash is not None and not hidden_debt
                        else None)
    revenue, ebit, net_income = bases["revenue"], bases["ebit"], bases["net_income"]
    fcf, sbc = bases["fcf"], bases["sbc"]
    return Valuation(
        as_of=as_of_iso, price=price, price_date=price_date,
        shares=shares, shares_date=shares_date, market_cap=market_cap,
        debt=debt, cash=cash, enterprise_value=enterprise_value,
        short_term_investments=investments, debt_unidentified=hidden_debt,
        revenue_ttm=revenue, operating_income_ttm=ebit, net_income_ttm=net_income,
        fcf_ttm=fcf, sbc_ttm=sbc,
        ev_sales=_positive_ratio(enterprise_value, revenue),
        ev_ebit=_positive_ratio(enterprise_value, ebit),
        pe=_positive_ratio(market_cap, net_income),
        p_fcf=_positive_ratio(market_cap, fcf),
        fcf_yield=_yield(fcf, market_cap),
        fcf_after_sbc_yield=_yield(fcf - sbc if fcf is not None and sbc is not None else None,
                                   market_cap),
        earnings_yield=_yield(net_income, market_cap),
    )


def assess(
    observations: pd.DataFrame, prices: pd.DataFrame, actions: Any,
    cik: str, ticker: str, as_of: Any, *,
    unidentified_debt_share: float = UNIDENTIFIED_DEBT_SHARE,
) -> Valuation:
    """The valuation reading on ``as_of``. Every missing input leaves its figures ``None``."""
    return combine(fundamentals_at(observations, cik, as_of), prices, actions, ticker, as_of,
                   unidentified_debt_share=unidentified_debt_share)


def history(
    observations: pd.DataFrame, prices: pd.DataFrame, actions: Any, cik: str, ticker: str,
    dates: Sequence[Any], *, unidentified_debt_share: float = UNIDENTIFIED_DEBT_SHARE,
) -> pd.DataFrame:
    """The multiples on each date, each one point-in-time. ``[date, *MULTIPLES]``.

    Fundamentals are computed at each filing date in the window (and at its start) and
    carried until the next one; the price is that of each date. Same answer as calling
    :func:`assess` on every date, because between two filing dates nothing filed changes.
    """
    columns = ["date", *MULTIPLES]
    if not len(dates):
        return pd.DataFrame(columns=columns)
    subset = observations[observations["series_id"].str.startswith(f"{cik}:")] \
        if not observations.empty else observations
    days = sorted(as_date(d) for d in dates)
    releases = sorted({as_date(r) for r in subset["ts_release"] if r}) if not subset.empty else []
    anchors = sorted({days[0], *[r for r in releases if days[0] < r <= days[-1]]})
    snapshots = [(anchor, fundamentals_at(subset, cik, anchor)) for anchor in anchors]
    rows, index = [], 0
    for day in days:
        while index + 1 < len(snapshots) and snapshots[index + 1][0] <= day:
            index += 1
        v = combine(snapshots[index][1], prices, actions, ticker, day,
                    unidentified_debt_share=unidentified_debt_share)
        rows.append({"date": v.as_of, **{m: getattr(v, m) for m in MULTIPLES}})
    return pd.DataFrame(rows, columns=columns)


def percentile_in_history(values: pd.Series, current: float | None) -> float | None:
    """Share of the history at or below ``current``: 0,9 = pricier than 90 % of it."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if current is None or clean.empty:
        return None
    return float((clean <= current).mean())


# --- Reverse DCF ------------------------------------------------------------------------


def band(now: Valuation, history: pd.DataFrame, *, min_points: int = 6) -> pd.DataFrame:
    """Where each multiple sits today within its own history: ``[multiple, today, low,
    median, high, percentile, points]``. Multiples with fewer than ``min_points`` months of
    history, or no value today, are left out — a band of three points is not a band.

    Read for the yields the other way round: a **high** FCF yield is a **cheap** price.
    """
    rows = []
    for name in MULTIPLES:
        today = getattr(now, name)
        values = pd.to_numeric(history.get(name), errors="coerce").dropna() \
            if history is not None and name in history else pd.Series(dtype=float)
        if today is None or len(values) < min_points:
            continue
        rows.append({"multiple": name, "today": float(today), "low": float(values.min()),
                     "median": float(values.median()), "high": float(values.max()),
                     "percentile": percentile_in_history(values, today),
                     "points": int(len(values))})
    return pd.DataFrame(rows, columns=["multiple", "today", "low", "median", "high",
                                       "percentile", "points"])


def present_value(base: float, growth: float, discount: float, terminal_growth: float,
                  years: int) -> float:
    """Value of a cash flow growing at ``growth`` for ``years``, then at
    ``terminal_growth`` forever, discounted at ``discount``. Requires discount > terminal."""
    value, flow = 0.0, base
    for year in range(1, years + 1):
        flow *= 1 + growth
        value += flow / (1 + discount) ** year
    terminal = flow * (1 + terminal_growth) / (discount - terminal_growth)
    return value + terminal / (1 + discount) ** years


@dataclass(frozen=True)
class ImpliedGrowth:
    """What the price assumes. ``growth`` is ``None`` when the question has no answer, with
    the reason in ``note`` — never a number forced out of a base that cannot carry it."""

    discount: float
    growth: float | None
    note: str


def implied_growth(
    target: float | None, base: float | None, discount: float, *,
    terminal_growth: float = 0.025, years: int = 10,
    low: float = -0.5, high: float = 1.5,
) -> ImpliedGrowth:
    """The constant yearly growth of ``base`` over ``years`` that makes its present value
    equal ``target`` (a market cap, since FCF after interest is a flow to equity)."""
    if target is None or base is None:
        return ImpliedGrowth(discount, None, "faltan datos")
    if base <= 0:
        return ImpliedGrowth(discount, None, "base negativa")
    if discount <= terminal_growth:
        return ImpliedGrowth(discount, None, "la tasa debe superar el crecimiento terminal")
    if present_value(base, high, discount, terminal_growth, years) < target:
        return ImpliedGrowth(discount, None, f"más de {high:.0%} al año")
    if present_value(base, low, discount, terminal_growth, years) > target:
        return ImpliedGrowth(discount, None, f"menos de {low:.0%} al año")
    for _ in range(200):
        mid = (low + high) / 2
        if present_value(base, mid, discount, terminal_growth, years) < target:
            low = mid
        else:
            high = mid
    return ImpliedGrowth(discount, (low + high) / 2, "")


def reverse_dcf(
    valuation: Valuation, discounts: Sequence[float], *, terminal_growth: float, years: int,
) -> dict[str, list[ImpliedGrowth]]:
    """Implied growth for each discount rate, on two bases: FCF and FCF after SBC."""
    after_sbc = (valuation.fcf_ttm - valuation.sbc_ttm
                 if valuation.fcf_ttm is not None and valuation.sbc_ttm is not None else None)
    bases: Mapping[str, float | None] = {"fcf": valuation.fcf_ttm, "fcf_after_sbc": after_sbc}
    return {
        name: [implied_growth(valuation.market_cap, base, r,
                              terminal_growth=terminal_growth, years=years) for r in discounts]
        for name, base in bases.items()
    }


# --- The other way round: the return a growth path implies (§15.4, point 7) ---------------


@dataclass(frozen=True)
class ImpliedReturn:
    """The annual return that makes today's market cap equal to the value of a free cash
    flow **per share** growing at ``growth`` for ``years`` (then ``terminal_growth``).

    The mirror of :class:`ImpliedGrowth`: that one asks what growth the price assumes at
    your rate; this one asks what return you get for the growth you believe. Growth is per
    share on purpose — dilution goes inside it, so the base is the FCF as filed (SBC added
    back) and the cost of the SBC is the dilution the investor has to subtract when choosing
    ``growth``. ``rate`` is ``None`` with the reason in ``note``.
    """

    growth: float
    rate: float | None
    note: str


def implied_return(market_cap: float | None, base: float | None, growth: float, *,
                   terminal_growth: float = 0.025, years: int = 10,
                   upper: float = 1.0) -> ImpliedReturn:
    """Bisection on the rate in ``(terminal_growth, upper]`` (see :class:`ImpliedReturn`)."""
    if market_cap is None or market_cap <= 0:
        return ImpliedReturn(growth, None, "sin capitalización")
    if base is None:
        return ImpliedReturn(growth, None, "sin FCF")
    if base <= 0:
        return ImpliedReturn(growth, None, "FCF negativo: no hay flujo que crezca")
    low, high = terminal_growth + 1e-6, upper
    if present_value(base, growth, high, terminal_growth, years) > market_cap:
        return ImpliedReturn(growth, None, f"más de {upper:.0%}".replace(".", ","))
    if present_value(base, growth, low, terminal_growth, years) < market_cap:
        return ImpliedReturn(growth, None, "menos que el crecimiento terminal")
    for _ in range(200):
        mid = (low + high) / 2
        if present_value(base, growth, mid, terminal_growth, years) > market_cap:
            low = mid
        else:
            high = mid
    return ImpliedReturn(growth, (low + high) / 2, "")


def return_grid(valuation: Valuation, growths: Sequence[float], *,
                terminal_growth: float = 0.025, years: int = 10) -> list[ImpliedReturn]:
    """:func:`implied_return` for each growth scenario, on the valuation's FCF."""
    return [implied_return(valuation.market_cap, valuation.fcf_ttm, g,
                           terminal_growth=terminal_growth, years=years) for g in growths]
