"""Growth of the business against growth per share — the gap section 2 is about.

"A company can grow revenue 30 % a year and destroy value per share if it dilutes 35 %."
The panel showed the dilution of one year; it did not put the two growth rates side by
side over the horizons a thesis lives on. This module does, for revenue, free cash flow and
net income, over 3 and 5 years.

**Which share count: the basic one** (section 9.15). In a loss period US GAAP sets diluted
shares equal to basic — counting options and convertibles would shrink the loss per share —
so the diluted series jumps every time a company crosses from profit to loss: HIMS 258 M →
228 M in 2026 without retiring a share. Growth over years needs a series that measures the
same thing at both ends, and that is the basic weighted average (shares actually
outstanding). The diluted count is the fallback when a company files no basic one; one
series per company, never a mix.

**Per share, without breaking section 9.11.** Each twelve-month (TTM) value is divided by
the share count *of the period ending on the same date*: the fiscal year's average
at a fiscal year-end (the same twelve months as the TTM — the exact match), otherwise the
average of the quarter ending there. Share counts are averages: they are read, never summed
or subtracted, and a date with no filed count has no per-share point rather than a guessed
one.

**Growth is compound** (section 9.10): ``cagr`` over the actual years between the two
points, which are paired **by date** (the point nearest to N years before the latest, within
``PAIR_TOLERANCE_DAYS``), never by counting rows. A starting value that is not positive
leaves the rate undefined — ``None``, said on the page.

**A jump in the share count is not dilution.** Across an IPO or a SPAC merger the diluted
count leaps (HIMS +333 % in 2021-Q1, Duolingo +124 %, Nautilus +125 %): before the listing,
preferred shares that later convert are not in it. A horizon whose window contains a
quarter-to-quarter change above ``jump_threshold`` (0,5) gets no per-share growth and says
why; the total growth still stands. A real issuance below it (ImmunityBio +32 % in 2024)
counts, because it is dilution.

Point-in-time: every series is what had been filed by ``as_of``.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from transform import fundamentals as fun

PAIR_TOLERANCE_DAYS = 45
JUMP_THRESHOLD = 0.5   # config: panel.per_share.jump_threshold
METRICS = ("revenue", "fcf", "net_income")


def _dated(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    out = pd.Series(frame["value"].astype(float).to_numpy(),
                    index=pd.to_datetime(frame["ts"].astype(str).str[:10]))
    return out[~out.index.duplicated(keep="last")].sort_index()


def ttm(observations: pd.DataFrame, cik: str, metric: str, as_of: Any) -> pd.Series:
    """The TTM series of a metric; ``fcf`` is operating cash flow minus capex, point by point
    (both legs over the same four quarters, as :func:`fundamentals.free_cash_flow`)."""
    if metric == "fcf":
        ocf = _dated(fun.ttm_series(observations, cik, "operating_cash_flow", as_of))
        capex = _dated(fun.ttm_series(observations, cik, "capex", as_of))
        both = pd.concat({"ocf": ocf, "capex": capex}, axis=1, join="inner").dropna()
        return (both["ocf"] - both["capex"]).sort_index()
    return _dated(fun.ttm_series(observations, cik, metric, as_of))


def shares_for(observations: pd.DataFrame, cik: str, as_of: Any,
               metric: str = "basic_shares") -> pd.Series:
    """One share-count series per period end: the fiscal year's average where there is one,
    the quarter's average elsewhere (module docstring)."""
    quarters = _dated(fun.known(observations, cik, metric, fun.QUARTER, as_of))
    annual = _dated(fun.known(observations, cik, metric, fun.ANNUAL, as_of))
    return pd.concat([quarters[~quarters.index.isin(annual.index)], annual]).sort_index()


def share_basis(observations: pd.DataFrame, cik: str, as_of: Any) -> tuple[str, pd.Series]:
    """``("basic_shares" | "diluted_shares", series)``: basic when it reaches the latest
    filed period, else diluted. A basic series that stopped (Duolingo files it by share
    class since 2023, outside the non-dimensional facts) would compare today with 2023."""
    basic = shares_for(observations, cik, as_of, "basic_shares")
    diluted = shares_for(observations, cik, as_of, "diluted_shares")
    if len(basic) and (not len(diluted) or basic.index[-1] >= diluted.index[-1]
                       - pd.Timedelta(days=PAIR_TOLERANCE_DAYS)):
        return "basic_shares", basic
    return "diluted_shares", diluted


def per_share(values: pd.Series, shares: pd.Series, tolerance_days: int = 5) -> pd.Series:
    """``values / shares`` where a count exists for the same period end (± a few days, for
    52/53-week fiscal years). A point without its count is dropped, never guessed."""
    if values.empty or shares.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    out = {}
    for day, value in values.items():
        near = shares[(shares.index >= day - pd.Timedelta(days=tolerance_days))
                      & (shares.index <= day + pd.Timedelta(days=tolerance_days))]
        if len(near) and near.iloc[-1] > 0:
            out[day] = value / float(near.iloc[-1])
    return pd.Series(out, dtype=float).sort_index()


def loss_periods(observations: pd.DataFrame, cik: str, as_of: Any,
                 dates: pd.DatetimeIndex) -> pd.Series:
    """For each share-count date: whether that period's net income was a loss (the fiscal
    year's at a year-end, the quarter's otherwise — the same period as the count). ``NaN``
    where the income is not filed."""
    quarters = _dated(fun.known(observations, cik, "net_income", fun.QUARTER, as_of))
    annual = _dated(fun.known(observations, cik, "net_income", fun.ANNUAL, as_of))
    income = pd.concat([quarters[~quarters.index.isin(annual.index)], annual]).sort_index()
    out = {}
    for day in dates:
        near = income[abs((income.index - day).days) <= 5]
        out[day] = float(near.iloc[-1] < 0) if len(near) else float("nan")
    return pd.Series(out, dtype=float)


def start_of(series: pd.Series, years: int) -> pd.Timestamp | None:
    """The point nearest to ``years`` before the latest, if within the tolerance."""
    if series is None or len(series) < 2:
        return None
    target = series.index[-1] - pd.DateOffset(years=years)
    gaps = abs((series.index - target).days)
    return None if gaps.min() > PAIR_TOLERANCE_DAYS else series.index[gaps.argmin()]


def growth(series: pd.Series, years: int) -> float | None:
    """CAGR from the point nearest to ``years`` before the latest (paired by date) to the
    latest; ``None`` without such a point or over a non-positive start."""
    start_day = start_of(series, years)
    if start_day is None:
        return None
    end_day = series.index[-1]
    span = (end_day - start_day).days / 365.25
    return fun.cagr(float(series[start_day]), float(series.iloc[-1]), span)


def has_jump(counts: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
             threshold: float = JUMP_THRESHOLD) -> bool:
    """Whether the share count changes by more than ``threshold`` between two consecutive
    filed periods inside ``(start, end]`` — an IPO, a SPAC or a merger, not dilution."""
    window = counts[(counts.index >= start - pd.Timedelta(days=PAIR_TOLERANCE_DAYS))
                    & (counts.index <= end)]
    return bool((window.pct_change().abs() > threshold).any()) if len(window) > 1 else False


@dataclass(frozen=True)
class PerShareGrowth:
    """``jumps``: per horizon, whether its window crosses a share-count jump; ``flips``:
    whether, on the diluted basis, its two ends sit on different sides of zero income (both
    leave the per-share columns and the share growth ``None``). ``basis``: which count.
    ``table``: ``[metric, years, total, per_share, gap]`` — ``gap`` = total − per share,
    the growth that went to new shareholders instead of the existing ones.
    ``shares``: CAGR of the share count per horizon. ``chart``: revenue TTM and revenue per
    share, both as indices (first common point = 100)."""

    table: pd.DataFrame
    shares: dict[int, float | None]
    chart: pd.DataFrame
    jumps: dict[int, bool]
    basis: str
    flips: dict[int, bool]
    shares_yoy: float | None


def assess(observations: pd.DataFrame, cik: str, as_of: Any,
           horizons: tuple[int, ...] = (3, 5), *,
           jump_threshold: float = JUMP_THRESHOLD) -> PerShareGrowth:
    """See :class:`PerShareGrowth`."""
    basis, counts = share_basis(observations, cik, as_of)
    end = counts.index[-1] if len(counts) else pd.Timestamp(as_of)
    jumps = {y: has_jump(counts, end - pd.DateOffset(years=y), end, jump_threshold)
             for y in horizons}
    # With the diluted count (no basic one filed), two ends on different sides of zero
    # income measure different things: a loss period counts no options (ASC 260).
    flips = {y: False for y in horizons}
    losses = pd.Series(dtype=float)
    if basis == "diluted_shares" and len(counts):
        losses = loss_periods(observations, cik, as_of, counts.index)
        for y in horizons:
            first = start_of(counts, y)
            if first is not None and not pd.isna(losses.get(first)) \
                    and not pd.isna(losses.iloc[-1]) and losses.get(first) != losses.iloc[-1]:
                flips[y] = True
    blocked = {y: jumps[y] or flips[y] for y in horizons}
    rows = []
    series_by_metric = {}
    for metric in METRICS:
        total = ttm(observations, cik, metric, as_of)
        each = per_share(total, counts)
        series_by_metric[metric] = (total, each)
        for years in horizons:
            t = growth(total, years)
            e = None if blocked[years] else growth(each, years)
            rows.append({"metric": metric, "years": years, "total": t, "per_share": e,
                         "gap": None if t is None or e is None else t - e})
    revenue, revenue_ps = series_by_metric["revenue"]
    both = pd.concat({"total": revenue, "per_share": revenue_ps}, axis=1, join="inner").dropna()
    # The chart covers the longest horizon only, rebased at its first point: from the very
    # first filing, a company that sold almost nothing makes any index absurd.
    if len(both):
        both = both[both.index >= both.index[-1] - pd.DateOffset(years=max(horizons))]
        both = both[(both > 0).all(axis=1)]
        # ...and after the last share-count jump, which would bend the per-share line for a
        # reason that is not dilution (an IPO's preferred shares converting).
        changes = counts.pct_change().abs()
        last_jump = changes[changes > jump_threshold].index.max() if len(changes) else None
        if last_jump is not None and not pd.isna(last_jump):
            both = both[both.index >= last_jump]
        if len(losses.dropna()):
            sign_change = losses.dropna().diff().abs()
            last_flip = sign_change[sign_change > 0].index.max()
            if not pd.isna(last_flip):
                both = both[both.index >= last_flip]
    chart = (both / both.iloc[0] * 100).rename_axis("date").reset_index() if len(both) \
        else pd.DataFrame(columns=["date", "total", "per_share"])
    return PerShareGrowth(
        table=pd.DataFrame(rows, columns=["metric", "years", "total", "per_share", "gap"]),
        shares={y: None if blocked[y] else growth(counts, y) for y in horizons},
        shares_yoy=None if has_jump(counts, end - pd.DateOffset(years=1), end, jump_threshold)
        else growth(counts, 1),
        chart=chart, jumps=jumps, basis=basis, flips=flips,
    )
