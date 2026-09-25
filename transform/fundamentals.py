"""Level 3 of the checklist — "is the thesis still alive?" (CLAUDE.md sections 2, 8 phase 2).

Pure transformations over the XBRL facts written by ``ingest/sec_xbrl.py``: no network, no
database handle (section 10). Every reading is computed from what was **publicly filed** by
the simulated date and never from what was later revised (section 9.4).

Two domain conventions are implemented here rather than assumed anywhere else, because
both produce answers that look perfectly reasonable when they are wrong:

**No company files a fourth-quarter 10-Q.** Q4 exists only inside the 10-K, which reports
the full year. A fiscal year therefore arrives as *three* reported quarters plus an annual
figure, and Q4 has to be derived: ``FY − Q1 − Q2 − Q3``. A panel that plots quarterly
revenue straight from XBRL silently drops every fourth bar — the series still looks like a
series. The derivation lives in :func:`derive_fourth_quarter`, once, and it inherits the
**10-K's filing date**, because that is the day the number first became knowable.

**Growth is geometric, never an average of percentages** (section 9.10). The arithmetic
mean of period-over-period changes always overstates compounding, and the bias grows with
volatility — so it systematically rewards erratic companies over steady ones, which is the
opposite of what a ranking should do. :func:`cagr` is the only multi-period growth function
here, and it returns ``None`` when the starting value is not positive rather than inventing
a number that "comes out".

A metric that cannot be computed is ``None`` and visible. There is no partial TTM, no
extrapolated quarter, no filled gap (section 12).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

# The point-in-time rule is shared discipline, not a macro concept; it lives in
# transform/macro.py because that is where it was first needed (section 9.4).
from transform.macro import point_in_time

QUARTER = "q"
ANNUAL = "fy"
HALF = "ytd2"           # cumulative, two quarters into the fiscal year
THREE_QUARTERS = "ytd3"  # cumulative, three quarters in

# Metrics that are a period AVERAGE, not something that accumulates. They must never be
# summed or differenced across periods: a nine-month average share count minus a six-month
# one is not a quarter's share count, it is nothing at all. Defined here rather than in
# value_accrual because this is the layer that does the arithmetic and therefore the layer
# that has to refuse (section 9.11).
AVERAGE_METRICS = frozenset({"diluted_shares"})

# A fiscal quarter is ~91 days. Anything outside this is a gap in the data, a fiscal-year
# change, or a company that reports on a different calendar — never something to smooth over.
QUARTER_GAP_DAYS = (80, 100)
YEAR_DAYS = (350, 380)


@dataclass(frozen=True)
class Snapshot:
    """Level-3 reading for one company, as knowable on a date.

    Attributes:
        cik: Company key (section 9.3).
        as_of: Date the reading was computed for.
        revenue_ttm: Trailing-twelve-month revenue, or ``None`` without four quarters.
        net_income_ttm: TTM net income.
        operating_income_ttm: TTM operating income.
        fcf_ttm: TTM free cash flow, operating cash flow minus capital expenditure.
        operating_margin: Operating income over revenue.
        net_margin: Net income over revenue.
        fcf_margin: Free cash flow over revenue.
        cash_conversion: FCF over net income — the check on accounting profit that never
            reaches the bank account (section 2).
        revenue_growth_yoy: TTM revenue against the TTM one year earlier.
        assets / liabilities / equity / cash / long_term_debt: Latest filed balance sheet.
        quarters_available: How many quarters the TTM figures rest on. Below four, the
            TTM values are ``None`` — a three-quarter "year" is not a year.
        stale_days: Days between the newest fact used and ``as_of``. A quarterly filer is
            legitimately stale for weeks; the panel shows it so nobody reads a January
            balance sheet as today's.
    """

    cik: str
    as_of: str
    revenue_ttm: float | None
    net_income_ttm: float | None
    operating_income_ttm: float | None
    fcf_ttm: float | None
    operating_margin: float | None
    net_margin: float | None
    fcf_margin: float | None
    cash_conversion: float | None
    revenue_growth_yoy: float | None
    assets: float | None
    liabilities: float | None
    equity: float | None
    cash: float | None
    long_term_debt: float | None
    quarters_available: int
    stale_days: int | None


# --- Reading the long-format frame --------------------------------------------


def series(
    observations: pd.DataFrame, cik: str, metric: str, period: str = ANNUAL
) -> pd.DataFrame:
    """Every stored version of one metric: ``[ts, ts_release, value]``, oldest first.

    Args:
        observations: Long-format frame as stored.
        cik: Ten-digit CIK.
        metric: Canonical metric name, e.g. ``'revenue'``.
        period: ``'q'``, ``'fy'``, or ``''`` for a balance-sheet instant.
    """
    columns = ["ts", "ts_release", "value"]
    if observations is None or observations.empty:
        return pd.DataFrame(columns=columns)
    series_id = f"{cik}:{metric}" + (f":{period}" if period else "")
    rows = observations[observations["series_id"] == series_id]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    return rows.sort_values(["ts", "ts_release"])[columns].reset_index(drop=True)


def known(
    observations: pd.DataFrame, cik: str, metric: str, period: str, as_of: Any
) -> pd.DataFrame:
    """One metric as it was knowable on ``as_of`` — one row per period (section 9.4).

    Restatements are resolved the same way everywhere: the version in force on that date
    wins, which is why the database keeps them all (section 9.6).
    """
    columns = ["ts", "ts_release", "value"]
    if observations is None or observations.empty:
        return pd.DataFrame(columns=columns)
    series_id = f"{cik}:{metric}" + (f":{period}" if period else "")
    rows = observations[observations["series_id"] == series_id]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    visible = point_in_time(rows, as_of)
    if visible.empty:
        return pd.DataFrame(columns=columns)
    return visible.sort_values("ts")[columns].reset_index(drop=True)


BALANCE_REFERENCE = "assets"


def latest_instant(
    observations: pd.DataFrame, cik: str, metric: str, as_of: Any
) -> float | None:
    """The balance-sheet value on the **latest balance sheet** filed by ``as_of``.

    The latest balance sheet is the newest date ``assets`` was filed for — every balance
    sheet has total assets. A line whose last value is older was not on it: HIMS last
    tagged ``DebtCurrent`` in 2020, and "the most recent value" would have put a
    five-year-old figure next to today's, in the EV and the invested capital, without an
    error. Such a line is ``None``, which every caller already reads as "not there".
    Found 2026-09-25 (RESEARCH.md §2.39). Without any ``assets``, the metric's own latest.
    """
    rows = known(observations, cik, metric, "", as_of)
    if rows.empty:
        return None
    if metric != BALANCE_REFERENCE:
        reference = known(observations, cik, BALANCE_REFERENCE, "", as_of)
        if not reference.empty and \
                str(rows["ts"].iloc[-1])[:10] < str(reference["ts"].iloc[-1])[:10]:
            return None
    return float(rows["value"].iloc[-1])


# --- The fourth quarter nobody files ------------------------------------------


def derive_fourth_quarter(
    quarters: pd.DataFrame, annual: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Complete each fiscal year with the quarter that was never filed on its own.

    ``FY − Q1 − Q2 − Q3``, and only when **exactly three** quarters fall inside the year.
    Fewer or more means a gap, a fiscal-year change, or a restated period, and any of those
    would turn a subtraction into an invention — so the year is skipped and reported.

    The derived row inherits the **annual figure's** publication date: Q4 becomes knowable
    the day the 10-K lands, not the day the quarter ended. Using the quarter-end date would
    hand a backtest two months of the future, which is the whole trap of section 9.4.

    Returns:
        ``(quarters_including_q4, skipped)`` — the reasons, one per fiscal year skipped.
    """
    if annual is None or annual.empty:
        return quarters.copy(), []

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    existing = set(quarters["ts"]) if not quarters.empty else set()

    for year in annual.itertuples():
        year_end = pd.Timestamp(year.ts)
        window_start = year_end - pd.Timedelta(days=YEAR_DAYS[1])
        if year.ts in existing:
            continue  # the fourth quarter was filed after all; nothing to derive
        inside = quarters[
            (pd.to_datetime(quarters["ts"]) > window_start)
            & (pd.to_datetime(quarters["ts"]) < year_end)
        ] if not quarters.empty else pd.DataFrame(columns=quarters.columns)

        if len(inside) != 3:
            skipped.append(
                f"{year.ts}: {len(inside)} quarter(s) inside the fiscal year, expected 3 "
                "(gap, restatement or a changed fiscal year end)"
            )
            continue
        rows.append({
            "ts": year.ts,
            # Knowable only when the 10-K was filed — and never earlier than the quarters
            # it is derived from.
            "ts_release": max([year.ts_release, *inside["ts_release"]]),
            "value": float(year.value) - float(inside["value"].sum()),
        })

    if not rows:
        return quarters.copy(), skipped
    combined = pd.concat([quarters, pd.DataFrame(rows)], ignore_index=True)
    return combined.sort_values("ts").reset_index(drop=True), skipped


def derive_from_cumulatives(
    quarters: pd.DataFrame,
    half: pd.DataFrame,
    three_quarters: pd.DataFrame,
    *,
    metric: str = "",
) -> tuple[pd.DataFrame, list[str]]:
    """Recover the quarters a company reports only inside year-to-date cumulatives.

    Many companies file the cash-flow statement as **Q1 plus cumulatives**: a three-month
    figure for the first quarter, then six- and nine-month totals. Measured on HIMS, whose
    only three-month operating cash flow in each fiscal year is Q1 — so the quarterly
    series has one point a year, never four contiguous, and the TTM (and therefore free
    cash flow, and therefore cash conversion) is simply unavailable.

    The arithmetic is the same family as :func:`derive_fourth_quarter`::

        Q2 = YTD2 − Q1          Q3 = YTD3 − YTD2

    and so is the publication rule, which is the part that matters: a derived quarter
    inherits the **latest** filing date of its inputs, because it only became knowable when
    the document carrying the cumulative landed. Dating it to the quarter end would hand a
    backtest weeks of the future, invisibly, since the number itself is right (section 9.4).

    A quarter the company *did* file on its own is never overwritten: reported beats
    derived, always.

    ⚠️ Refuses outright for an average metric (section 9.11). Differencing a nine-month
    weighted-average share count against a six-month one produces a float and means
    nothing.

    Returns:
        ``(quarters_including_derived, skipped)`` — one reason per cumulative that could
        not be resolved, never a silent gap.
    """
    columns = ["ts", "ts_release", "value"]
    if metric in AVERAGE_METRICS:
        return quarters.copy(), [
            f"{metric}: is a period average, so quarters are never derived from "
            "cumulatives by subtraction (section 9.11)"
        ]

    base = quarters.copy() if quarters is not None else pd.DataFrame(columns=columns)
    skipped: list[str] = []
    # Q2 comes from the two-quarter cumulative minus the quarter before it; Q3 from the
    # three-quarter cumulative minus the two-quarter one. Both subtract the cumulative that
    # ends exactly one quarter earlier, so the two passes share their logic.
    passes = (
        (half, base, "Q2", "the preceding quarter"),
        (three_quarters, half, "Q3", "the two-quarter cumulative"),
    )

    derived: list[dict[str, Any]] = []
    for cumulative, previous, label, described in passes:
        if cumulative is None or cumulative.empty:
            continue
        for row in cumulative.itertuples():
            if not base.empty and row.ts in set(base["ts"]):
                continue  # the company filed that quarter on its own; reported wins
            earlier = _one_quarter_before(previous, row.ts)
            if earlier is None:
                skipped.append(
                    f"{row.ts}: cannot derive {label}, {described} is missing"
                )
                continue
            derived.append({
                "ts": row.ts,
                # Knowable only with the later of the two documents (section 9.4).
                "ts_release": max(str(row.ts_release), str(earlier["ts_release"])),
                "value": float(row.value) - float(earlier["value"]),
            })

    if not derived:
        return base, skipped
    combined = pd.concat([base, pd.DataFrame(derived, columns=columns)], ignore_index=True)
    return combined.sort_values("ts").reset_index(drop=True), skipped


def _one_quarter_before(frame: pd.DataFrame, ts: Any) -> dict[str, Any] | None:
    """The row of ``frame`` ending roughly one quarter before ``ts``, or ``None``.

    Matched by date rather than by position: counting rows assumes a cadence this data does
    not have, which is precisely the mistake section 9.11 records.
    """
    if frame is None or frame.empty:
        return None
    target = pd.Timestamp(ts)
    ends = pd.to_datetime(frame["ts"])
    gaps = (target - ends).dt.days
    candidates = gaps[(gaps >= QUARTER_GAP_DAYS[0]) & (gaps <= QUARTER_GAP_DAYS[1])]
    if candidates.empty:
        return None
    return frame.loc[candidates.index[0]].to_dict()


def quarterly(
    observations: pd.DataFrame, cik: str, metric: str, as_of: Any
) -> tuple[pd.DataFrame, list[str]]:
    """The complete quarterly series as knowable on ``as_of``, every quarter included.

    Two derivations, in the only order that works. First the quarters hidden inside
    year-to-date cumulatives (Q2 and Q3 for a company that files the cash-flow statement as
    Q1 plus totals), then the fourth quarter, which needs the other three to exist before it
    can be computed. Each is reported when it cannot be done, never silently skipped.
    """
    quarters = known(observations, cik, metric, QUARTER, as_of)
    annual = known(observations, cik, metric, ANNUAL, as_of)
    half = known(observations, cik, metric, HALF, as_of)
    three_quarters = known(observations, cik, metric, THREE_QUARTERS, as_of)

    quarters, from_cumulatives = derive_from_cumulatives(
        quarters, half, three_quarters, metric=metric
    )
    completed, skipped = derive_fourth_quarter(quarters, annual)
    skipped = [*from_cumulatives, *skipped]
    # A derived Q4 may be dated before as_of but only became knowable with the 10-K, so
    # the publication filter is applied again after deriving.
    if not completed.empty:
        cutoff = pd.Timestamp(as_of).date().isoformat()
        completed = completed[completed["ts_release"].str[:10] <= cutoff]
    return completed.reset_index(drop=True), skipped


# --- Trailing twelve months ---------------------------------------------------


def ttm(quarters: pd.DataFrame) -> float | None:
    """Sum of the last four **contiguous** quarters, or ``None``.

    Contiguity is checked, not assumed: four quarters with a hole in the middle add up to a
    number that is not a year. Rather than return a partial figure dressed as a yearly one,
    the function returns ``None`` and the panel shows a hole (section 12). The usual causes
    are a company that changed its fiscal year end and a filing this project has not
    ingested yet — both real, neither worth papering over.
    """
    if quarters is None or len(quarters) < 4:
        return None
    last = quarters.sort_values("ts").tail(4)
    ends = pd.to_datetime(last["ts"])
    gaps = ends.diff().dropna().dt.days
    if not gaps.between(QUARTER_GAP_DAYS[0], QUARTER_GAP_DAYS[1]).all():
        return None
    return float(last["value"].sum())


def ttm_series(
    observations: pd.DataFrame, cik: str, metric: str, as_of: Any
) -> pd.DataFrame:
    """Rolling trailing-twelve-month series: ``[ts, value]``, one point per quarter end.

    Each point sums the four quarters ending there, and only when those four are
    contiguous — the same rule :func:`ttm` applies to the latest window, applied along the
    whole history, so a fiscal-year change leaves a visible gap instead of a quiet step.

    Note what this is and is not: it is *today's* view of history, filtered by what was
    published on ``as_of``. It is not a series of what each date's TTM looked like *at that
    date*, which would need one point-in-time pass per point; for a chart of how a business
    has evolved, the former is the right object and the latter would mix restatements into
    the shape of the line.
    """
    quarters, _skipped = quarterly(observations, cik, metric, as_of)
    if len(quarters) < 4:
        return pd.DataFrame(columns=["ts", "value"])

    ordered = quarters.sort_values("ts").reset_index(drop=True)
    points: list[dict[str, Any]] = []
    for end in range(3, len(ordered)):
        window = ordered.iloc[end - 3:end + 1]
        value = ttm(window)
        if value is not None:
            points.append({"ts": window["ts"].iloc[-1], "value": value})
    return pd.DataFrame(points, columns=["ts", "value"])


def ttm_at(observations: pd.DataFrame, cik: str, metric: str, as_of: Any) -> float | None:
    """Trailing twelve months of one metric, as knowable on ``as_of``."""
    quarters, _skipped = quarterly(observations, cik, metric, as_of)
    return ttm(quarters)


# --- Growth, the geometric way (section 9.10) ---------------------------------


def cagr(first: float | None, last: float | None, years: float) -> float | None:
    """Compound annual growth rate between two values.

    The only multi-period growth function in this module, deliberately. The arithmetic
    mean of period-over-period changes always overstates compounding — ``+100%`` then
    ``−50%`` averages to ``+25%`` a year on a value that did not move — and the bias grows
    with volatility, so a ranking built on it orders companies by erraticism disguised as
    growth (section 9.10).

    Returns ``None`` when the starting value is not positive: the rate is genuinely
    undefined there, and substituting an arithmetic mean "because it produces a number" is
    exactly what section 12 forbids.
    """
    if first is None or last is None or years <= 0:
        return None
    if first <= 0:
        return None
    return (float(last) / float(first)) ** (1.0 / years) - 1.0


def growth(current: float | None, previous: float | None) -> float | None:
    """Simple period-over-period change, or ``None`` when the base is not positive.

    A growth rate over a negative or zero base is not a small number, it is a meaningless
    one — and for loss-making companies it is the single easiest way to publish nonsense.
    """
    if current is None or previous is None or previous <= 0:
        return None
    return float(current) / float(previous) - 1.0


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Margin-style ratio, ``None`` when the denominator is missing or zero."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


# --- Derived readings ---------------------------------------------------------


def cash_conversion(fcf: float | None, net_income: float | None) -> float | None:
    """FCF over net income, or ``None`` when the profit base is not positive.

    The guard is the whole function. "Cash conversion of 0.50" means half the accounting
    profit reached the bank account; the sentence has no meaning when there was no profit.
    Computed anyway, a loss-making company with **positive** free cash flow reports a
    *negative* conversion — which reads as the worst possible result when the underlying
    fact is the opposite and notable.

    Measured on HIMS (2026-09-23): free cash flow +61.65M against net income −142.03M gives
    −0.43. The honest reading is not a ratio at all; it is that the company generated cash
    while reporting a loss, and the panel says that in words instead (section 12, and the
    same principle as :func:`growth` refusing a non-positive base).
    """
    if fcf is None or net_income is None or net_income <= 0:
        return None
    return float(fcf) / float(net_income)


def free_cash_flow(observations: pd.DataFrame, cik: str, as_of: Any) -> float | None:
    """TTM operating cash flow minus TTM capital expenditure.

    Capex is filed as a positive outflow (``PaymentsToAcquirePropertyPlantAndEquipment``),
    so it is subtracted. Both legs must span the same four quarters or the result is
    ``None``: mixing a full-year cash flow with three quarters of capex would produce a
    flattering figure that is not free cash flow at all.
    """
    operating = ttm_at(observations, cik, "operating_cash_flow", as_of)
    capex = ttm_at(observations, cik, "capex", as_of)
    if operating is None or capex is None:
        return None
    return operating - capex


def snapshot(observations: pd.DataFrame, cik: str, as_of: Any) -> Snapshot:
    """Everything level 3 reports for one company on one date. See :class:`Snapshot`."""
    as_of_iso = pd.Timestamp(as_of).date().isoformat()

    revenue_quarters, _ = quarterly(observations, cik, "revenue", as_of)
    revenue = ttm(revenue_quarters)
    net_income = ttm_at(observations, cik, "net_income", as_of)
    operating_income = ttm_at(observations, cik, "operating_income", as_of)
    fcf = free_cash_flow(observations, cik, as_of)

    # Year-on-year on a TTM basis: the same four-quarter window, one year earlier, and
    # computed with what was public *then* as well.
    previous = None
    if len(revenue_quarters) >= 8:
        earlier = revenue_quarters.iloc[:-4]
        previous = ttm(earlier)

    facts_used = [
        frame["ts_release"].max()
        for frame in (revenue_quarters,)
        if frame is not None and not frame.empty
    ]
    stale_days = None
    if facts_used:
        newest = max(facts_used)
        stale_days = (pd.Timestamp(as_of_iso) - pd.Timestamp(newest)).days

    return Snapshot(
        cik=cik,
        as_of=as_of_iso,
        revenue_ttm=revenue,
        net_income_ttm=net_income,
        operating_income_ttm=operating_income,
        fcf_ttm=fcf,
        operating_margin=ratio(operating_income, revenue),
        net_margin=ratio(net_income, revenue),
        fcf_margin=ratio(fcf, revenue),
        cash_conversion=cash_conversion(fcf, net_income),
        revenue_growth_yoy=growth(revenue, previous),
        assets=latest_instant(observations, cik, "assets", as_of),
        liabilities=latest_instant(observations, cik, "liabilities", as_of),
        equity=latest_instant(observations, cik, "equity", as_of),
        cash=latest_instant(observations, cik, "cash", as_of),
        long_term_debt=latest_instant(observations, cik, "long_term_debt", as_of),
        quarters_available=len(revenue_quarters),
        stale_days=stale_days,
    )
