"""Level 4 of the checklist — "what do I actually hold?" (CLAUDE.md sections 2, 8 phase 1).

Pure transformations over the frames the IBKR Flex ingester wrote: no network, no database
handle, no hidden state (section 10). A missing input yields ``None``, never an estimate
(section 12) — a position whose price never arrived shows an empty valuation, not a stale
one dressed up as current.

**Everything here is computed independently of IBKR and then checked against it.** The
account report already contains IBKR's own valuation, its own FIFO PnL and its own NAV, so
this module could have copied them. It recomputes them instead, because two independent
derivations that agree are evidence and one copied number is not: :func:`reconcile_nav`
and the realized-PnL tests exist precisely to make the two collide. That is the phase-1
acceptance criterion, and it is also the only way a bug in the lot matching ever surfaces.

**Why the FIFO engine has to exist at all.** IBKR reports a realized PnL, but the fiscal
layer needs lots with their own acquisition date and a cost frozen in the local currency at that day's rate
(section 11), which is a different object from a broker's PnL summary. Building the lots
here means the fiscal layer inherits them instead of re-deriving them from scratch.

A sale with no matching lot — the statement window is 365 days, so a position bought before
it has no buy row — is **reported, never priced at zero cost**. A zero-cost lot would turn
missing history into a spectacular fake profit.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

# Series suffixes written by ingest/ibkr_flex.py and ingest/prices.py.
POSITION_QTY = "position_qty"
POSITION_COST = "position_cost_basis"
CLOSE_RAW = "close_raw"
NAV_TOTAL = "NAV:total"
NAV_CASH = "NAV:cash"
NAV_STOCK = "NAV:stock"

UNCLASSIFIED = "sin clasificar"


@dataclass(frozen=True)
class IncomeSummary:
    """Cash the portfolio produced and the costs it paid, all as reported.

    Attributes:
        dividends_gross: Dividends credited, before withholding.
        withholding: Tax withheld at source. Negative, as IBKR reports it.
        dividends_net: ``dividends_gross + withholding``.
        payments_in_lieu: Kept apart from dividends: it may not be taxed the same way
            (section 11), and merging them would corrupt the only withholding figure the
            panel is allowed to use.
        interest: Broker interest, net of interest paid.
        fees: Non-trade fees (market data, activity, commission adjustments). Negative.
        commissions: Trade commissions. Negative.
        deposits: Cash in and out. Not income — it is what separates a contribution from
            a return, and a NAV that grew because money was added has not grown.
        effective_withholding_rate: ``-withholding / dividends_gross`` — the rate actually
            observed, never an assumed one, and ``None`` when there were no dividends.
            It is an observation about this account, not a statement about tax law.
    """

    dividends_gross: float
    withholding: float
    dividends_net: float
    payments_in_lieu: float
    interest: float
    fees: float
    commissions: float
    deposits: float
    effective_withholding_rate: float | None


@dataclass(frozen=True)
class Reconciliation:
    """This module's valuation against the one IBKR reports, on the same date.

    Attributes:
        as_of: Date both sides describe. ``None`` when they never overlap.
        computed: Market value from stored raw closes (source: yfinance).
        reported: ``NAV:stock`` as IBKR marked it.
        difference: ``computed - reported``.
        relative: ``difference / reported``, or ``None`` when there is nothing to divide by.
        tolerance: Relative gap treated as agreement.
        agrees: Whether ``relative`` is within ``tolerance``. ``None`` when either side is
            missing — unknown is not the same as agreeing, and must not render as a tick.
        detail: One line for the panel, in Spanish.
    """

    as_of: str | None
    computed: float | None
    reported: float | None
    difference: float | None
    relative: float | None
    tolerance: float
    agrees: bool | None
    detail: str


# --- Reading the long-format frame --------------------------------------------


def _latest_by_series(observations: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Most recent row of every ``TICKER:<suffix>`` series, as ``[ticker, value, ts]``."""
    empty = pd.DataFrame(columns=["ticker", "value", "ts"])
    if observations is None or observations.empty:
        return empty
    wanted = observations[observations["series_id"].str.endswith(f":{suffix}")]
    if wanted.empty:
        return empty
    wanted = wanted.sort_values("ts")
    latest = wanted.groupby("series_id", as_index=False).last()
    latest["ticker"] = latest["series_id"].str.rsplit(":", n=1).str[0]
    return latest[["ticker", "value", "ts"]].reset_index(drop=True)


def statement_date(observations: pd.DataFrame) -> str | None:
    """The date of the most recent Flex statement, or ``None`` without account data.

    Read from the daily NAV, which every statement carries even when nothing is held, and
    otherwise from the positions themselves.
    """
    if observations is None or observations.empty:
        return None
    ids = observations["series_id"]
    rows = observations[(ids == NAV_TOTAL) | ids.str.endswith(f":{POSITION_QTY}")]
    return None if rows.empty else str(rows["ts"].max())


def latest_positions(observations: pd.DataFrame) -> pd.DataFrame:
    """Positions open at the last statement: ``[ticker, quantity, cost_basis, ts]``.

    Positions live in ``observations`` rather than a table of their own so that a new field
    in the IBKR report never becomes a migration, and so their history comes for free
    (RESEARCH.md section 2.1).

    ⚠️ **Open means present in the latest statement**, not "the last row of each series".
    Flex reports the positions open at the statement's end and says nothing about the ones
    closed: after a full sale there is simply no new row, and the old one — with its old
    quantity — would stay the series' last word forever. Found 2026-09-24, before any sale
    triggered it.
    """
    columns = ["ticker", "quantity", "cost_basis", "ts"]
    quantities = _latest_by_series(observations, POSITION_QTY).rename(
        columns={"value": "quantity"}
    )
    costs = _latest_by_series(observations, POSITION_COST).rename(
        columns={"value": "cost_basis"}
    )[["ticker", "cost_basis"]]
    if quantities.empty:
        return pd.DataFrame(columns=columns)
    quantities = quantities[quantities["ts"] == statement_date(observations)]
    if quantities.empty:
        return pd.DataFrame(columns=columns)
    merged = quantities.merge(costs, on="ticker", how="left")
    return merged[columns].sort_values("ticker").reset_index(drop=True)


def latest_prices(observations: pd.DataFrame) -> pd.DataFrame:
    """Last stored raw close per ticker: ``[ticker, price, price_ts]``.

    Raw, not adjusted: the adjusted series is a fact plus a viewpoint that changes with
    every future split (section 9.1). For valuing what is held today the raw close *is* the
    price, and it stays comparable with the trade prices actually paid.
    """
    prices = _latest_by_series(observations, CLOSE_RAW)
    return prices.rename(columns={"value": "price", "ts": "price_ts"})


def nav_series(observations: pd.DataFrame, series_id: str = NAV_TOTAL) -> pd.DataFrame:
    """One ``NAV:*`` series in chronological order, as ``[ts, value]``.

    Feeds the equity curve. The public view rebases it to 100 before rendering
    (``app/format.py``), which is what makes it publishable at all.
    """
    empty = pd.DataFrame(columns=["ts", "value"])
    if observations is None or observations.empty:
        return empty
    rows = observations[observations["series_id"] == series_id]
    if rows.empty:
        return empty
    return rows.sort_values("ts")[["ts", "value"]].reset_index(drop=True)


# --- Valuation ----------------------------------------------------------------


def valuation(positions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Value the open positions: ``[ticker, quantity, cost_basis, price, price_ts,
    market_value, unrealized_pnl, unrealized_return, weight]``.

    A position with no stored price keeps its row with ``NaN`` in the priced columns — a
    hole the panel can show — instead of vanishing from the table, which would quietly
    shrink the portfolio.

    ``weight`` is computed over the **priced** subset. Use :func:`unpriced` and say so in
    the interface when it is not empty; otherwise the weights silently describe a
    portfolio that is not the whole one.
    """
    columns = [
        "ticker", "quantity", "cost_basis", "price", "price_ts",
        "market_value", "unrealized_pnl", "unrealized_return", "weight",
    ]
    if positions is None or positions.empty:
        return pd.DataFrame(columns=columns)

    frame = positions.merge(
        prices[["ticker", "price", "price_ts"]] if not prices.empty
        else pd.DataFrame(columns=["ticker", "price", "price_ts"]),
        on="ticker", how="left",
    )
    frame["market_value"] = frame["quantity"] * frame["price"]
    frame["unrealized_pnl"] = frame["market_value"] - frame["cost_basis"]
    frame["unrealized_return"] = frame["unrealized_pnl"] / frame["cost_basis"].where(
        frame["cost_basis"] != 0
    )
    total = frame["market_value"].sum(skipna=True)
    frame["weight"] = frame["market_value"] / total if total else float("nan")
    return frame[columns].sort_values("market_value", ascending=False).reset_index(drop=True)


def unpriced(valued: pd.DataFrame) -> list[str]:
    """Tickers held but not priced. Non-empty means every weight below is partial."""
    if valued is None or valued.empty:
        return []
    return sorted(valued.loc[valued["market_value"].isna(), "ticker"])


def target_drift(
    valued: pd.DataFrame, target_weights: Mapping[str, float] | None
) -> pd.DataFrame:
    """Actual weight against the written target: ``[ticker, weight, target_weight, drift]``.

    Tickers in the target but not held appear with weight 0 — an unfilled target is a drift,
    not an absence. With no targets configured the frame comes back empty rather than
    showing a drift against an implicit zero, which would read as "everything is wrong".
    """
    columns = ["ticker", "weight", "target_weight", "drift"]
    if not target_weights:
        return pd.DataFrame(columns=columns)

    held = (
        valued.set_index("ticker")["weight"].to_dict()
        if valued is not None and not valued.empty else {}
    )
    rows = [
        {
            "ticker": ticker,
            "weight": float(held.get(ticker, 0.0)) if pd.notna(held.get(ticker, 0.0)) else None,
            "target_weight": float(target),
            "drift": (float(held.get(ticker, 0.0)) - float(target))
            if pd.notna(held.get(ticker, 0.0)) else None,
        }
        for ticker, target in sorted({**{t: 0.0 for t in held}, **dict(target_weights)}.items())
        if ticker in target_weights or ticker in held
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("drift").reset_index(drop=True)


# --- FIFO lots ----------------------------------------------------------------


def fifo_lots(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Match sales against purchases, oldest first.

    Args:
        trades: Rows as stored by the Flex ingester — ``quantity`` unsigned, direction in
            ``side``, ``commission`` negative because it is a cost.

    Returns:
        ``(open_lots, disposals, unmatched)``:

        - ``open_lots``: ``[ticker, ts, quantity, price, commission, cost]`` still held.
          ``cost`` includes the share of the buy commission, which is what makes the
          average cost comparable with the broker's.
        - ``disposals``: ``[ticker, ts, quantity, proceeds, cost, realized_pnl,
          acquired_ts, holding_days, buy_commission, sell_commission]``. ``realized_pnl``
          nets **both** commissions, which is IBKR's convention — verified against its own
          FIFO summary in the tests. The two commission shares (negative) are also given
          apart, because a currency conversion needs them at different dates: the buy leg
          at the purchase date's rate, the sell leg at the sale's.
        - ``unmatched``: one label per sale with no lot to match. Those are excluded from
          the disposals rather than matched against an invented zero-cost purchase: the
          usual cause is a 365-day statement window that starts after the position was
          opened, and pricing that at zero would invent a spectacular profit.
    """
    lot_columns = ["ticker", "ts", "quantity", "price", "commission", "cost"]
    disposal_columns = [
        "ticker", "ts", "quantity", "proceeds", "cost", "realized_pnl",
        "acquired_ts", "holding_days", "buy_commission", "sell_commission",
    ]
    if trades is None or trades.empty:
        return pd.DataFrame(columns=lot_columns), pd.DataFrame(columns=disposal_columns), []

    ordered = trades.sort_values(["ts", "trade_id"], kind="stable")
    books: dict[str, deque[dict[str, Any]]] = {}
    disposals: list[dict[str, Any]] = []
    unmatched: list[str] = []

    for row in ordered.to_dict("records"):
        ticker = row["ticker"]
        quantity = abs(float(row["quantity"]))
        price = float(row["price"])
        commission = float(row["commission"] or 0.0)
        per_unit_commission = commission / quantity if quantity else 0.0
        book = books.setdefault(ticker, deque())

        if row["side"] == "buy":
            book.append({
                "ts": row["ts"], "quantity": quantity, "price": price,
                "commission": commission, "per_unit_commission": per_unit_commission,
            })
            continue

        remaining = quantity
        while remaining > 1e-12 and book:
            lot = book[0]
            matched = min(remaining, lot["quantity"])
            realized = (
                (price - lot["price"]) * matched
                + lot["per_unit_commission"] * matched      # buy leg, negative
                + per_unit_commission * matched             # sell leg, negative
            )
            disposals.append({
                "ticker": ticker,
                "ts": row["ts"],
                "quantity": matched,
                "proceeds": price * matched,
                "cost": lot["price"] * matched,
                "realized_pnl": realized,
                "acquired_ts": lot["ts"],
                "holding_days": _days_between(lot["ts"], row["ts"]),
                "buy_commission": lot["per_unit_commission"] * matched,
                "sell_commission": per_unit_commission * matched,
            })
            lot["quantity"] -= matched
            remaining -= matched
            if lot["quantity"] <= 1e-12:
                book.popleft()

        if remaining > 1e-9:
            unmatched.append(
                f"{ticker}: sold {remaining:g} share(s) with no purchase in the data "
                f"(statement window starts after the position was opened)"
            )

    open_lots = [
        {
            "ticker": ticker, "ts": lot["ts"], "quantity": lot["quantity"],
            "price": lot["price"],
            "commission": lot["per_unit_commission"] * lot["quantity"],
            "cost": lot["price"] * lot["quantity"] - lot["per_unit_commission"] * lot["quantity"],
        }
        for ticker, book in books.items()
        for lot in book
    ]
    return (
        pd.DataFrame(open_lots, columns=lot_columns),
        pd.DataFrame(disposals, columns=disposal_columns),
        unmatched,
    )


def _days_between(start: str, end: str) -> int | None:
    """Calendar days between two ISO8601 stamps, or ``None`` if either is unusable."""
    try:
        return (pd.Timestamp(end) - pd.Timestamp(start)).days
    except (ValueError, TypeError):
        return None


def realized_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """Realized PnL per ticker: ``[ticker, realized_pnl, quantity_sold, disposals]``."""
    _lots, disposals, _unmatched = fifo_lots(trades)
    columns = ["ticker", "realized_pnl", "quantity_sold", "disposals"]
    if disposals.empty:
        return pd.DataFrame(columns=columns)
    grouped = disposals.groupby("ticker").agg(
        realized_pnl=("realized_pnl", "sum"),
        quantity_sold=("quantity", "sum"),
        disposals=("quantity", "size"),
    ).reset_index()
    return grouped[columns].sort_values("realized_pnl", ascending=False).reset_index(drop=True)


def average_cost(open_lots: pd.DataFrame) -> pd.DataFrame:
    """Average cost per share of what is still held: ``[ticker, quantity, cost, avg_cost]``.

    Commission-inclusive, so it is comparable with the broker's cost basis and with the
    price a sale would have to clear to break even.
    """
    columns = ["ticker", "quantity", "cost", "avg_cost"]
    if open_lots is None or open_lots.empty:
        return pd.DataFrame(columns=columns)
    grouped = open_lots.groupby("ticker").agg(
        quantity=("quantity", "sum"), cost=("cost", "sum")
    ).reset_index()
    grouped["avg_cost"] = grouped["cost"] / grouped["quantity"].where(grouped["quantity"] != 0)
    return grouped[columns].sort_values("ticker").reset_index(drop=True)


# --- Income and costs ---------------------------------------------------------


def income_summary(
    cash_transactions: pd.DataFrame, trades: pd.DataFrame | None = None
) -> IncomeSummary:
    """Aggregate the cash side of the account. See :class:`IncomeSummary`."""

    def total(kind: str) -> float:
        if cash_transactions is None or cash_transactions.empty:
            return 0.0
        rows = cash_transactions[cash_transactions["kind"] == kind]
        return float(rows["amount"].sum()) if not rows.empty else 0.0

    dividends = total("dividend")
    withholding = total("withholding_tax")
    commissions_total = 0.0
    if trades is not None and not trades.empty and "commission" in trades:
        commissions_total = float(trades["commission"].fillna(0.0).sum())

    return IncomeSummary(
        dividends_gross=dividends,
        withholding=withholding,
        dividends_net=dividends + withholding,
        payments_in_lieu=total("payment_in_lieu"),
        interest=total("interest"),
        fees=total("fee"),
        commissions=commissions_total,
        deposits=total("deposit"),
        # Observed, not assumed (section 11). What it means for the owner's tax residence is a
        # question for an accountant, and this module does not answer it.
        effective_withholding_rate=(-withholding / dividends) if dividends else None,
    )


def cost_drag(income: IncomeSummary, nav: float | None) -> float | None:
    """Commissions and fees as a fraction of NAV, or ``None`` without a NAV.

    The number the panel owes an investor with recurring contributions: on a small account
    a per-order minimum is a large percentage, and over-trading is this project's stated
    first risk (section 2). Making it a *ratio* also keeps it publishable (RESEARCH.md
    section 2.8).
    """
    if not nav:
        return None
    return (income.commissions + income.fees) / nav


# --- Reconciliation against the broker ----------------------------------------


def reconcile_nav(
    observations: pd.DataFrame, *, tolerance: float = 0.005
) -> Reconciliation:
    """Value the positions from stored prices and compare with IBKR's own mark.

    Both sides are taken **on the statement date**, not "latest of each": comparing today's
    close against last week's NAV would manufacture a discrepancy and hide a real one.

    A gap beyond ``tolerance`` is a **visible flag, never an automatic adjustment**
    (RESEARCH.md section 2.3). The two sides are genuinely different measurements — IBKR
    marks with its own feed, this project stores yfinance closes — so disagreement is
    information about the data, which is exactly what section 9.8 asks to keep visible.
    """
    positions = latest_positions(observations)
    reported_series = nav_series(observations, NAV_STOCK)

    if positions.empty or reported_series.empty:
        return Reconciliation(
            None, None, None, None, None, tolerance, None,
            "Sin datos suficientes para reconciliar: falta la cartera o el NAV de IBKR.",
        )

    as_of = str(positions["ts"].max())
    reported_rows = reported_series[reported_series["ts"] <= as_of]
    if reported_rows.empty:
        return Reconciliation(
            as_of, None, None, None, None, tolerance, None,
            f"IBKR no reporta NAV en la fecha de la cartera ({as_of[:10]}).",
        )
    reported = float(reported_rows["value"].iloc[-1])

    prices_on_date = _prices_as_of(observations, as_of)
    valued = valuation(positions, prices_on_date)
    missing = unpriced(valued)
    if missing:
        return Reconciliation(
            as_of, None, reported, None, None, tolerance, None,
            f"No se puede reconciliar: sin precio almacenado para {', '.join(missing)} "
            f"el {as_of[:10]}.",
        )

    computed = float(valued["market_value"].sum())
    difference = computed - reported
    relative = difference / reported if reported else None
    agrees = abs(relative) <= tolerance if relative is not None else None
    detail = (
        f"Cartera valorada en {_es(computed)} USD frente a {_es(reported)} USD que reporta "
        f"IBKR el {as_of[:10]}: {_es(difference, sign=True)} USD "
        f"({_es(relative * 100, sign=True)} %)."
        if relative is not None else
        "IBKR reporta un NAV de posiciones nulo; no hay base para el porcentaje."
    )
    return Reconciliation(
        as_of, computed, reported, difference, relative, tolerance, agrees, detail
    )


def _es(value: float, *, sign: bool = False) -> str:
    """Spanish notation for the Spanish prose this module writes (1.234,56).

    The panel formats in ``app.format``, which ``transform`` must not import; the one
    sentence built here used English notation ("1,234.50") until 2026-09-25.
    """
    text = f"{value:+,.2f}" if sign else f"{value:,.2f}"
    return text.replace(",", "\u00a0").replace(".", ",").replace("\u00a0", ".")


def _prices_as_of(observations: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Last raw close on or before ``as_of`` per ticker — the like-for-like comparison."""
    empty = pd.DataFrame(columns=["ticker", "price", "price_ts"])
    if observations is None or observations.empty:
        return empty
    rows = observations[
        observations["series_id"].str.endswith(f":{CLOSE_RAW}")
        & (observations["ts"] <= as_of)
    ]
    if rows.empty:
        return empty
    latest = rows.sort_values("ts").groupby("series_id", as_index=False).last()
    latest["ticker"] = latest["series_id"].str.rsplit(":", n=1).str[0]
    return latest.rename(columns={"value": "price", "ts": "price_ts"})[
        ["ticker", "price", "price_ts"]
    ]


# --- Concentration (section 5.3) ----------------------------------------------


def concentration(
    valued: pd.DataFrame,
    metadata: Mapping[str, Mapping[str, Any]] | None,
    by: str = "thesis_category",
) -> pd.DataFrame:
    """Group the portfolio by an axis of the written universe.

    Args:
        valued: Output of :func:`valuation`.
        metadata: ``ticker -> {"sector": ..., "thesis_category": ...}``, from the thesis
            cards in settings.local.yaml.
        by: Which key to group on.

    Returns:
        ``[group, market_value, weight, positions, tickers]``, heaviest first.

    Section 5.3 is the reason this is not just a sector chart: eight companies from five
    GICS sectors can be one bet on interest rates staying low. A position with no thesis
    card lands in "sin clasificar" — visible, because an unclassified position is exactly
    the one whose failure mode nobody has written down.
    """
    columns = ["group", "market_value", "weight", "positions", "tickers"]
    if valued is None or valued.empty:
        return pd.DataFrame(columns=columns)

    lookup = metadata or {}
    frame = valued.copy()
    frame["group"] = [
        str(lookup.get(ticker, {}).get(by) or UNCLASSIFIED) for ticker in frame["ticker"]
    ]
    grouped = frame.groupby("group").agg(
        market_value=("market_value", "sum"),
        positions=("ticker", "size"),
        tickers=("ticker", lambda names: ", ".join(sorted(names))),
    ).reset_index()
    total = grouped["market_value"].sum()
    grouped["weight"] = grouped["market_value"] / total if total else float("nan")
    return grouped[columns].sort_values("market_value", ascending=False).reset_index(drop=True)


def thesis_metadata(tracked: Sequence[Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Index the written thesis cards by ticker, for :func:`concentration`."""
    return {
        str(card["ticker"]): dict(card)
        for card in (tracked or [])
        if card.get("ticker")
    }


# --- Performance against the market, net of contributions ------------------------------


@dataclass(frozen=True)
class Performance:
    """How the account did, and what the same money would have done in the benchmark.

    Attributes:
        start / end: The window, first and last NAV dates.
        twr: Time-weighted return — the daily NAV changes chained with each contribution
            taken out, so money coming in is not mistaken for money made. Measures the
            decisions, not the deposits.
        benchmark_return: The benchmark's total return over the same window.
        excess: ``twr − benchmark_return``. Positive = beat it, over this window only.
        nav_end: Last NAV.
        shadow_end: What the starting NAV plus every contribution, each invested in the
            benchmark on its own date, would be worth at the end — "if I had just bought
            the index". The money-weighted comparison, in the account's currency.
        contributions: Net external flows over the window (deposits − withdrawals).
    """

    start: str
    end: str
    twr: float
    benchmark_return: float
    excess: float
    nav_end: float
    shadow_end: float
    contributions: float


def external_flows(cash_transactions: pd.DataFrame) -> pd.Series:
    """Deposits (and withdrawals, negative) per calendar day, from the Flex cash rows."""
    if cash_transactions is None or cash_transactions.empty:
        return pd.Series(dtype=float)
    rows = cash_transactions[cash_transactions["kind"] == "deposit"]
    if rows.empty:
        return pd.Series(dtype=float)
    days = pd.to_datetime(rows["ts"].str[:10])
    return rows["amount"].astype(float).groupby(days.to_numpy()).sum()


def performance(
    nav: pd.DataFrame, flows: pd.Series, benchmark: pd.Series
) -> tuple[Performance | None, pd.DataFrame]:
    """The account against a total-return benchmark, contributions removed.

    Args:
        nav: ``[ts, value]`` daily NAV (``nav_series``).
        flows: Output of :func:`external_flows`. A flow is taken as arriving **before**
            that day's close, which is how the NAV jumps in the statement (verified against
            the real account: the NAV rises by the deposit on the deposit's own date).
        benchmark: Total-return index (dividends added on their dates, section 9.1),
            indexed by date. The benchmark on each NAV date is its latest close at or before.

    Returns:
        ``(summary, curves)`` — ``curves`` has ``[date, account, benchmark]`` rebased to 100
        on the first NAV date, both net of contributions. ``(None, empty)`` when there is
        not enough data; a partial window is not passed off as the full one (section 12).
    """
    empty = pd.DataFrame(columns=["date", "account", "benchmark"])
    if nav is None or len(nav) < 2 or benchmark is None or benchmark.dropna().empty:
        return None, empty
    series = pd.Series(nav["value"].astype(float).to_numpy(),
                       index=pd.to_datetime(nav["ts"].str[:10]))
    series = series[~series.index.duplicated(keep="last")].sort_index()
    bench = benchmark.dropna().sort_index()
    bench_on = bench.reindex(series.index, method="ffill")
    if bench_on.isna().iloc[0]:
        return None, empty
    daily_flows = flows.reindex(series.index, fill_value=0.0) if not flows.empty \
        else pd.Series(0.0, index=series.index)
    # A flow dated outside the NAV calendar (a weekend deposit) lands on the next NAV day.
    if not flows.empty:
        stray = flows[~flows.index.isin(series.index)]
        for day, amount in stray.items():
            later = series.index[series.index > day]
            if len(later):
                daily_flows.loc[later[0]] += float(amount)

    previous = series.shift(1)
    ratio = ((series - daily_flows) / previous).where(previous > 0)
    account_index = ratio.fillna(1.0).cumprod()
    bench_index = bench_on / bench_on.iloc[0]

    units = series.iloc[0] / bench_on.iloc[0]
    for day, amount in daily_flows.iloc[1:].items():
        if amount:
            units += amount / bench_on.loc[day]
    summary = Performance(
        start=series.index[0].date().isoformat(), end=series.index[-1].date().isoformat(),
        twr=float(account_index.iloc[-1] - 1), benchmark_return=float(bench_index.iloc[-1] - 1),
        excess=float(account_index.iloc[-1] - bench_index.iloc[-1]),
        nav_end=float(series.iloc[-1]), shadow_end=float(units * bench_on.iloc[-1]),
        contributions=float(daily_flows.iloc[1:].sum()),
    )
    curves = pd.DataFrame({"date": series.index, "account": account_index.to_numpy() * 100,
                           "benchmark": bench_index.to_numpy() * 100})
    return summary, curves


# --- Money-weighted return, attribution and the cost of cash (§15.1.5, 2026-09-25) -----


@dataclass(frozen=True)
class MoneyWeighted:
    """The internal rate of return of the account's own cash flows.

    Where the time-weighted return (:class:`Performance`) measures the decisions, this one
    measures **the investor's experience**: a deposit that arrived just before a fall weighs
    more than one that arrived after it. The two differ exactly by the timing of the money.

    Attributes:
        annual_rate: The IRR, annualized (``(1 + r)^(365/days)`` convention).
        period_return: The same rate over the window's length — the figure to read when the
            window is shorter than a year, because annualizing a few months magnifies noise.
        days: Length of the window.
    """

    annual_rate: float
    period_return: float
    days: int


def money_weighted_return(nav: pd.DataFrame, flows: pd.Series) -> MoneyWeighted | None:
    """IRR of: −first NAV, −each deposit (+ each withdrawal), +last NAV.

    Solved by bisection on the annual rate in [−99 %, +1000 %]; ``None`` with fewer than
    two NAV points, a window under 30 days, or no sign change to bracket.
    """
    if nav is None or len(nav) < 2:
        return None
    series = pd.Series(nav["value"].astype(float).to_numpy(),
                       index=pd.to_datetime(nav["ts"].astype(str).str[:10])).sort_index()
    start, end = series.index[0], series.index[-1]
    days = (end - start).days
    if days < 30:
        return None
    cash_flows = [(0.0, -float(series.iloc[0]))]
    if flows is not None and not flows.empty:
        for day, amount in flows.items():
            day = pd.Timestamp(day)
            if start < day <= end and amount:
                cash_flows.append(((day - start).days / 365.0, -float(amount)))
    cash_flows.append((days / 365.0, float(series.iloc[-1])))

    def npv(rate: float) -> float:
        return sum(cf / (1 + rate) ** t for t, cf in cash_flows)

    low, high = -0.99, 10.0
    if npv(low) * npv(high) > 0:
        return None
    for _ in range(200):
        mid = (low + high) / 2
        if npv(low) * npv(mid) <= 0:
            high = mid
        else:
            low = mid
    rate = (low + high) / 2
    return MoneyWeighted(annual_rate=rate, period_return=(1 + rate) ** (days / 365.0) - 1,
                         days=days)


def position_attribution(trades: pd.DataFrame, cash_transactions: pd.DataFrame,
                         valued: pd.DataFrame) -> pd.DataFrame:
    """What each ticker added to the account, in dollars, since the statement starts.

    ``[ticker, bought, sold, commissions, income, market_value, pnl, share, complete]``:
    ``pnl = market value today + sales − purchases + commissions (negative) + dividends net
    of withholding``. ``share`` is its part of the summed PnL (only when that sum is positive,
    otherwise a share reads backwards). ``complete`` is False when the ticker sold more than
    the window shows it bought: its purchase is older than the statement and its PnL would
    be inflated, so it is flagged instead of trusted (the FIFO rule of this module).
    """
    columns = ["ticker", "bought", "sold", "commissions", "income", "market_value", "pnl",
               "share", "complete"]
    if trades is None or trades.empty:
        return pd.DataFrame(columns=columns)
    frame = trades.assign(value=trades["quantity"].astype(float) * trades["price"].astype(float),
                          side=trades["side"].str.lower())
    rows = []
    income = {}
    if cash_transactions is not None and not cash_transactions.empty:
        paid = cash_transactions[cash_transactions["kind"].isin(["dividend", "withholding_tax"])]
        income = paid.groupby("ticker")["amount"].sum().to_dict()
    held = (valued.set_index("ticker")["market_value"].to_dict()
            if valued is not None and not valued.empty else {})
    for ticker, group in frame.groupby("ticker"):
        buys, sells = group[group["side"] == "buy"], group[group["side"] == "sell"]
        bought, sold = float(buys["value"].sum()), float(sells["value"].sum())
        commissions = float(pd.to_numeric(group["commission"], errors="coerce").fillna(0).sum())
        value = held.get(ticker)
        value = 0.0 if value is None else float(value)
        if pd.isna(value):
            value = None
        net_income = float(income.get(ticker, 0.0))
        rows.append({
            "ticker": ticker, "bought": bought, "sold": sold, "commissions": commissions,
            "income": net_income, "market_value": value,
            "pnl": None if value is None else value + sold - bought + commissions + net_income,
            "complete": float(sells["quantity"].sum()) <= float(buys["quantity"].sum()) + 1e-9,
        })
    out = pd.DataFrame(rows, columns=[c for c in columns if c != "share"])
    total = out["pnl"].dropna().sum()
    out["share"] = out["pnl"] / total if total > 0 else None
    return out[columns].sort_values("pnl", ascending=False, na_position="last") \
        .reset_index(drop=True)


def cash_drag(nav_total: pd.DataFrame, nav_cash: pd.DataFrame,
              benchmark: pd.Series) -> float | None:
    """Approximate return given up by the cash, against the benchmark, over the window.

    ``Σ w_cash(t−1) × r_benchmark(t)``: each day, the part of the account in cash times what
    the benchmark did that day — what that cash would have added had it been invested in
    it. An arithmetic sum of daily contributions, so an approximation (it ignores
    compounding), labelled as such. Positive = the cash cost return (the market rose);
    negative = the cash protected it. ``None`` without the three series.
    """
    if nav_total is None or nav_cash is None or len(nav_total) < 2 or benchmark is None \
            or benchmark.dropna().empty:
        return None
    total = pd.Series(nav_total["value"].astype(float).to_numpy(),
                      index=pd.to_datetime(nav_total["ts"].astype(str).str[:10])).sort_index()
    cash = pd.Series(nav_cash["value"].astype(float).to_numpy(),
                     index=pd.to_datetime(nav_cash["ts"].astype(str).str[:10])).sort_index()
    weight = (cash / total.where(total > 0)).reindex(total.index)
    bench = benchmark.dropna().sort_index().reindex(total.index, method="ffill")
    daily = bench.pct_change()
    contributions = (weight.shift(1) * daily).dropna()
    return float(contributions.sum()) if len(contributions) else None
