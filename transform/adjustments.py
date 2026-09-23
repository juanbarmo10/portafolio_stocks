"""Corporate-action adjustment, **to a stated cutoff date** (CLAUDE.md section 9.1).

The counterpart of ``ingest/prices.py``. That module un-adjusts on the way in so the
database keeps the raw close, which is true forever; this one re-adjusts on the way out,
for the one calculation that needs it — comparing a price across a split.

**The rule the whole module exists to enforce: an adjusted price is not a fact, it is a
fact plus a date.** "The adjusted close of AAPL on 2020-08-28" has no answer. It was
499.23 before the 2020 split and 124.81 after, and the next split will change it again.
The question only becomes answerable as "the adjusted close *as known on* some date", and
that date is a required argument here, never a default:

    adjusted(d | cutoff) = raw(d) / ∏ { ratio : ex_date of the split is
                                        strictly after d, and on or before cutoff }

Two boundaries carry all the risk:

**Strictly after ``d``.** The bar on a split's own ex-date is already quoted post-split,
so its factor is 1. Including it multiplies that single bar by the ratio — a 4x spike on
one day, invisible in a yearly chart and ruinous in any return calculation. Comparing
normalized calendar dates, not timestamps (``ingest.prices.split_factor`` has the measured
example).

**On or before ``cutoff``.** This is the look-ahead guard, and it is the direct analogue
of ``ts_release`` in section 9.4: a backtest standing on 2020-06-01 did not know about the
August split, so its series must not contain it. Because the adjustment is applied to
*every earlier bar*, forgetting this contaminates the whole history rather than one row —
which is why section 9.1 calls it more insidious than the look-ahead it mirrors. The data
*looks* clean either way.

**Dividends are never adjusted into the price.** Section 9.1 is explicit: a total return
adds the dividend on its own date. Back-adjusting prices for dividends is what makes a
"clean-looking" series silently depend on every payment made since, and it is the reason a
downloaded adjusted series is not reproducible. :func:`total_return_index` does the former.

Pure functions over the frames already stored: no network, no database handle (section 10).
A missing input yields an empty frame or ``None``, never an estimate (section 12).
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping, Sequence

import pandas as pd

SPLIT = "split"
DIVIDEND = "dividend"

# Split ratios that are ordinary. A real split is a small rational number a board voted
# on: 2:1, 3:1, 4:1, 3:2, 1:10. Anything else is usually not a split at all — yfinance
# encodes spin-offs as splits, and XLF's 2016 real-estate spin-off arrives as a "split" of
# ratio 1.231 (section 9.2). Used to flag, never to correct.
CLEAN_RATIOS = frozenset({
    1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 20.0,
    0.5, 1 / 3, 0.25, 0.2, 1 / 6, 0.125, 0.1, 0.05,
})
RATIO_TOLERANCE = 1e-6


def as_date(value: Any) -> dt.date | None:
    """Coerce an ISO8601 string, date or datetime to a plain date, or ``None``."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def as_frame(data: Sequence[Mapping[str, Any]] | pd.DataFrame | None) -> pd.DataFrame:
    """Accept records or a frame, so callers pass whatever the database handed them."""
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(list(data))


def actions_until(
    actions: Sequence[Mapping[str, Any]] | pd.DataFrame | None,
    ticker: str,
    as_of: Any,
    kind: str = SPLIT,
    source: str | None = None,
) -> dict[dt.date, float]:
    """Corporate actions of one kind for one ticker, **knowable on ``as_of``**.

    Args:
        actions: Rows of the ``corporate_actions`` table.
        ticker: Ticker as stored (the table keys on ticker, because ETFs have no CIK —
            RESEARCH.md section 1.5).
        as_of: Cutoff. Actions with a later ex-date are excluded; that exclusion is the
            point of the function.
        kind: ``'split'`` or ``'dividend'``.
        source: Restrict to one provider, e.g. ``'yfinance'`` or ``'ibkr'``. ``None``
            takes every source, which is only correct when one provider is present —
            mixing them double-counts (section 9.8).

    Returns:
        ``{ex_date: value}`` — the ratio for splits, the per-share amount for dividends.
    """
    frame = as_frame(actions)
    if frame.empty:
        return {}

    cutoff = as_date(as_of)
    rows = frame[(frame["ticker"] == ticker) & (frame["kind"] == kind)]
    if source is not None:
        rows = rows[rows["source"] == source]

    column = "ratio" if kind == SPLIT else "amount"
    out: dict[dt.date, float] = {}
    for row in rows.to_dict("records"):
        ex_date = as_date(row.get("ex_date"))
        value = row.get(column)
        if ex_date is None or value is None or pd.isna(value):
            continue
        if cutoff is not None and ex_date > cutoff:
            continue
        # Same ex-date twice means the same event reported twice; summing dividends would
        # double the payment and multiplying ratios would square the split.
        out[ex_date] = float(value)
    return out


def adjustment_factor(bar_date: Any, splits: Mapping[dt.date, float]) -> float:
    """Divisor that turns a raw close on ``bar_date`` into a split-adjusted one.

    The product of every ratio whose ex-date is **strictly after** ``bar_date``. The
    inverse of :func:`ingest.prices.split_factor`, deliberately: the two must agree or a
    round trip through the database changes the number.
    """
    day = as_date(bar_date)
    if day is None:
        return 1.0
    factor = 1.0
    for ex_date, ratio in splits.items():
        if ex_date > day and ratio:
            factor *= float(ratio)
    return factor


def split_adjusted(
    prices: pd.DataFrame,
    actions: Sequence[Mapping[str, Any]] | pd.DataFrame | None,
    ticker: str,
    as_of: Any,
    *,
    source: str | None = "yfinance",
) -> pd.DataFrame:
    """Raw closes re-expressed in the share units in force on ``as_of``: ``[ts, value]``.

    Args:
        prices: ``[ts, value]`` raw closes, as stored. Extra columns are ignored.
        actions: Rows of ``corporate_actions``.
        ticker: Which ticker's splits to apply.
        as_of: The cutoff. **Required** — see the module docstring.
        source: Provider whose splits to use; one provider only (section 9.8).

    Returns:
        ``[ts, value]`` on the same rows. With no splits on or before the cutoff the
        output equals the input, which is the correct answer, not a no-op to optimize away.
    """
    columns = ["ts", "value"]
    if prices is None or prices.empty:
        return pd.DataFrame(columns=columns)

    splits = actions_until(actions, ticker, as_of, SPLIT, source)
    out = prices.sort_values("ts")[columns].reset_index(drop=True).copy()
    if not splits:
        return out
    factors = [adjustment_factor(ts, splits) for ts in out["ts"]]
    out["value"] = out["value"].astype(float) / pd.Series(factors, index=out.index)
    return out


def total_return_index(
    prices: pd.DataFrame,
    actions: Sequence[Mapping[str, Any]] | pd.DataFrame | None,
    ticker: str,
    as_of: Any,
    *,
    source: str | None = "yfinance",
    base: float = 100.0,
) -> pd.DataFrame:
    """Total-return index: price performance **plus dividends added on their own dates**.

    Section 9.1 requires this shape rather than a dividend-adjusted price series. The
    difference is not cosmetic: back-adjusting for dividends rewrites every earlier bar
    each time a payment is made, so the series is only reproducible by someone who knows
    the exact payment history the provider had that day. Adding the cash on its ex-date
    leaves the price series alone and keeps the result reconstructible from stored facts.

    The dividend is treated as reinvested at that day's close, which is the standard total
    return convention and the one that makes the index comparable with an ETF's.

    Returns:
        ``[ts, value]`` rebased to ``base`` at the first priced bar. Empty when there is
        nothing to index. Scale-invariant, so it is safe in public mode (RESEARCH.md 2.8).
    """
    columns = ["ts", "value"]
    adjusted = split_adjusted(prices, actions, ticker, as_of, source=source)
    if adjusted.empty:
        return pd.DataFrame(columns=columns)

    splits = actions_until(actions, ticker, as_of, SPLIT, source)
    dividends = actions_until(actions, ticker, as_of, DIVIDEND, source)

    # The stored dividend is per raw share; the price series is in adjusted units, so the
    # cash per adjusted share is scaled by the same factor. Skipping this pays the
    # post-split dividend on a pre-split share count.
    per_share = {
        ex_date: amount / adjustment_factor(ex_date, splits)
        for ex_date, amount in dividends.items()
    }

    values = adjusted["value"].astype(float).tolist()
    stamps = [as_date(ts) for ts in adjusted["ts"]]

    index: list[float] = []
    level = base
    previous: float | None = None
    for day, price in zip(stamps, values):
        if pd.isna(price) or price <= 0:
            index.append(float("nan"))
            continue
        if previous is not None:
            cash = per_share.get(day, 0.0) if day is not None else 0.0
            level *= (price + cash) / previous
        previous = price
        index.append(level)

    return pd.DataFrame({"ts": adjusted["ts"], "value": index}, columns=columns)


def unclean_split_ratios(
    actions: Sequence[Mapping[str, Any]] | pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Splits whose ratio is not a plain rational number — probably not splits.

    Section 9.2: a board votes a split as 2:1, 3:1, 4:1 or 3:2. A ratio of 1.231 is not a
    split, it is what a spin-off looks like when a provider encodes it as one — measured on
    XLF's 2016 real-estate separation, where yfinance reports a "split" of exactly that
    ratio. The distinction matters because a spin-off **divides the cost base between two
    entities**; treating it as a split leaves the base whole on the original and falsifies
    the PnL permanently.

    **Flagged, never corrected** (section 9.2). Returns the offending rows so the panel can
    say which position needs a human.
    """
    frame = as_frame(actions)
    if frame.empty:
        return []

    out: list[dict[str, Any]] = []
    for row in frame[frame["kind"] == SPLIT].to_dict("records"):
        ratio = row.get("ratio")
        if ratio is None or pd.isna(ratio):
            continue
        ratio = float(ratio)
        if any(abs(ratio - clean) <= RATIO_TOLERANCE for clean in CLEAN_RATIOS):
            continue
        out.append({
            "ticker": row.get("ticker"),
            "ex_date": row.get("ex_date"),
            "ratio": ratio,
            "source": row.get("source"),
            "detail": (
                f"Ratio {ratio:g} no es un split limpio. Un spin-off codificado como "
                "split reparte la base de coste entre dos entidades; tratarlo como split "
                "la deja entera en la original y falsea el PnL para siempre (§9.2)."
            ),
        })
    return sorted(out, key=lambda row: (str(row["ticker"]), str(row["ex_date"])))
