"""Validation metrics: forward returns, permutation test, Benjamini-Hochberg (section 8 phase 4).

Ported from cryptodash (``validation/metrics.py``) with one change of substance, the
sampling (:func:`grid_dates`), explained there. Pure functions over prices and dates; the
point-in-time discipline of the signals lives in whoever builds them (section 9.4).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def grid_dates(index: pd.DatetimeIndex, step_days: int, offset: int = 0) -> pd.DatetimeIndex:
    """Sessions at least ``step_days`` apart: the first at ``offset``, then the first on or
    after each previous one plus the step.

    **Why a grid.** A regime signal is sticky — red for weeks at a time — and forward
    returns over a horizon overlap from one day to the next. Counting every red day as an
    observation turns one three-month episode into sixty "independent" data points and
    makes almost anything look significant. Sampling both groups on one grid of
    non-overlapping windows makes ``n`` an honest count and the two groups disjoint by
    construction — section 8's "disjoint random-date baseline". cryptodash thinned only the
    signal and kept every baseline day; here both sides are sampled the same way.
    """
    if len(index) == 0:
        return index
    index = index.sort_values()
    chosen = []
    position = min(offset, len(index) - 1)
    while position < len(index):
        chosen.append(index[position])
        target = index[position] + pd.Timedelta(days=step_days)
        position = int(index.searchsorted(target))
    return pd.DatetimeIndex(chosen)


def forward_return(prices: pd.Series, at: pd.Timestamp, horizon_days: int) -> float | None:
    """Return from the close of the **first session after** ``at`` to the first close at
    least ``horizon_days`` later. ``None`` when either end is missing.

    Entering the next session, not the same one: a signal dated ``at`` rests on data
    published during that day, and some of it lands after the close (the Fed's H.4.1 comes
    out at 16:30 New York time). Entering at that same close would trade on it before it
    existed. Erring late costs a day of return; erring early is look-ahead (section 9.4).
    """
    series = prices.dropna()
    after = series.loc[series.index > at]
    if after.empty or after.iloc[0] <= 0:
        return None
    entry_day, entry = after.index[0], float(after.iloc[0])
    exit_ = series.loc[series.index >= entry_day + pd.Timedelta(days=horizon_days)]
    if exit_.empty:
        return None
    return float(exit_.iloc[0]) / entry - 1.0


def permutation_pvalue(
    signal: Sequence[float], baseline: Sequence[float], *, n: int = 10_000, seed: int = 0
) -> float | None:
    """Two-sided permutation p-value that the two groups' means differ.

    Pools both samples and reshuffles the labels ``n`` times; the p-value is the share of
    shuffles whose absolute mean difference is at least the observed one, with the usual
    +1 correction so it is never exactly zero (a finite shuffle cannot prove p = 0).
    ``None`` if either group is empty.
    """
    sig = np.asarray(signal, dtype=float)
    base = np.asarray(baseline, dtype=float)
    if sig.size == 0 or base.size == 0:
        return None
    observed = abs(sig.mean() - base.mean())
    pool = np.concatenate([sig, base])
    rng = np.random.default_rng(seed)
    order = rng.random((n, pool.size)).argsort(axis=1)
    shuffled = pool[order]
    diffs = np.abs(shuffled[:, :sig.size].mean(axis=1) - shuffled[:, sig.size:].mean(axis=1))
    # A tiny tolerance: identical sums computed in another order differ in the last bit.
    extreme = int((diffs >= observed - 1e-12).sum())
    return (extreme + 1) / (n + 1)


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.10) -> list[tuple[float, bool]]:
    """Benjamini-Hochberg over a family of p-values: ``(qvalue, significant)`` per input.

    Testing thirty signal-horizon pairs at 5 % each expects one or two "discoveries" by
    luck alone. BH controls the *false discovery rate* — the expected share of false ones
    among those declared significant — across the whole battery (section 9.7).
    """
    m = len(pvalues)
    if m == 0:
        return []
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
    q = np.empty(m)
    q[order] = adjusted
    return [(float(q[i]), bool(q[i] <= alpha)) for i in range(m)]
