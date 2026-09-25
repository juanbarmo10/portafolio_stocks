"""How the stock has behaved: against the market, its risk, and around its results.

Level 3 context for the company page (CLAUDE.md §15.1.2). Daily resolution only — the user
chose a daily price chart (2026-09-25), and nothing here is intraday or live (§2).

Everything is built on the **total-return index** (dividends added on their dates, §9.1),
so a dividend payer is not penalised against the benchmark, and on closes up to the date
being viewed: the page's date picker applies here too.

**Reaction to results.** An 8-K item 2.02 is filed either before the open or after the
close, and its filing date does not say which. The window therefore runs from the close of
the session **before** the filing date to the close of the session **after** it — two
sessions that contain the reaction in both cases, and a little market noise that the
excess over the benchmark takes out.

Pure functions over series; no network, no database (section 10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def as_series(frame: pd.DataFrame) -> pd.Series:
    """``[ts, value]`` rows (as ``transform.adjustments`` returns them) → a dated series."""
    if frame is None or frame.empty:
        # Dated even when empty, so a comparison against a date works on an empty series.
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    series = pd.Series(pd.to_numeric(frame["value"], errors="coerce").to_numpy(),
                       index=pd.to_datetime(frame["ts"].astype(str).str[:10]))
    return series[~series.index.duplicated(keep="last")].sort_index().dropna()


@dataclass(frozen=True)
class Behaviour:
    """The stock over a window ending on ``end``, against a benchmark.

    Attributes:
        total_return / benchmark_return / excess: Over the window, dividends included.
        volatility: Annualized standard deviation of daily returns.
        beta: Covariance with the benchmark's daily returns over its variance — how much
            of the market's move it tends to amplify. ``None`` below 60 common sessions.
        drawdown_now: Fall from the highest close in the window to the last one.
        max_drawdown: Worst peak-to-trough fall inside the window.
    """

    start: str
    end: str
    total_return: float | None
    benchmark_return: float | None
    excess: float | None
    volatility: float | None
    beta: float | None
    drawdown_now: float | None
    max_drawdown: float | None


def _window(series: pd.Series, end: Any, days: int) -> pd.Series:
    end = pd.Timestamp(end)
    return series[(series.index > end - pd.Timedelta(days=days)) & (series.index <= end)]


def behaviour(stock: pd.Series, benchmark: pd.Series, end: Any, days: int = 365) -> Behaviour:
    """See :class:`Behaviour`. Missing data leaves the affected fields ``None``."""
    s = _window(stock, end, days)
    b = _window(benchmark, end, days)
    if len(s) < 2:
        return Behaviour(str(pd.Timestamp(end).date()), str(pd.Timestamp(end).date()),
                         None, None, None, None, None, None, None)
    total = float(s.iloc[-1] / s.iloc[0] - 1)
    bench = None
    if len(b) >= 2:
        b_aligned = b.reindex(s.index, method="ffill").dropna()
        if len(b_aligned) >= 2:
            bench = float(b_aligned.iloc[-1] / b_aligned.iloc[0] - 1)
    returns = s.pct_change().dropna()
    volatility = float(returns.std() * math.sqrt(TRADING_DAYS)) if len(returns) > 20 else None
    beta = None
    if len(b) >= 2:
        both = pd.concat({"s": returns, "b": b.pct_change()}, axis=1, join="inner").dropna()
        if len(both) >= 60 and both["b"].var() > 0:
            beta = float(both["s"].cov(both["b"]) / both["b"].var())
    running_max = s.cummax()
    return Behaviour(
        start=s.index[0].date().isoformat(), end=s.index[-1].date().isoformat(),
        total_return=total, benchmark_return=bench,
        excess=None if bench is None else total - bench,
        volatility=volatility, beta=beta,
        drawdown_now=float(s.iloc[-1] / running_max.iloc[-1] - 1),
        max_drawdown=float((s / running_max - 1).min()),
    )


def rebased(series: pd.Series, start: Any) -> pd.Series:
    """The series from ``start`` on, rebased to 100 on its first value."""
    tail = series[series.index >= pd.Timestamp(start)]
    return tail / tail.iloc[0] * 100 if len(tail) else tail


def earnings_reactions(stock: pd.Series, benchmark: pd.Series, dates: Sequence[Any]
                       ) -> pd.DataFrame:
    """Move around each results filing: ``[date, move, benchmark_move, excess]``.

    From the close of the session before the filing date to the close of the session after
    it (module docstring). A filing whose window is not complete in the data is left out.
    """
    columns = ["date", "move", "benchmark_move", "excess"]
    rows = []
    index = stock.index
    for raw in dates:
        day = pd.Timestamp(str(raw)[:10])
        before = index[index < day]
        after = index[index > day]
        if not len(before) or not len(after):
            continue
        start, end = before[-1], after[0]
        move = float(stock.loc[end] / stock.loc[start] - 1)
        bench = None
        if not benchmark.empty:
            b0 = benchmark[benchmark.index <= start]
            b1 = benchmark[benchmark.index <= end]
            if len(b0) and len(b1):
                bench = float(b1.iloc[-1] / b0.iloc[-1] - 1)
        rows.append({"date": day.date().isoformat(), "move": move, "benchmark_move": bench,
                     "excess": None if bench is None else move - bench})
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("date", ascending=False).reset_index(drop=True)


def typical_reaction(reactions: pd.DataFrame) -> float | None:
    """Median absolute excess move: how much results usually move this stock by."""
    values = pd.to_numeric(reactions.get("excess"), errors="coerce").dropna() \
        if not reactions.empty else pd.Series(dtype=float)
    return float(np.median(values.abs())) if len(values) else None
