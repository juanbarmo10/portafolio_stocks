"""Portfolio risk and position size (CLAUDE.md §15.1.8), and shared failure modes (§5.3).

Two questions a weight alone does not answer:

- **How much of the portfolio's swings comes from each position?** A 10 % weight in a
  stock with 90 % volatility moves the account more than a 30 % weight in a steady one.
  The *risk contribution* (Euler decomposition over the covariance of daily returns) says
  it directly, and adds up to the portfolio's volatility. Cash is a position too: weight,
  zero volatility, zero contribution.
- **What do the positions have in common?** (§5.3: diversification is by failure mode, not
  by number of tickers.) Each holding's weekly return is regressed on a few factor proxies
  built from data the panel already has — the market, the 10-year yield, the dollar, small
  against large, technology against the market, cyclical against defensive consumption.
  A scenario ("rates +1 pp") times those sensitivities says which positions tend to fall
  together. **Estimated from past co-movement, not a forecast**, with the uncertainty
  shown: a sensitivity whose t-statistic is below 2 is marked as not distinguishable
  from zero.

The limits the user writes (``portfolio.max_position``, ``portfolio.max_per_thesis_category``)
are checked, never invented: absent, the page says they are not written.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252
WEEKS = 52


def daily_returns(indices: Mapping[str, pd.Series], end: Any, days: int = 365) -> pd.DataFrame:
    """Daily returns of each series over the window ending on ``end``, sessions in common."""
    start = pd.Timestamp(end) - pd.Timedelta(days=days)
    frame = pd.DataFrame({k: v[(v.index > start) & (v.index <= pd.Timestamp(end))]
                          for k, v in indices.items() if v is not None and not v.empty})
    return frame.sort_index().pct_change().dropna(how="all").iloc[1:] if not frame.empty \
        else frame


@dataclass(frozen=True)
class RiskBreakdown:
    """``table``: ``[ticker, weight, volatility, contribution, share_of_risk]``.
    ``portfolio_volatility``: annualized, cash included. ``correlation``: of the holdings."""

    table: pd.DataFrame
    portfolio_volatility: float | None
    correlation: pd.DataFrame


def risk_breakdown(weights: Mapping[str, float], returns: pd.DataFrame) -> RiskBreakdown:
    """Euler risk contributions: ``w_i · (Σ w)_i / σ_p``, summing to ``σ_p``.

    ``weights`` over the whole account (cash under ``"CASH"``, which gets zero variance).
    A holding without returns is left out of the covariance and reported with ``None``.
    """
    held = [t for t in weights if t != "CASH" and t in returns.columns]
    columns = ["ticker", "weight", "volatility", "contribution", "share_of_risk"]
    rows = []
    if len(held) and len(returns.dropna(subset=held)) > 20:
        clean = returns[held].dropna()
        cov = clean.cov().to_numpy() * TRADING_DAYS
        w = np.array([weights[t] for t in held])
        variance = float(w @ cov @ w)
        sigma = math.sqrt(variance) if variance > 0 else None
        marginal = cov @ w
        for i, t in enumerate(held):
            contribution = float(w[i] * marginal[i] / sigma) if sigma else None
            rows.append({"ticker": t, "weight": float(w[i]),
                         "volatility": float(math.sqrt(cov[i, i])),
                         "contribution": contribution,
                         "share_of_risk": contribution / sigma if sigma and contribution
                         is not None else None})
        correlation = clean.corr()
    else:
        sigma, correlation = None, pd.DataFrame()
    for t, w in weights.items():
        if t not in held:
            rows.append({"ticker": t, "weight": float(w),
                         "volatility": 0.0 if t == "CASH" else None,
                         "contribution": 0.0 if t == "CASH" else None,
                         "share_of_risk": 0.0 if t == "CASH" else None})
    table = pd.DataFrame(rows, columns=columns).sort_values("weight", ascending=False)
    return RiskBreakdown(table.reset_index(drop=True), sigma, correlation)


def limit_breaches(weights: Mapping[str, float], categories: Mapping[str, str | None],
                   max_position: float | None, max_per_category: float | None
                   ) -> list[str]:
    """The user's written limits, checked. Unwritten limits check nothing."""
    out = []
    if max_position is not None:
        out += [f"{t} pesa {w:.1%} (máximo escrito {max_position:.0%})".replace(".", ",")
                for t, w in weights.items() if t != "CASH" and w > max_position]
    if max_per_category is not None:
        totals: dict[str, float] = {}
        for t, w in weights.items():
            if t != "CASH":
                key = categories.get(t) or "sin clasificar"
                totals[key] = totals.get(key, 0.0) + w
        out += [f"«{c}» suma {w:.1%} (máximo escrito {max_per_category:.0%})".replace(".", ",")
                for c, w in totals.items() if w > max_per_category]
    return out


# --- Factor sensitivities and shared failure modes ---------------------------------------

# name → (label, how it is built, scenario shock, scenario label)
FACTORS = {
    "market": ("Mercado (SPY)", "SPY semanal", -0.20, "mercado −20 %"),
    "rates": ("Tipos a 10 años", "cambio semanal del 10 años, en puntos", 1.0,
              "el 10 años sube 1 punto"),
    "dollar": ("Dólar amplio", "dólar amplio semanal", 0.05, "el dólar se aprecia 5 %"),
    "size": ("Pequeñas − grandes", "IWM − SPY semanal", -0.10,
             "las pequeñas pierden 10 % frente a las grandes"),
    "tech": ("Tecnología − mercado", "XLK − SPY semanal", -0.10,
             "la tecnología pierde 10 % frente al mercado"),
    "consumer": ("Consumo cíclico − defensivo", "XLY − XLP semanal", -0.10,
                 "el consumo cíclico pierde 10 % frente al defensivo"),
}


def weekly(series: pd.Series) -> pd.Series:
    """Friday-to-Friday series (the last value of each week). Empty in, empty out — with a
    date index, so the arithmetic that follows does not break on a missing series."""
    clean = series.dropna() if series is not None else pd.Series(dtype=float)
    if clean.empty or not isinstance(clean.index, pd.DatetimeIndex):
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return clean.resample("W-FRI").last().dropna()


def factor_returns(indices: Mapping[str, pd.Series], yield10: pd.Series,
                   dollar: pd.Series) -> pd.DataFrame:
    """Weekly factor proxies (``FACTORS``). ``indices`` holds total-return indices of SPY,
    IWM, XLK, XLY, XLP; ``yield10`` is the 10-year yield in percent; ``dollar`` a level."""
    r = {k: weekly(v).pct_change() for k, v in indices.items()}
    out = pd.DataFrame({
        "market": r.get("SPY"),
        "rates": weekly(yield10).diff() if yield10 is not None and not yield10.empty else None,
        "dollar": weekly(dollar).pct_change() if dollar is not None and not dollar.empty
        else None,
        "size": r.get("IWM") - r.get("SPY") if "IWM" in r and "SPY" in r else None,
        "tech": r.get("XLK") - r.get("SPY") if "XLK" in r and "SPY" in r else None,
        "consumer": r.get("XLY") - r.get("XLP") if "XLY" in r and "XLP" in r else None,
    })
    return out.dropna(how="all")


@dataclass(frozen=True)
class Sensitivity:
    """One stock's weekly-return regression on the factor proxies.

    ``betas`` and ``t_stats`` per factor; ``weeks`` of data; ``r2``. A beta is the stock's
    expected weekly move per unit of the factor, holding the others fixed.
    """

    ticker: str
    betas: dict[str, float]
    t_stats: dict[str, float]
    weeks: int
    r2: float


def sensitivity(stock: pd.Series, factors: pd.DataFrame, ticker: str,
                min_weeks: int = 52) -> Sensitivity | None:
    """OLS of the stock's weekly return on the factors (with intercept). ``None`` below
    ``min_weeks`` of complete data — a year of weeks is the least that says anything."""
    y = weekly(stock).pct_change().rename("y")
    data = pd.concat([y, factors], axis=1, join="inner").dropna()
    if len(data) < min_weeks:
        return None
    x = np.column_stack([np.ones(len(data)), data[factors.columns].to_numpy()])
    target = data["y"].to_numpy()
    coef, *_ = np.linalg.lstsq(x, target, rcond=None)
    residual = target - x @ coef
    dof = max(1, len(data) - x.shape[1])
    sigma2 = float(residual @ residual) / dof
    try:
        cov = sigma2 * np.linalg.inv(x.T @ x)
    except np.linalg.LinAlgError:
        return None
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    names = list(factors.columns)
    total = float(((target - target.mean()) ** 2).sum())
    return Sensitivity(
        ticker=ticker,
        betas={n: float(coef[i + 1]) for i, n in enumerate(names)},
        t_stats={n: float(coef[i + 1] / se[i + 1]) if se[i + 1] > 0 else 0.0
                 for i, n in enumerate(names)},
        weeks=len(data), r2=1 - float(residual @ residual) / total if total else 0.0,
    )


def failure_modes(sensitivities: Sequence[Sensitivity], *, t_threshold: float = 2.0
                  ) -> pd.DataFrame:
    """Scenario × position: the move implied by each sensitivity, and whether it is
    distinguishable from zero. ``[scenario, ticker, move, reliable]``.

    Rows are what the page turns into "these fall together": a scenario with several
    positions reliably negative is a shared failure mode.
    """
    rows = []
    for name, (_label, _how, shock, scenario) in FACTORS.items():
        for s in sensitivities:
            if name not in s.betas:
                continue
            rows.append({"scenario": scenario, "factor": name, "ticker": s.ticker,
                         "move": s.betas[name] * shock,
                         "reliable": abs(s.t_stats[name]) >= t_threshold})
    return pd.DataFrame(rows, columns=["scenario", "factor", "ticker", "move", "reliable"])
