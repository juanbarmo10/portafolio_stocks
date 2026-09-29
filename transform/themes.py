"""Thematic exposure: one reference ETF per theme (CLAUDE.md §15.5, point 10).

A position can beat the market because the company did well or because its whole theme did.
The two are different theses and fail differently: a biotech that rose with XBI has proven
nothing about its own drug. Given a theme's ETF, the return against SPY splits exactly in
two factors:

    (1 + company) / (1 + SPY) = (1 + company) / (1 + theme) × (1 + theme) / (1 + SPY)

- **theme against SPY**: did the whole theme lead or lag the market?
- **company against its theme**: did the company do better or worse than its own theme?

Themes live in ``universe.themes`` (``settings.local.yaml``), each a name, an ETF and the
tickers in it. A card whose ``thesis_category`` equals a theme's name joins it without
being listed. A ticker may sit in several themes (a Brazilian fintech is fintech and Brazil).
The ETF is the user's choice: the panel measures, it does not classify.

Returns are total returns (dividends added on their date, section 9.1), read at or before the
date. A horizon whose anchor sits more than ``ANCHOR_SLACK_DAYS`` before its target, or whose
series starts after it, is ``None`` rather than measured over a shorter span (section 12).

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

ANCHOR_SLACK_DAYS = 7
HORIZONS = {"3m": 91, "6m": 182, "12m": 365}


@dataclass(frozen=True)
class Theme:
    name: str
    etf: str
    tickers: tuple[str, ...]


def problems(raw: Any) -> list[str]:
    """Why ``universe.themes`` cannot be read; empty when it can. ``None`` is no themes."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        return ["must be a list of {name, etf, tickers}"]
    out: list[str] = []
    names: set[str] = set()
    for i, entry in enumerate(raw):
        label = f"themes[{i}]"
        if not isinstance(entry, dict):
            out.append(f"{label}: must be a mapping")
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            out.append(f"{label}: name is required")
        elif name.strip().lower() in names:
            out.append(f"{label}: theme {name!r} is repeated")
        else:
            names.add(name.strip().lower())
        etf = entry.get("etf")
        if not isinstance(etf, str) or not etf.strip():
            out.append(f"{label}: etf is required")
        tickers = entry.get("tickers", [])
        if not isinstance(tickers, list) or not all(isinstance(t, str) and t.strip()
                                                    for t in tickers):
            out.append(f"{label}: tickers must be a list of tickers — a string would be read "
                       "one letter at a time")
        elif isinstance(etf, str) and etf.strip().upper() in {t.upper() for t in tickers}:
            out.append(f"{label}: {etf} is the theme's reference, not a member")
    return out


def parse(raw: Any, cards: Iterable[Mapping[str, Any]] = ()) -> list[Theme]:
    """Themes of a valid ``universe.themes``, each with its listed tickers plus the tracked
    cards whose ``thesis_category`` is the theme's name."""
    by_category: dict[str, list[str]] = {}
    for card in cards:
        category = str(card.get("thesis_category") or "").strip().lower()
        if category and card.get("ticker"):
            by_category.setdefault(category, []).append(str(card["ticker"]).upper())
    out = []
    for entry in raw or []:
        name = str(entry["name"]).strip()
        listed = [str(t).strip().upper() for t in entry.get("tickers") or []]
        members = tuple(dict.fromkeys([*listed, *by_category.get(name.lower(), [])]))
        out.append(Theme(name=name, etf=str(entry["etf"]).strip().upper(), tickers=members))
    return out


def themes_of(ticker: str, themes: Sequence[Theme]) -> list[Theme]:
    return [t for t in themes if ticker.upper() in t.tickers]


def horizon_return(series: pd.Series, as_of: Any, days: int) -> float | None:
    """Return of a (total-return) series from ``as_of − days`` to ``as_of``, both read at or
    before; ``None`` without a close near enough to either end."""
    if series is None or series.empty:
        return None
    day = pd.Timestamp(as_of)
    known = series[series.index <= day].dropna()
    if known.empty or (day - known.index[-1]).days > ANCHOR_SLACK_DAYS:
        return None
    target = known.index[-1] - pd.Timedelta(days=days)
    before = known[known.index <= target]
    if before.empty or (target - before.index[-1]).days > ANCHOR_SLACK_DAYS:
        return None
    start = float(before.iloc[-1])
    return None if start <= 0 else float(known.iloc[-1]) / start - 1


def relative(numerator: float | None, denominator: float | None) -> float | None:
    """``(1 + a) / (1 + b) − 1``: how much ``a`` beat ``b``, compounded (never ``a − b``)."""
    if numerator is None or denominator is None or denominator <= -1:
        return None
    return (1 + numerator) / (1 + denominator) - 1


@dataclass(frozen=True)
class Split:
    """A company's return against SPY, split into its theme's part and its own."""

    horizon: str
    company: float | None
    theme: float | None
    spy: float | None

    @property
    def company_vs_spy(self) -> float | None:
        return relative(self.company, self.spy)

    @property
    def theme_vs_spy(self) -> float | None:
        return relative(self.theme, self.spy)

    @property
    def company_vs_theme(self) -> float | None:
        return relative(self.company, self.theme)


def split(company: pd.Series, theme: pd.Series, spy: pd.Series, as_of: Any,
          horizons: Mapping[str, int] = HORIZONS) -> list[Split]:
    return [Split(name, horizon_return(company, as_of, days), horizon_return(theme, as_of, days),
                  horizon_return(spy, as_of, days))
            for name, days in horizons.items()]


def theme_table(themes: Sequence[Theme], series: Mapping[str, pd.Series], as_of: Any,
                weights: Mapping[str, float] | None = None,
                horizons: Mapping[str, int] = HORIZONS) -> pd.DataFrame:
    """One row per theme: its ETF against SPY at each horizon, and — given the account's
    ``weights`` by ticker — which holdings are in it and how much they weigh together."""
    spy = series.get("SPY", pd.Series(dtype=float))
    weights = weights or {}
    rows = []
    for theme in themes:
        etf = series.get(theme.etf, pd.Series(dtype=float))
        row: dict[str, Any] = {"theme": theme.name, "etf": theme.etf}
        for name, days in horizons.items():
            row[f"vs_spy_{name}"] = relative(horizon_return(etf, as_of, days),
                                             horizon_return(spy, as_of, days))
        held = [t for t in theme.tickers if weights.get(t)]
        row["held"] = ", ".join(held)
        row["weight"] = sum(weights[t] for t in held) if held else None
        row["members"] = ", ".join(theme.tickers)
        rows.append(row)
    return pd.DataFrame(rows)
