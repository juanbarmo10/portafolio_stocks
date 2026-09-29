"""How a rule is judged. Three honest tests, a strategy simulator, and a contribution simulator.

Every test enters at the close of the **session after** the signal (a signal dated ``d``
rests on data published during ``d``, some after the close: entering on ``d`` would be
look-ahead, §9.4), measures **total return in excess of the benchmark**, and reports:

- ``n``: independent observations — dates on a grid of non-overlapping windows, never every
  day of a sticky signal (``validation/metrics.grid_dates``);
- ``mean`` and the share of observations of the expected sign;
- ``p``: a permutation or sign-flip test against "no effect";
- the two **halves** of the period separately: an effect that lives in one half only is a
  regime, not a rule (the lesson of §2.49, §2.65);
- and it is **written to the registry** (``lab.registry``), so the multiple-testing
  correction counts every rule you ever tried, not just today's.

The three tests:

``timing``        a market signal (on/off per session): the asset's return after "on" dates
                  against "off" dates on the same grid.
``cross_section`` a score or flag per stock and date: within each date, the flagged (or top
                  group) minus the rest of that day's members. Comparing inside the date
                  cancels the market's move (the lesson of §2.65).
``events``        dated events per stock: excess return after them against the same stocks
                  on ordinary dates; optionally a placebo (the same events shifted back).

``strategy`` simulates a long-only portfolio rebalanced on a schedule, with costs, against
the benchmark. **It is a simulation, not a validation**: a curve that beats SPY proves
nothing until the rule behind it passes the tests above and the holdout.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from lab import registry
from lab.data import BENCHMARK, Lab
from validation.metrics import grid_dates, permutation_pvalue, sign_flip_pvalue

FILL = 5   # sessions a price hole is bridged; longer is a delisting, not a hole


@dataclass
class Result:
    """One test of one rule at one horizon."""

    rule: str
    kind: str
    horizon: int
    n: int
    mean: float | None             # mean excess (difference) per observation
    hit: float | None              # share of observations with the expected sign
    p: float | None
    halves: tuple[float | None, float | None]
    period: tuple[str | None, str | None]
    params: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def row(self) -> dict[str, Any]:
        out = asdict(self)
        out["half_1"], out["half_2"] = out.pop("halves")
        out["from"], out["to"] = out.pop("period")
        return out

    def __str__(self) -> str:
        def pct(x):
            return "—" if x is None or pd.isna(x) else f"{x * 100:+.2f} %"
        p = "—" if self.p is None else f"{self.p:.3f}"
        return (f"{self.rule} [{self.kind}, {self.horizon} d] n={self.n} media={pct(self.mean)}"
                f" p={p} mitades={pct(self.halves[0])} / {pct(self.halves[1])}")


def table(results: Sequence[Result]) -> pd.DataFrame:
    """Results as a frame, one row each."""
    return pd.DataFrame([r.row() for r in results])


def _window(lab: Lab, start: Any, end: Any, holdout: bool, *, timing: bool = False
            ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The dates a test may use. By default everything **before** ``lab.holdout_from``
    (settings ``lab.holdout_from``): the holdout is looked at once, at the end, with
    ``holdout=True`` — never while the rule is still being shaped (§9.7). Market tests start
    at ``first_date_timing`` (the data decides where each signal begins); stock tests at
    ``first_date``, where the members' closes begin."""
    cfg = lab.settings.raw.get("lab", {}) or {}
    default = cfg.get("first_date_timing", "1999-01-01") if timing \
        else cfg.get("first_date", "2017-01-01")
    first = pd.Timestamp(start or default)
    cut = cfg.get("holdout_from")
    last = pd.Timestamp(end) if end else pd.Timestamp.today()
    if cut and not holdout:
        last = min(last, pd.Timestamp(cut))
    elif cut and holdout:
        first = max(first, pd.Timestamp(cut))
    return first, last


def _halves(dates: Sequence[pd.Timestamp], values: Sequence[float]
            ) -> tuple[float | None, float | None]:
    if len(values) < 4:
        return None, None
    order = np.argsort(np.asarray(dates, dtype="datetime64[ns]"))
    v = np.asarray(values, dtype=float)[order]
    mid = len(v) // 2
    return float(v[:mid].mean()), float(v[mid:].mean())


def _entry_exit(index: pd.DatetimeIndex, day: pd.Timestamp, horizon: int
                ) -> tuple[int, int] | None:
    entry = index.searchsorted(day, side="right")
    if entry >= len(index):
        return None
    exit_ = index.searchsorted(index[entry] + pd.Timedelta(days=horizon))
    return None if exit_ >= len(index) else (entry, exit_)


def forward_excess(tr: pd.DataFrame, bench: pd.Series, day: pd.Timestamp, horizon: int,
                   *, future: pd.DataFrame | None = None) -> pd.Series:
    """Per column: total return from the session after ``day`` to ``horizon`` days later,
    minus the benchmark's over the same sessions. Empty past the data. ``future``: ``tr``
    already back-filled (pass it when calling in a loop)."""
    ends = _entry_exit(tr.index, day, horizon)
    if ends is None:
        return pd.Series(dtype=float)
    future = tr.bfill(limit=FILL) if future is None else future
    a, b = future.iloc[ends[0]], future.iloc[ends[1]]
    raw = (b / a.where(a > 0) - 1).dropna()
    ba, bb = bench.iloc[ends[0]], bench.iloc[ends[1]]
    if pd.isna(ba) or pd.isna(bb) or ba <= 0:
        return pd.Series(dtype=float)
    return raw - (bb / ba - 1)


def excess_at(future: pd.DataFrame, bench: pd.Series, tickers: Sequence[str],
              days: Sequence[pd.Timestamp], horizon: int) -> np.ndarray:
    """Vectorized :func:`forward_excess` for pairs ``(tickers[i], days[i])``: NaN where a
    close is missing or the window runs past the data."""
    index = future.index
    when = pd.DatetimeIndex(days)
    entry = index.searchsorted(when, side="right")
    ok = entry < len(index)
    exit_ = np.full(len(when), len(index))
    exit_[ok] = index.searchsorted(index[entry[ok]] + pd.Timedelta(days=horizon))
    ok &= exit_ < len(index)
    cols = future.columns.get_indexer(list(tickers))
    ok &= cols >= 0
    values = future.to_numpy(dtype=float)
    b = bench.reindex(index).to_numpy(dtype=float)
    out = np.full(len(when), np.nan)
    e, x, c = entry[ok], exit_[ok], cols[ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        stock = values[x, c] / np.where(values[e, c] > 0, values[e, c], np.nan) - 1
        market = b[x] / np.where(b[e] > 0, b[e], np.nan) - 1
    out[ok] = stock - market
    return out


# --- 1. Market timing ---------------------------------------------------------------------


def timing(lab: Lab, signal: pd.Series, *, rule: str, horizons: Sequence[int] = (30, 90, 180),
           asset: str = BENCHMARK, expected: str = "lower", start: Any = None, end: Any = None,
           holdout: bool = False, params: Mapping[str, Any] | None = None,
           record: bool = True) -> list[Result]:
    """``signal``: 1/True = "on", 0/False = "off", NaN = unknown, per session. The asset's
    total return after "on" dates minus after "off" dates, on a disjoint grid per horizon.
    ``expected``: "lower" if the hypothesis is that "on" precedes worse returns (a brake),
    "higher" otherwise — it only sets the sign of ``hit``."""
    first, last = _window(lab, start, end, holdout, timing=True)
    # Cut at the window's end: a date before the holdout whose forward window runs into it
    # would be reading the reserved period.
    tr = lab.total_return([asset])[asset].dropna().loc[:last]
    sessions = tr.index[tr.index >= first]
    on_signal = signal.reindex(tr.index).ffill(limit=FILL)
    out = []
    for h in horizons:
        on, off, on_days = [], [], []
        for day in grid_dates(sessions, int(h)):
            ends = _entry_exit(tr.index, day, int(h))
            if ends is None:
                continue
            state = on_signal.get(day)
            if state is None or pd.isna(state):
                continue
            ret = tr.iloc[ends[1]] / tr.iloc[ends[0]] - 1
            (on if bool(state) else off).append(ret)
            if bool(state):
                on_days.append(day)
        diff = (np.mean(on) - np.mean(off)) if on and off else None
        sign = -1 if expected == "lower" else 1
        hit = float(np.mean([(r - np.mean(off)) * sign > 0 for r in on])) if on and off else None
        base = float(np.mean(off)) if off else 0.0
        result = Result(rule, "timing", int(h), len(on), diff, hit,
                        permutation_pvalue(on, off) if len(on) >= 3 and len(off) >= 3 else None,
                        _halves(on_days, [r - base for r in on]),
                        (str(sessions.min().date()) if len(sessions) else None,
                         str(sessions.max().date()) if len(sessions) else None),
                        dict(params or {}),
                        f"{len(on)} fechas «on» frente a {len(off)} «off»")
        out.append(result)
    if record:
        registry.record(out, holdout=holdout)
    return out


# --- 2. Cross-section ---------------------------------------------------------------------


def cross_section(lab: Lab, scores: Callable[[pd.Timestamp, list[str]], pd.Series], *,
                  rule: str, horizons: Sequence[int] = (90, 180), mode: str = "flag",
                  groups: int = 5, min_names: int = 5, expected: str = "higher",
                  benchmark: str = BENCHMARK, start: Any = None, end: Any = None,
                  holdout: bool = False, params: Mapping[str, Any] | None = None,
                  record: bool = True) -> tuple[list[Result], pd.DataFrame]:
    """``scores(day, members) -> Series`` by ticker: booleans for ``mode="flag"`` (flagged
    minus the rest of the day's members), numbers for ``mode="top"`` (top group minus all)
    or ``mode="spread"`` (top minus bottom group). Scores must use only what was known on
    ``day``. Returns ``(results, per-date table)``."""
    first, last = _window(lab, start, end, holdout)
    tr = lab.total_return().loc[:last]          # forward windows never cross ``last``
    bench = tr[benchmark]
    future = tr.bfill(limit=FILL)
    sessions = tr.index[tr.index >= first]
    rows = []
    for h in horizons:
        for day in grid_dates(sessions, int(h)):
            members = lab.members_on(day)
            if len(members) < min_names:
                continue
            excess = forward_excess(tr[members], bench, day, int(h),
                                    future=future[members])
            if excess.empty:
                continue
            s = scores(day, members)
            s = s.reindex(excess.index).dropna() if s is not None else pd.Series(dtype=float)
            if s.empty:
                continue
            ex = excess.reindex(s.index)
            if mode == "flag":
                flagged = s.astype(bool)
                if flagged.sum() < min_names or (~flagged).sum() < min_names:
                    continue
                value = ex[flagged].mean() - ex[~flagged].mean()
                names = int(flagged.sum())
            else:
                if len(s) < groups * min_names:
                    continue
                ranks = pd.qcut(s.rank(method="first"), groups, labels=False)
                top = ex[ranks == groups - 1].mean()
                value = top - (ex[ranks == 0].mean() if mode == "spread" else ex.mean())
                names = int((ranks == groups - 1).sum())
            rows.append({"horizon": int(h), "date": day, "value": float(value), "names": names})
    frame = pd.DataFrame(rows, columns=["horizon", "date", "value", "names"])
    out = []
    sign = 1 if expected == "higher" else -1
    for h in horizons:
        part = frame[frame["horizon"] == int(h)]
        values = part["value"].tolist()
        out.append(Result(
            rule, f"cross_section:{mode}", int(h), len(values),
            float(np.mean(values)) if values else None,
            float(np.mean([v * sign > 0 for v in values])) if values else None,
            sign_flip_pvalue(values) if len(values) >= 5 else None,
            _halves(part["date"].tolist(), values),
            (str(part["date"].min().date()) if len(part) else None,
             str(part["date"].max().date()) if len(part) else None),
            dict(params or {}),
            f"mediana de {int(part['names'].median()) if len(part) else 0} acciones por fecha"))
    if record:
        registry.record(out, holdout=holdout)
    return out, frame


# --- 3. Events ----------------------------------------------------------------------------


def events(lab: Lab, dated: pd.DataFrame, *, rule: str, horizons: Sequence[int] = (30, 90, 180),
           benchmark: str = BENCHMARK, baseline_step: int = 30, away_days: int = 0,
           placebo_days: int | None = 120,
           members_only: bool = True, expected: str = "higher", start: Any = None,
           end: Any = None, holdout: bool = False, params: Mapping[str, Any] | None = None,
           record: bool = True) -> tuple[list[Result], pd.DataFrame]:
    """``dated``: ``[ticker, date]``, the date the event became **public**. Excess return
    after each event against the same tickers on **any** date of a grid (every
    ``baseline_step`` days, while members). With ``placebo_days``, the same events shifted
    back as a third group: a real effect should not appear before the event.

    ⚠️ The baseline excludes nothing by default (``away_days=0``). Excluding the dates near
    each event removes from the baseline exactly the windows that **contain** the event's own
    move — the jump before a big reaction, the fall before an insider buys — and biases it.
    Measured 2026-09-29: excluding ±horizon manufactured +1.6 pp of "drift" after earnings
    reactions that an unconditional baseline puts at +0.08 pp."""
    first, last = _window(lab, start, end, holdout)
    tr = lab.total_return().loc[:last]          # forward windows never cross ``last``
    bench = tr[benchmark]
    future = tr.bfill(limit=FILL)
    ev = dated[(dated["date"] >= first) & (dated["date"] <= last)]
    ev = ev[ev["ticker"].isin(tr.columns)].copy()
    ev["date"] = pd.to_datetime(ev["date"])
    member = lab.members() if members_only else None

    def is_member(tickers: pd.Series, days: pd.Series) -> np.ndarray:
        if member is None:
            return np.ones(len(tickers), dtype=bool)
        pos = member.index.searchsorted(pd.DatetimeIndex(days), side="right") - 1
        cols = member.columns.get_indexer(list(tickers))
        mask = member.to_numpy()
        ok = (pos >= 0) & (cols >= 0)
        out = np.zeros(len(tickers), dtype=bool)
        out[ok] = mask[pos[ok], cols[ok]]
        return out

    grid = pd.date_range(first, last, freq=f"{baseline_step}D")
    rows = []
    for h in horizons:
        h = int(h)
        groups = {"event": ev[["ticker", "date"]]}
        if placebo_days:
            groups["placebo"] = ev[["ticker", "date"]].assign(
                date=ev["date"] - pd.Timedelta(days=placebo_days))
        # Any date of the same tickers (see the docstring: excluding near-event dates biases).
        base = []
        for t, days in ev.groupby("ticker")["date"]:
            d = days.to_numpy(dtype="datetime64[ns]")
            gap = np.abs((grid.to_numpy()[:, None] - d[None, :]).astype("timedelta64[D]")
                         .astype(int)).min(axis=1)
            base.append(pd.DataFrame({"ticker": t, "date": grid[gap > away_days]
                                      if away_days > 0 else grid}))
        groups["base"] = pd.concat(base, ignore_index=True) if base else \
            pd.DataFrame(columns=["ticker", "date"])
        for name, frame_ in groups.items():
            if frame_.empty:
                continue
            values = excess_at(future, bench, frame_["ticker"], frame_["date"], h)
            keep = ~np.isnan(values) & is_member(frame_["ticker"], frame_["date"])
            rows.append(pd.DataFrame({
                "group": name, "horizon": h, "ticker": frame_["ticker"].to_numpy()[keep],
                # the placebo keeps the real event's date, for the halves
                "date": (ev["date"].to_numpy()[keep] if name == "placebo"
                         else frame_["date"].to_numpy()[keep]),
                "value": values[keep]}))
    rows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["group", "horizon", "ticker", "date", "value"])
    frame = rows
    out = []
    sign = 1 if expected == "higher" else -1
    for h in horizons:
        part = frame[frame["horizon"] == int(h)]
        e = part[part["group"] == "event"]
        b = part[part["group"] == "base"]["value"].tolist()
        pl = part[part["group"] == "placebo"]["value"]
        base = float(np.mean(b)) if b else 0.0
        vals = e["value"].tolist()
        out.append(Result(
            rule, "events", int(h), len(vals),
            float(np.mean(vals)) - base if vals and b else None,
            float(np.mean([(v - base) * sign > 0 for v in vals])) if vals and b else None,
            permutation_pvalue(vals, b) if len(vals) >= 3 and len(b) >= 3 else None,
            _halves(e["date"].tolist(), [v - base for v in vals]),
            (str(e["date"].min().date()) if len(e) else None,
             str(e["date"].max().date()) if len(e) else None),
            dict(params or {}),
            (f"base {len(b)} puntos"
             + (f"; placebo {pl.mean() * 100:+.2f} % frente a evento "
                f"{np.mean(vals) * 100:+.2f} %" if len(pl) and vals else ""))))
    if record:
        registry.record(out, holdout=holdout)
    return out, frame


# --- 4. Strategy simulation ---------------------------------------------------------------


@dataclass
class Backtest:
    """A simulated portfolio against the benchmark. **Not** a validation."""

    curve: pd.DataFrame            # [strategy, benchmark] indexed by session, start = 1
    cagr: float | None
    benchmark_cagr: float | None
    volatility: float | None
    max_drawdown: float | None
    benchmark_max_drawdown: float | None
    turnover: float | None         # average share of the portfolio replaced per rebalance
    holdings: pd.DataFrame         # [date, tickers]

    def __str__(self) -> str:
        def pct(x):
            return "—" if x is None else f"{x * 100:+.2f} %"
        return (f"CAGR {pct(self.cagr)} (benchmark {pct(self.benchmark_cagr)}) · volatilidad "
                f"{pct(self.volatility)} · caída máx. {pct(self.max_drawdown)} (benchmark "
                f"{pct(self.benchmark_max_drawdown)}) · rotación {pct(self.turnover)} por "
                "rebalanceo")


def _cagr(curve: pd.Series) -> float | None:
    curve = curve.dropna()
    if len(curve) < 2 or curve.iloc[0] <= 0:
        return None
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    return float((curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1) if years > 0 else None


def _max_dd(curve: pd.Series) -> float | None:
    curve = curve.dropna()
    return float((curve / curve.cummax() - 1).min()) if len(curve) else None


def strategy(lab: Lab, select: Callable[[pd.Timestamp, list[str]], Sequence[str]], *,
             rebalance_days: int = 91, cost_bps: float = 10.0, benchmark: str = BENCHMARK,
             start: Any = None, end: Any = None, holdout: bool = False,
             cash_when_empty: bool = True) -> Backtest:
    """Equal weight in ``select(day, members)`` from the session after each rebalance date
    until the next, total return, minus ``cost_bps`` on the traded fraction. An empty
    selection holds cash at 0 % (``cash_when_empty``). Survivorship: members that later lost
    their prices are missing, which flatters any strategy (§2.24)."""
    tr = lab.total_return()
    first, last = _window(lab, start, end, holdout)
    sessions = tr.index[(tr.index >= first) & (tr.index <= last)]
    dates = grid_dates(sessions, rebalance_days)
    value, path, held_rows, turns = 1.0, {}, [], []
    previous: set[str] = set()
    for i, day in enumerate(dates):
        nxt = dates[i + 1] if i + 1 < len(dates) else sessions[-1]
        chosen = [t for t in select(day, lab.members_on(day)) if t in tr.columns]
        held_rows.append({"date": day, "tickers": chosen})
        new = set(chosen)
        traded = (len(new ^ previous) / max(len(new | previous), 1)) if (new or previous) else 0
        turns.append(traded)
        value *= 1 - traded * cost_bps / 1e4
        previous = new
        window = tr.loc[(tr.index > day) & (tr.index <= nxt)]
        if not chosen or window.empty:
            for d in window.index:
                path[d] = value
            continue
        prices = tr[chosen].loc[(tr.index >= day) & (tr.index <= nxt)].ffill(limit=FILL)
        start_px = prices.iloc[0]
        rel = (prices.iloc[1:] / start_px.where(start_px > 0)).mean(axis=1, skipna=True)
        for d, r in rel.items():
            path[d] = value * float(r) if pd.notna(r) else value
        if len(rel):
            value = path[rel.index[-1]]
    curve = pd.Series(path).sort_index()
    bench = tr[benchmark].reindex(curve.index)
    frame = pd.DataFrame({"strategy": curve / curve.iloc[0] if len(curve) else curve,
                          "benchmark": bench / bench.iloc[0] if len(bench) else bench})
    daily = frame["strategy"].pct_change().dropna()
    return Backtest(frame, _cagr(frame["strategy"]), _cagr(frame["benchmark"]),
                    float(daily.std() * math.sqrt(252)) if len(daily) > 20 else None,
                    _max_dd(frame["strategy"]), _max_dd(frame["benchmark"]),
                    float(np.mean(turns)) if turns else None, pd.DataFrame(held_rows))


# --- 5. Monthly contributions with a brake ------------------------------------------------


def contributions(lab: Lab, brake: pd.Series, *, asset: str = BENCHMARK, start: Any = None,
                  end: Any = None, holdout: bool = False) -> dict[str, Any]:
    """One unit a month into ``asset``; while ``brake`` (1 = engaged) was on at the previous
    close, the money waits in cash at the fed funds rate as published. The panel's tested
    simulator (``validation.brake.simulate_dca``) against always investing."""
    from validation.brake import simulate_dca  # noqa: PLC0415

    tr = lab.total_return([asset])[asset].dropna()
    first, last = _window(lab, start, end, holdout, timing=True)
    rate = lab.fred("DFF")
    braked = simulate_dca(tr, brake.astype(float), rate, first, last)
    always = simulate_dca(tr, None, rate, first, last)
    return {"con_freno": braked, "siempre": always,
            "diferencia": braked.final_wealth / always.final_wealth - 1
            if always.final_wealth else None}
