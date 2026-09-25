"""Variants of the purchase brake, judged by what a brake is for (RESEARCH.md §2.30).

Phase 4 asked whether the regime light *predicts returns*, and it does not. But section 2
never promised that: the light is a **brake on contributions**. So this module asks the two
questions that decide whether a brake is worth having, for a monthly contributor:

1. **Protection.** After the brake engages, does the market fall further than usual? Measured
   as the forward *maximum drawdown* from the next session's close — the pain a purchase made
   that day would have gone through. A useful brake engages before deeper drawdowns.
2. **Cost, in the user's own workflow.** Every month a contribution arrives. With the brake
   engaged it waits in cash, earning the Fed funds rate; as soon as the brake releases, all
   the waiting cash is invested. Compared with contributing every month regardless, what is
   the final wealth?

**Written before running, and not changed after (2026-09-24).** The variants are few and
each carries its reason; the one whose reason came from looking at data says so:

- ``V0_actual``: the light as it is — red = majority of the eight components risk-off.
- ``V1_sin_rsp``: the same without RSP/SPY. **Data-snooped**: it is proposed because
  RSP/SPY failed in and out of sample. Its result is weaker evidence than the others'.
- ``V2_tendencia``: the S&P 500 below its 200-session average. The classic trend filter of
  the literature (Faber, 2007), chosen from outside this project's data. Price only, so it
  can be judged from 1999 — through 2000-2002 and 2008, out of sample for everything here.
- ``V3_ambos``: V0 **and** V2 — the light's red only counts when price confirms it.

"Better" is decided by rule, fixed here: a variant improves on V0 if it protects at least as
well **and** costs less, in **both halves** of the sample. Nothing in production changes by
this module; it reports, the user decides.

No look-ahead: the brake decided at a session's close is applied to the **next** session's
purchase. Pure functions; no network, no database.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from transform import regime as rg

VARIANT_LABELS = {
    "V0_actual": "V0 — semáforo actual (8 componentes)",
    "V1_sin_rsp": "V1 — sin RSP/SPY (propuesta tras ver datos)",
    "V2_tendencia": "V2 — S&P 500 bajo su media de 200 sesiones",
    "V3_ambos": "V3 — semáforo rojo Y precio bajo su media",
}


def price_trend_brake(prices: pd.Series, window: int = 200,
                      min_window_fraction: float = 0.95) -> pd.Series:
    """1.0 where the price closes below its trailing average, 0.0 above, NaN without one."""
    periods = max(1, math.ceil(window * min_window_fraction))
    average = prices.rolling(window, min_periods=periods).mean()
    return (prices < average).astype(float).where(average.notna())


def variants(built: rg.RegimeBuild, *, drop: str = "equal_weight", window: int = 200,
             min_window_fraction: float = 0.95, min_components: int = 4
             ) -> dict[str, pd.Series]:
    """Each variant as a brake series on the sessions: 1.0 engaged, 0.0 released, NaN
    where it cannot decide (the light without enough components)."""
    frame = built.frame
    judged = frame["verdict"] != rg.INSUFFICIENT
    v0 = (frame["verdict"] == rg.RISK_OFF).astype(float).where(judged)

    without = {k: v for k, v in built.votes.items() if k != drop}
    frame1 = rg.regime_frame(without, min_components=min_components)
    judged1 = frame1["verdict"] != rg.INSUFFICIENT
    v1 = (frame1["verdict"] == rg.RISK_OFF).astype(float).where(judged1)

    trend = price_trend_brake(built.benchmark.dropna(), window, min_window_fraction)
    v3 = (v0.reindex(trend.index) * trend).where(v0.reindex(trend.index).notna() & trend.notna())
    return {"V0_actual": v0, "V1_sin_rsp": v1, "V2_tendencia": trend, "V3_ambos": v3}


# --- Protection -------------------------------------------------------------------------


def forward_drawdown(prices: pd.Series, at: pd.Timestamp, horizon_days: int) -> float | None:
    """Worst fall below the entry within ``horizon_days``, entering the session after ``at``.

    ``0`` if the price never went below the entry; ``None`` if the window is not complete.
    """
    series = prices.dropna()
    after = series.loc[series.index > at]
    if after.empty or after.iloc[0] <= 0:
        return None
    entry_day, entry = after.index[0], float(after.iloc[0])
    end = entry_day + pd.Timedelta(days=horizon_days)
    if series.index[-1] < end:
        return None
    window = series.loc[entry_day:end]
    return min(0.0, float(window.min()) / entry - 1.0)


# --- Cost, in the monthly-contribution workflow -----------------------------------------


@dataclass(frozen=True)
class DcaResult:
    final_wealth: float           # per unit contributed
    contributions: int
    months_deferred: int          # contributions that did not go in on their own month
    mean_days_waiting: float      # average days a contribution sat in cash


def simulate_dca(
    prices: pd.Series,
    brake: pd.Series | None,
    cash_rate: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
) -> DcaResult:
    """One unit contributed on the first session of each month from ``start``.

    ``prices`` is a total-return index (dividends reinvested, section 9.1). While the brake
    engaged at the **previous** close is 1, cash waits and accrues ``cash_rate`` (annual
    percent, day count 365); when it releases, everything waiting is invested at that close.
    ``brake=None`` is the always-invest baseline. An undecided brake (NaN) does not block:
    not knowing is not a reason to wait.
    """
    full = prices.dropna()
    px = full.loc[start:end] if end is not None else full.loc[start:]
    days = px.index
    price = px.to_numpy(dtype=float)
    # Shifted on the whole calendar, THEN cut: the first session of a simulation still sees
    # the brake of the close before it. Cutting first would leave that close out and let
    # every simulation's first contribution through unbraked.
    engaged = (np.zeros(len(px)) if brake is None
               else brake.reindex(full.index).shift(1).reindex(days).fillna(0.0).to_numpy())
    rate = cash_rate.reindex(days).ffill().fillna(0.0).to_numpy(dtype=float)
    gaps = np.diff(days.values).astype("timedelta64[D]").astype(int)
    period = days.to_period("M")
    is_contribution = np.r_[True, period[1:] != period[:-1]]

    cash, units, contributions, deferred = 0.0, 0.0, 0, 0
    waiting: list[int] = []
    waits: list[int] = []
    for i in range(len(price)):
        if i and cash > 0:
            cash *= 1 + rate[i] / 100 * gaps[i - 1] / 365
        if is_contribution[i]:
            cash += 1.0
            contributions += 1
            waiting.append(i)
        if cash > 0 and engaged[i] != 1:
            units += cash / price[i]
            cash = 0.0
            waits.extend(int((days[i] - days[j]).days) for j in waiting)
            deferred += sum(1 for j in waiting if j != i)
            waiting = []
    final = units * price[-1] + cash
    return DcaResult(
        final_wealth=final / contributions if contributions else float("nan"),
        contributions=contributions, months_deferred=deferred + len(waiting),
        mean_days_waiting=float(np.mean(waits)) if waits else 0.0,
    )


def dca_comparison(
    prices: pd.Series, brakes: Mapping[str, pd.Series], cash_rate: pd.Series,
    *, first_start: pd.Timestamp, last_start: pd.Timestamp, end: pd.Timestamp | None = None,
) -> dict[str, dict[str, float]]:
    """Relative final wealth of each brake against always-investing, over every start month
    in ``[first_start, last_start]``: median, 10th percentile, and share of starts where the
    brake ends ahead. The start month is arbitrary, so one start would be an anecdote."""
    sessions = pd.Series(prices.dropna().loc[first_start:last_start].index)
    firsts = sessions.groupby(sessions.dt.to_period("M")).first().tolist()
    baseline = {start: simulate_dca(prices, None, cash_rate, start, end).final_wealth
                for start in firsts}

    out: dict[str, dict[str, float]] = {}
    for name, brake in brakes.items():
        rel, deferred = [], []
        for start in firsts:
            result = simulate_dca(prices, brake, cash_rate, start, end)
            rel.append(result.final_wealth / baseline[start] - 1)
            deferred.append(result.months_deferred / result.contributions)
        arr = np.asarray(rel)
        out[name] = {
            "starts": len(arr),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "share_ahead": float((arr > 0).mean()),
            "deferred_share": float(np.mean(deferred)),
        }
    return out


# --- The study, as pre-registered in config (brake_study) -------------------------------


def run_study(built: rg.RegimeBuild, cash_rate: pd.Series, cfg: Mapping) -> dict:
    """Protection and cost of every variant, and the verdict of the pre-written rule."""
    from validation.backtest import LOWER, Signal, benjamini_hochberg, test_signal  # noqa: PLC0415
    from validation.metrics import forward_return  # noqa: PLC0415

    brakes = variants(built, window=int(cfg.get("trend_window", 200)))
    brakes = {k: v for k, v in brakes.items() if k in cfg.get("variants", brakes)}
    prices = built.benchmark_total.dropna()
    horizons = [int(h) for h in cfg.get("horizons_days", [30, 90, 180])]

    tests = []
    for key, series in brakes.items():
        signal = Signal(key, VARIANT_LABELS[key], LOWER, series)
        for h in horizons:
            tests.append(("caída máxima", test_signal(signal, prices, h, outcome=forward_drawdown)))
            tests.append(("rentabilidad", test_signal(signal, prices, h, outcome=forward_return)))
    testable = [t for _, t in tests if t.status == "ok"]
    for t, (q, sig) in zip(testable, benjamini_hochberg([t.pvalue for t in testable], 0.10)):
        t.qvalue, t.significant = q, sig

    months = int(cfg.get("min_months_to_end", 36))

    def dca(period: list[str], which: Mapping[str, pd.Series]) -> dict:
        start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
        return dca_comparison(prices, which, cash_rate, first_start=start,
                              last_start=end - pd.DateOffset(months=months), end=end)

    halves = [dca(h, brakes) for h in cfg["halves"]]
    long = dca(cfg["long_period"], {"V2_tendencia": brakes["V2_tendencia"]}) \
        if "long_period" in cfg and "V2_tendencia" in brakes else {}

    horizon = int(cfg.get("protection_horizon", 90))
    protection = {
        key: next((t.edge for kind, t in tests if kind == "caída máxima"
                   and t.signal == key and t.horizon == horizon), None)
        for key in brakes
    }
    verdicts = {}
    for key in brakes:
        if key == "V0_actual":
            continue
        protects = (protection[key] is not None and protection["V0_actual"] is not None
                    and protection[key] <= protection["V0_actual"])
        cheaper = all(h[key]["median"] > h["V0_actual"]["median"] for h in halves)
        verdicts[key] = {"protects": protects, "cheaper_both_halves": cheaper,
                         "improves": protects and cheaper}
    return {"tests": tests, "halves": halves, "long": long, "protection": protection,
            "verdicts": verdicts, "cfg": cfg, "brakes": brakes}


def _p(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:+.1f} %".replace(".", ",")


def report_study(study: dict, generated: str) -> str:
    cfg = study["cfg"]
    lines = [f"# Estudio del freno de aportes — {generated}", "",
             "## 1. Protección y coste por señal (rejilla disjunta, permutación, BH q = 0,10)",
             "", "Ventaja = media tras freno activado − media sin freno. En caída máxima, "
             "negativa = el freno se activa antes de caídas más hondas (protege).", "",
             "| Variante | Medida | h | n freno | n base | Media freno | Media base | Ventaja "
             "| p | q |", "|---|---|---|---|---|---|---|---|---|---|"]
    for kind, t in study["tests"]:
        mark = " ✅" if t.significant else ""
        lines.append(f"| {t.label} | {kind} | {t.horizon} | {t.n_signal} | {t.n_baseline} | "
                     f"{_p(t.mean_signal)} | {_p(t.mean_baseline)} | {_p(t.edge)} | "
                     f"{'—' if t.pvalue is None else f'{t.pvalue:.3f}'} | "
                     f"{'—' if t.qvalue is None else f'{t.qvalue:.3f}'}{mark} |")
    lines += ["", "## 2. Coste en el aporte mensual (riqueza final frente a aportar siempre)", "",
              "Efectivo retenido remunerado al tipo de la Fed (DFF). Una simulación por mes de "
              f"arranque, con al menos {cfg.get('min_months_to_end', 36)} meses hasta el final "
              "del periodo.", ""]
    periods = [*cfg["halves"]] + ([cfg["long_period"]] if study["long"] else [])
    for period, table in zip(periods, [*study["halves"], study["long"]] if study["long"]
                             else study["halves"]):
        lines += [f"**{period[0]} → {period[1]}**", "",
                  "| Variante | Arranques | Mediana | Percentil 10 | Acaba por delante | "
                  "Aportes retrasados |", "|---|---|---|---|---|---|"]
        for key, row in table.items():
            lines.append(f"| {VARIANT_LABELS[key]} | {row['starts']} | {_p(row['median'])} | "
                         f"{_p(row['p10'])} | {row['share_ahead']:.0%} | "
                         f"{row['deferred_share']:.0%} |")
        lines.append("")
    lines += ["## 3. Veredicto de la regla escrita antes", "",
              f"Mejora = protege al menos como V0 (caída máxima a "
              f"{cfg.get('protection_horizon', 90)} d) **y** cuesta menos en las dos mitades.", "",
              "| Variante | Protege ≥ V0 | Cuesta menos en ambas mitades | ¿Mejora? |",
              "|---|---|---|---|"]
    for key, v in study["verdicts"].items():
        lines.append(f"| {VARIANT_LABELS[key]} | {'sí' if v['protects'] else 'no'} | "
                     f"{'sí' if v['cheaper_both_halves'] else 'no'} | "
                     f"**{'sí' if v['improves'] else 'no'}** |")
    return "\n".join(lines)
