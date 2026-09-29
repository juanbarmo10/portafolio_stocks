"""The rules, as functions you can call with any parameters.

Three kinds, one per test of ``lab.evaluate``:

``timing``  ``fn(lab, **params) -> Series`` on the sessions: 1 = "on", 0 = "off", NaN =
            unknown. Judged by ``evaluate.timing`` on SPY (or any asset).
``cross``   ``fn(lab, **params) -> scores(day, members) -> Series`` by ticker: a flag or a
            number, **using only what was known on ``day``**. Judged by
            ``evaluate.cross_section`` inside each date.
``events``  ``fn(lab, **params) -> DataFrame[ticker, date]``, dated when the event became
            public. Judged by ``evaluate.events``.

Every rule the panel has studied is here with the parameters it was tested with (the
defaults), so re-running it reproduces the study's question on the lab's data; the new ones
are marked ``nueva``. ``expected`` states the hypothesis: "lower" (after "on", worse
returns: a brake, a warning) or "higher".

Changing a default and re-testing is a **new test**: the registry counts it (§9.7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from lab.data import BENCHMARK, Lab


@dataclass(frozen=True)
class Rule:
    name: str
    kind: str                      # timing | cross | events
    fn: Callable[..., Any]
    expected: str                  # lower | higher
    description: str
    defaults: dict[str, Any] = field(default_factory=dict)
    mode: str = "flag"             # cross only: flag | top | spread
    origin: str = ""               # where the panel studied it, or "nueva"


RULES: dict[str, Rule] = {}


def rule(name: str, kind: str, expected: str, description: str, *, mode: str = "flag",
         origin: str = "nueva", **defaults: Any):
    def wrap(fn):
        RULES[name] = Rule(name, kind, fn, expected, description, defaults, mode, origin)
        return fn
    return wrap


def _series(values: pd.Series, known: pd.Series) -> pd.Series:
    """1.0 / 0.0 where ``known``, NaN elsewhere."""
    return values.astype(float).where(known)


# --- Timing: the regime light and its parts ------------------------------------------------


@rule("regime_red", "timing", "lower", "Semáforo en rojo (risk-off): el freno de compras.",
      origin="fase 4, RESEARCH §2.29-2.30")
def regime_red(lab: Lab) -> pd.Series:
    from transform import regime as rg  # noqa: PLC0415
    f = lab.regime().frame
    return _series(f["verdict"] == rg.RISK_OFF, f["verdict"] != rg.INSUFFICIENT)


@rule("regime_green", "timing", "higher", "Semáforo en verde (risk-on).",
      origin="fase 4, RESEARCH §2.29")
def regime_green(lab: Lab) -> pd.Series:
    from transform import regime as rg  # noqa: PLC0415
    f = lab.regime().frame
    return _series(f["verdict"] == rg.RISK_ON, f["verdict"] != rg.INSUFFICIENT)


@rule("regime_vote", "timing", "lower",
      "Un componente del semáforo votando risk-off (component: net_liquidity, nfci, credit, "
      "dollar, curve, breadth, equal_weight, volatility).",
      origin="fase 4, RESEARCH §2.29", component="credit", vote=-1)
def regime_vote(lab: Lab, component: str, vote: int) -> pd.Series:
    v = lab.regime().votes[component]["vote"]
    return _series(v == vote, v.notna())


@rule("brake", "timing", "lower",
      "Variantes del freno: V0_actual, V1_sin_rsp, V2_tendencia (Faber), V3_ambos.",
      origin="RESEARCH §2.30", variant="V0_actual")
def brake(lab: Lab, variant: str) -> pd.Series:
    from validation import brake as bk  # noqa: PLC0415
    return bk.variants(lab.regime())[variant]


@rule("trend_below_sma", "timing", "lower",
      "El activo cierra por debajo de su media de N sesiones (tendencia de Faber).",
      origin="RESEARCH §2.30 (V2)", asset=BENCHMARK, window=200)
def trend_below_sma(lab: Lab, asset: str, window: int) -> pd.Series:
    px = lab.prices([asset])[asset].dropna()
    sma = px.rolling(window, min_periods=int(window * 0.95)).mean()
    return _series(px < sma, sma.notna())


@rule("vix_inverted", "timing", "lower", "VIX por encima del VIX3M: estrés a corto plazo.",
      origin="componente del semáforo", ratio=1.0)
def vix_inverted(lab: Lab, ratio: float) -> pd.Series:
    r = lab.fred("VIXCLS") / lab.fred("VXVCLS")
    return _series(r > ratio, r.notna())


@rule("curve_inverted", "timing", "lower", "Curva 10a − 2a por debajo de cero.",
      origin="componente del semáforo", level=0.0)
def curve_inverted(lab: Lab, level: float) -> pd.Series:
    c = lab.fred("T10Y2Y")
    return _series(c < level, c.notna())


@rule("credit_widening", "timing", "lower",
      "Prima de crédito Baa − 10a por encima de su media de N sesiones.",
      origin="componente del semáforo", window=200)
def credit_widening(lab: Lab, window: int) -> pd.Series:
    c = lab.fred("BAA10Y")
    m = c.rolling(window, min_periods=int(window * 0.95)).mean()
    return _series(c > m, m.notna())


@rule("oil_shock", "timing", "lower",
      "El petróleo (WTI) sube al menos X en N sesiones (nueva; datos desde 2011).",
      days=63, rise=0.30)
def oil_shock(lab: Lab, days: int, rise: float) -> pd.Series:
    w = lab.fred("DCOILWTICO")
    prev = w.shift(days)
    change = w / prev.where(prev > 0) - 1
    return _series(change >= rise, change.notna())


@rule("real_rate_rising", "timing", "lower",
      "El tipo real a 10 años sube al menos X puntos en N sesiones (nueva).",
      days=63, rise=0.5)
def real_rate_rising(lab: Lab, days: int, rise: float) -> pd.Series:
    r = lab.fred("DFII10")
    change = r - r.shift(days)
    return _series(change >= rise, change.notna())


@rule("market_drawdown", "timing", "higher",
      "El activo está al menos X por debajo de su máximo de N sesiones: ¿comprar la caída? "
      "(nueva)", asset=BENCHMARK, depth=0.10, window=252)
def market_drawdown(lab: Lab, asset: str, depth: float, window: int) -> pd.Series:
    px = lab.total_return([asset])[asset].dropna()
    high = px.rolling(window, min_periods=int(window * 0.9)).max()
    return _series(px / high - 1 <= -depth, high.notna())


# --- Cross-section: per stock, inside each date --------------------------------------------


def _at(frame: pd.DataFrame, day: pd.Timestamp) -> pd.Series | None:
    rows = frame.index[frame.index <= day]
    return frame.loc[rows[-1]] if len(rows) else None


def _return_over(frame: pd.DataFrame, day: pd.Timestamp, days: int, skip: int = 0
                 ) -> pd.Series:
    past = frame.ffill(limit=5)
    end = _at(past, day - pd.Timedelta(days=skip))
    begin = _at(past, day - pd.Timedelta(days=days))
    if end is None or begin is None:
        return pd.Series(dtype=float)
    return (end / begin.where(begin > 0) - 1).dropna()


@rule("knife", "cross", "lower",
      "Cuchillo cayendo: bajo su media de N sesiones y con X o peor en D días.",
      origin="RESEARCH §2.65", sma=200, days=91, max_return=-0.20)
def knife(lab: Lab, sma: int, days: int, max_return: float):
    px = lab.prices()
    average = px.rolling(sma, min_periods=int(sma * 0.95)).mean()

    def scores(day, members):
        now, avg = _at(px[members], day), _at(average[members], day)
        ret = _return_over(px[members], day, days)
        if now is None or avg is None:
            return None
        ok = avg.notna() & now.notna()
        flag = (now < avg) & (ret.reindex(now.index) <= max_return)
        return flag[ok & ret.reindex(now.index).notna()]
    return scores


@rule("overextended", "cross", "lower",
      "Muy estirada: al menos X veces su media de N sesiones.",
      origin="RESEARCH §2.65", sma=200, ratio=1.30)
def overextended(lab: Lab, sma: int, ratio: float):
    px = lab.prices()
    average = px.rolling(sma, min_periods=int(sma * 0.95)).mean()

    def scores(day, members):
        now, avg = _at(px[members], day), _at(average[members], day)
        if now is None or avg is None:
            return None
        ok = avg.notna() & now.notna()
        return (now >= ratio * avg)[ok]
    return scores


@rule("momentum", "cross", "higher",
      "Momentum: rentabilidad de lookback a skip días antes (12-1 por defecto).",
      mode="top", origin="RESEARCH §2.49", lookback=365, skip=30)
def momentum(lab: Lab, lookback: int, skip: int):
    tr = lab.total_return()
    return lambda day, members: _return_over(tr[members], day, lookback, skip)


@rule("reversal", "cross", "higher",
      "Reversión a corto: lo que más cayó el último mes (puntuación = −rentabilidad).",
      mode="top", days=30)
def reversal(lab: Lab, days: int):
    tr = lab.total_return()
    return lambda day, members: -_return_over(tr[members], day, days)


@rule("low_volatility", "cross", "higher",
      "Baja volatilidad: la menor volatilidad diaria de N días (puntuación = −volatilidad).",
      mode="top", days=365)
def low_volatility(lab: Lab, days: int):
    tr = lab.total_return()

    def scores(day, members):
        window = tr[members].loc[(tr.index > day - pd.Timedelta(days=days)) & (tr.index <= day)]
        vol = window.pct_change(fill_method=None).std()
        return -vol[window.notna().sum() >= 0.9 * len(window)]
    return scores


@rule("near_high", "cross", "higher",
      "Cerca de su máximo de N días (precio / máximo).", mode="top", days=365)
def near_high(lab: Lab, days: int):
    px = lab.prices()

    def scores(day, members):
        window = px[members].loc[(px.index > day - pd.Timedelta(days=days)) & (px.index <= day)]
        if window.empty:
            return None
        return (window.ffill().iloc[-1] / window.max()).dropna()
    return scores


def _by_ticker(lab: Lab, by_cik: pd.Series, members: list[str]) -> pd.Series:
    cik_of = lab.cik_of()
    out, seen = {}, set()
    for t in members:
        cik = cik_of.get(t)
        if cik in by_cik.index and cik not in seen:     # one ticker per company
            out[t] = float(by_cik[cik])
            seen.add(cik)
    return pd.Series(out, dtype=float)


@rule("gross_profitability", "cross", "higher",
      "Calidad: beneficio bruto / activos del último ejercicio presentado.",
      mode="top", origin="RESEARCH §2.49", max_age_days=500)
def gross_profitability(lab: Lab, max_age_days: int):
    from validation import factors  # noqa: PLC0415
    records = lab.facts()
    return lambda day, members: _by_ticker(
        lab, factors.gross_profitability(records, day, max_age_days), members)


@rule("low_accruals", "cross", "higher",
      "Devengos bajos: (beneficio − flujo operativo) / activos (puntuación = −devengos).",
      mode="top", origin="RESEARCH §2.49", max_age_days=500)
def low_accruals(lab: Lab, max_age_days: int):
    from validation import factors  # noqa: PLC0415
    records = lab.facts()
    return lambda day, members: -_by_ticker(
        lab, factors.accruals(records, day, max_age_days), members)


# --- Events -------------------------------------------------------------------------------


@rule("insider_cluster", "events", "higher",
      "Varios directivos distintos comprando en N días (fecha: la presentación que completa "
      "el grupo).", origin="RESEARCH §2.42", min_insiders=3, window_days=30,
      cooldown_days=180)
def insider_cluster(lab: Lab, min_insiders: int, window_days: int, cooldown_days: int):
    from validation import insiders  # noqa: PLC0415
    ev = insiders.cluster_events(lab.insider_purchases(), min_insiders=min_insiders,
                                 window_days=window_days, cooldown_days=cooldown_days)
    return _events_by_ticker(lab, ev)


@rule("insider_executive", "events", "higher",
      "El CEO o el CFO compra al menos X USD.", origin="RESEARCH §2.42",
      min_value_usd=100_000, cooldown_days=180)
def insider_executive(lab: Lab, min_value_usd: float, cooldown_days: int):
    from validation import insiders  # noqa: PLC0415
    pattern = lab.settings.raw["insider_study"]["signals"]["executive"]["title_pattern"]
    ev = insiders.executive_events(lab.insider_purchases(), title_pattern=pattern,
                                   min_value_usd=min_value_usd, cooldown_days=cooldown_days)
    return _events_by_ticker(lab, ev)


def _events_by_ticker(lab: Lab, ev: pd.DataFrame) -> pd.DataFrame:
    """Insider events carry the issuer's CIK: map it to the member's ticker (§9.3)."""
    ticker_of: dict[str, str] = {}
    for t, cik in lab.cik_of().items():
        ticker_of.setdefault(cik, t)
    out = ev.assign(ticker=ev["issuer_cik"].astype(str).str.zfill(10).map(ticker_of))
    return out.dropna(subset=["ticker"])[["ticker", "date"]].reset_index(drop=True)


@rule("earnings_reaction", "events", "higher",
      "Reacción a resultados (8-K 2.02) de al menos ±X sobre SPY en la ventana; la deriva se "
      "mide desde el final de la ventana. side: up | down.",
      origin="RESEARCH §2.61", side="up", threshold=0.05, before=-1, after=1)
def earnings_reaction(lab: Lab, side: str, threshold: float, before: int, after: int):
    from validation.pead import reaction  # noqa: PLC0415
    px, spy = lab.prices(), lab.prices([BENCHMARK])[BENCHMARK]
    rows = []
    dates = lab.earnings_dates()
    for t, d in zip(dates["ticker"], dates["date"]):
        if t not in px.columns:
            continue
        move, last = reaction(px[t], spy, pd.Timestamp(d), (before, after))
        if move is None or last is None:
            continue
        if (side == "up" and move >= threshold) or (side == "down" and move <= -threshold):
            rows.append({"ticker": t, "date": last})
    return pd.DataFrame(rows, columns=["ticker", "date"])


# --- Helpers for your own rules -----------------------------------------------------------


def build(lab: Lab, name: str, **params: Any):
    """The rule ``name`` with its defaults overridden by ``params``: ``(output, params)``."""
    spec = RULES[name]
    merged = {**spec.defaults, **params}
    return spec.fn(lab, **merged), merged


def catalogue() -> pd.DataFrame:
    """Every rule: kind, hypothesis, defaults, origin."""
    return pd.DataFrame([{"regla": r.name, "tipo": r.kind, "hipótesis": r.expected,
                          "modo": r.mode if r.kind == "cross" else "", "por defecto":
                          r.defaults, "origen": r.origin, "qué es": r.description}
                         for r in RULES.values()])

