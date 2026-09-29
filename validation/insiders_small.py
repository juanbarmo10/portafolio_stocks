"""Do insider purchases precede returns above small caps, outside the S&P 500? (§15.5 p. 7)

The S&P 500 study (``validation/insiders``) found the opposite of the literature: purchases
preceded *worse* returns. The literature finds the effect strongest in small companies, which
that study could not test. The protocol lives in ``settings.yaml`` (``insider_small_study``),
committed before any price of these companies was downloaded; this module executes it.

- **Signals**: the S&P study's own definitions (``insider_study.signals``), unchanged.
- **Universe**: not an S&P 500 member on the event date, with a close of at least
  ``min_price`` and a median daily dollar volume of at least ``min_median_dollar_volume``
  over the ``liquidity_window_days`` **before** the date — nothing from the future decides
  who is in.
- **Measure**: excess return over IWM (price returns, split-adjusted) from the first session
  after the filing, as in the S&P study.
- **Baseline**: the same companies on a monthly grid, away from their events, under the same
  eligibility rule.
- **Placebo** (the lesson of §2.61, where a rule was met for the wrong reason): the same
  events moved ``placebo_shift_days`` earlier. If the placebo does as well as the event, the
  excess belongs to the kind of company that has insiders buying, not to the purchase.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from validation.insiders import Result, cluster_events, executive_events, normalize_symbol, run
from validation.metrics import forward_return


def liquidity(prices: Mapping[str, pd.DataFrame], window_days: int) -> dict[str, pd.DataFrame]:
    """``{ticker: [last_close, median_dollar_volume]}`` known **before** each session: the
    previous close and the median of close × volume over the ``window_days`` calendar days
    that end the day before (``closed='left'``)."""
    out = {}
    for ticker, frame in prices.items():
        f = frame.dropna(subset=["close"]).sort_index()
        dollar = (f["close"] * f["volume"]).rolling(f"{window_days}D", closed="left").median()
        out[ticker] = pd.DataFrame({"last_close": f["close"].shift(1), "median_dv": dollar})
    return out


def eligible(ticker: str, day: pd.Timestamp, member: pd.DataFrame,
             liquid: Mapping[str, pd.DataFrame], cfg: Mapping[str, Any]) -> bool:
    """Outside the S&P 500 that day, and liquid enough judged only by the past."""
    if ticker in member.columns:
        on = member.index[member.index <= day]
        if len(on) and bool(member.at[on[-1], ticker]):
            return False
    table = liquid.get(ticker)
    if table is None:
        return False
    known = table[table.index <= day]
    if known.empty:
        return False
    row = known.iloc[-1]
    return (pd.notna(row["last_close"]) and row["last_close"] >= float(cfg["min_price"])
            and pd.notna(row["median_dv"])
            and row["median_dv"] >= float(cfg["min_median_dollar_volume"]))


def excess(points: pd.DataFrame, closes: Mapping[str, pd.Series], bench: pd.Series,
           member: pd.DataFrame, liquid: Mapping[str, pd.DataFrame], cfg: Mapping[str, Any],
           horizons: list[int]) -> pd.DataFrame:
    """``points`` ``[symbol, date, ...]`` that pass :func:`eligible`, with ``excess_{h}``."""
    rows = []
    for row in points.to_dict("records"):
        ticker, day = row["symbol"], pd.Timestamp(row["date"])
        if ticker not in closes or not eligible(ticker, day, member, liquid, cfg):
            continue
        out = dict(row)
        for h in horizons:
            stock, market = forward_return(closes[ticker], day, h), forward_return(bench, day, h)
            out[f"excess_{h}"] = None if stock is None or market is None else stock - market
        rows.append(out)
    return pd.DataFrame(rows)


def baseline_points(events: pd.DataFrame, closes: Mapping[str, pd.Series], *,
                    step_days: int, away_days: int) -> pd.DataFrame:
    """Monthly grid dates over each event company's price history, farther than
    ``away_days`` from any of its events (eligibility is checked afterwards)."""
    rows = []
    for symbol, group in events.groupby("symbol"):
        series = closes.get(symbol)
        if series is None or series.dropna().empty:
            continue
        days = series.dropna().index
        own = pd.to_datetime(group["date"]).to_numpy()
        for day in pd.date_range(days.min(), days.max(), freq=f"{step_days}D"):
            if np.min(np.abs((own - day.to_datetime64()) / np.timedelta64(1, "D"))) > away_days:
                rows.append({"symbol": symbol, "date": day})
    return pd.DataFrame(rows, columns=["symbol", "date"])


def placebo_gaps(events: pd.DataFrame, placebo: pd.DataFrame,
                 horizons: list[int]) -> dict[tuple[str, int], float | None]:
    """``{(signal, h): event mean − placebo mean}``."""
    out = {}
    for signal in sorted(events["signal"].unique()):
        ev, pl = events[events["signal"] == signal], placebo[placebo["signal"] == signal]
        for h in horizons:
            x = pd.to_numeric(ev[f"excess_{h}"], errors="coerce").dropna()
            y = pd.to_numeric(pl[f"excess_{h}"], errors="coerce").dropna() if len(pl) else []
            out[(signal, h)] = float(x.mean() - y.mean()) if len(x) and len(y) else None
    return out


def decision(results: list[Result], gaps: Mapping[tuple[str, int], float | None],
             cfg: Mapping[str, Any]) -> dict[str, bool]:
    """The rule written before running (``insider_small_study``)."""
    out: dict[str, bool] = {}
    for r in results:
        gap = gaps.get((r.signal, r.horizon))
        ok = (r.horizon in [int(h) for h in cfg["decision_horizons"]] and r.significant
              and r.mean is not None and r.base_mean is not None and r.mean > r.base_mean
              and all(v is not None and v > 0 for v in r.halves)
              and gap is not None and gap > 0)
        out[r.signal] = out.get(r.signal, False) or ok
    return out


def study(purchases: pd.DataFrame, prices: Mapping[str, pd.DataFrame], intervals: Any,
          signals_cfg: Mapping[str, Any], cooldown_days: int,
          cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Events → eligibility → excess over IWM → battery, placebo → decision. Pure.

    Args:
        prices: ``{ticker: [close, volume]}``, split-adjusted, benchmark included.
        signals_cfg / cooldown_days: from ``insider_study`` (the same definitions).
    """
    from transform import breadth as br  # noqa: PLC0415

    events = pd.concat([
        cluster_events(purchases, min_insiders=int(signals_cfg["cluster"]["min_insiders"]),
                       window_days=int(signals_cfg["cluster"]["window_days"]),
                       cooldown_days=cooldown_days),
        executive_events(purchases, title_pattern=str(signals_cfg["executive"]["title_pattern"]),
                         min_value_usd=float(signals_cfg["executive"]["min_value_usd"]),
                         cooldown_days=cooldown_days),
    ], ignore_index=True)
    events["symbol"] = events["symbol"].map(normalize_symbol)
    bench = prices[str(cfg["benchmark"])]["close"].dropna()
    closes = {t: f["close"] for t, f in prices.items() if t != str(cfg["benchmark"])}
    liquid = liquidity({t: f for t, f in prices.items() if t != str(cfg["benchmark"])},
                       int(cfg["liquidity_window_days"]))
    index = bench.index
    member = br.membership_mask(intervals, index, sorted(closes))
    horizons = [int(h) for h in cfg["horizons_days"]]

    in_window = events[events["date"] >= pd.Timestamp(cfg["price_start"])
                       + pd.Timedelta(days=int(cfg["liquidity_window_days"]))]
    scored = excess(in_window, closes, bench, member, liquid, cfg, horizons)
    usable = scored.dropna(subset=[f"excess_{horizons[0]}"]) if not scored.empty else scored
    base = excess(baseline_points(usable, closes, step_days=int(cfg["baseline_step_days"]),
                                  away_days=cooldown_days),
                  closes, bench, member, liquid, cfg, horizons)
    shifted = usable[["signal", "symbol", "date"]].assign(
        date=usable["date"] - pd.Timedelta(days=int(cfg["placebo_shift_days"])))
    placebo = excess(shifted, closes, bench, member, liquid, cfg, horizons)
    results = run(usable, base, cfg)
    gaps = placebo_gaps(usable, placebo, horizons)
    unpriced = sorted(set(in_window["symbol"]) - set(closes))
    counts = {
        "purchases": len(purchases),
        "events": in_window.groupby("signal").size().to_dict(),
        "eligible": usable.groupby("signal").size().to_dict(),
        "symbols": int(in_window["symbol"].nunique()),
        "symbols_unpriced": len(unpriced),
        "baseline_points": int(base[f"excess_{horizons[0]}"].notna().sum()) if len(base) else 0,
        "first_date": str(usable["date"].min().date()) if len(usable) else None,
        "last_date": str(usable["date"].max().date()) if len(usable) else None,
    }
    return {"results": results, "gaps": gaps, "decision": decision(results, gaps, cfg),
            "counts": counts, "events": usable}


def report(outcome: Mapping[str, Any], cfg: Mapping[str, Any], generated: str) -> str:
    """The Markdown report, what does not work first (section 8, phase 4)."""
    def pct(x: float | None) -> str:
        return "—" if x is None or pd.isna(x) else f"{x * 100:+.2f} %".replace(".", ",")

    def num(x: float | None, d: int = 3) -> str:
        return "—" if x is None or pd.isna(x) else f"{x:.{d}f}".replace(".", ",")

    counts, results, gaps = outcome["counts"], outcome["results"], outcome["gaps"]
    lines = [f"# Compras de directivos en empresas pequeñas (fuera del S&P 500) ({generated})",
             "", "Protocolo y regla escritos en `settings.yaml` (`insider_small_study`) y "
             "guardados en git antes de bajar ningún precio; una sola ejecución.", "",
             "## Veredicto", ""]
    for signal, ok in outcome["decision"].items():
        lines.append(f"- **{signal}**: " + ("VALIDADA según la regla escrita." if ok else
                                          "**no validada** según la regla escrita."))
    lines += ["", "## Pruebas (exceso sobre IWM: eventos frente a las mismas empresas en días "
              "normales, y frente al placebo)", "",
              "| Señal | Horizonte | n | Exceso medio | Base | Diferencia | Mediana | "
              "% positivos (base) | p | q (BH) | 1.ª mitad | 2.ª mitad | Evento − placebo |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        diff = None if r.mean is None or r.base_mean is None else r.mean - r.base_mean
        lines.append(
            f"| {r.signal} | {r.horizon} d | {r.n} | {pct(r.mean)} | {pct(r.base_mean)} | "
            f"{pct(diff)} | {pct(r.median)} | {num(r.hit_rate, 2)} ({num(r.base_hit_rate, 2)}) "
            f"| {num(r.p)} | {num(r.q)}{' ✓' if r.significant else ''} | {pct(r.halves[0])} | "
            f"{pct(r.halves[1])} | {pct(gaps.get((r.signal, r.horizon)))} |")
    lines += ["", "## Datos", "",
              f"- Compras (Form 4, código P): {counts['purchases']:,}.",
              f"- Eventos en la ventana: {counts['events']}; elegibles (fuera del S&P, "
              f"líquidos antes del evento, con precio): {counts['eligible']} "
              f"({counts['first_date']} → {counts['last_date']}).",
              f"- Valores con eventos: {counts['symbols']}; **sin historia de precios en "
              f"yfinance: {counts['symbols_unpriced']}** (supervivencia, a favor de la señal).",
              f"- Puntos de la base: {counts['baseline_points']:,}.",
              "", "## Límites", "",
              "- **Supervivencia a favor de la señal:** faltan las empresas que dejaron de "
              "cotizar; entre ellas, las pequeñas donde los directivos compraron y luego se "
              "hundieron.",
              "- Rentabilidades de precio, sin dividendos, en los dos lados.",
              "- El código P mezcla compras en bolsa y colocaciones privadas.",
              "- Liquidez con precios ajustados por splits: el producto precio × volumen no "
              "depende del split."]
    return "\n".join(lines)
