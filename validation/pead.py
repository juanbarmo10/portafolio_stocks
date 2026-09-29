"""Does a stock keep moving in the direction of its earnings reaction? (§15.5, point 6)

Post-earnings announcement drift (PEAD; Ball and Brown, 1968; Bernard and Thomas, 1989) is
one of the most documented anomalies — measured with the surprise against analysts'
consensus, which is paid data here (§4.5). This study measures the surprise by the market's
own **reaction** and asks whether it keeps going. The protocol and the decision rule live in
``settings.yaml`` (``pead_study``), written and committed before any data was downloaded;
this module only executes them.

**Events.** 8-K filings with item 2.02 of S&P 500 members, dated by the filing date, from
the full submission history of each company (``earnings_dates_from``).

**Reaction** (the signal): the stock's return minus SPY's from the close of the session
before the filing date to the close of the session after it — three sessions, enough for an
announcement before the open or after the close.

**Drift** (the measure): excess return over SPY from the close of the first session after
the reaction window to the first close ``h`` calendar days later (``metrics.forward_return``).
Entering after the reaction is known is what makes it tradable; entering earlier would
count the reaction as drift.

**Baseline**, **test** and **limits** as in ``validation/insiders``: the same companies on a
monthly grid away from their announcements; two-sided permutation of the difference in
means, Benjamini-Hochberg over the battery, the sign in each half.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

from validation.insiders import baseline_points, excess_returns
from validation.metrics import benjamini_hochberg, forward_return, permutation_pvalue


def earnings_dates_from(pages: Iterable[Mapping[str, Any]], item: str = "2.02") -> list[str]:
    """Filing dates of 8-Ks carrying ``item``, from submission pages — the ``recent`` block
    of ``submissions`` and each older ``files`` page, which share the column layout."""
    dates = set()
    for page in pages:
        forms, items, filed = page.get("form", []), page.get("items", []), \
            page.get("filingDate", [])
        for form, its, day in zip(forms, items, filed):
            if (form or "").upper() == "8-K" and item in [i.strip() for i in (its or "").split(",")]:
                dates.add(day)
    return sorted(dates)


def reaction(prices: pd.Series, spy: pd.Series, day: pd.Timestamp,
             window: tuple[int, int] = (-1, 1)) -> tuple[float | None, pd.Timestamp | None]:
    """``(excess return over the window, its last session)``. Session 0 is the first session
    on or after ``day``; the window is in sessions of the stock. ``(None, None)`` when a
    close is missing."""
    series = prices.dropna()
    sessions = series.index
    on_or_after = sessions[sessions >= day]
    if not len(on_or_after):
        return None, None
    zero = sessions.get_loc(on_or_after[0])
    start, end = zero + window[0], zero + window[1]
    if start < 0 or end >= len(sessions):
        return None, None
    d0, d1 = sessions[start], sessions[end]
    bench = spy.dropna()
    b0, b1 = bench[bench.index <= d0], bench[bench.index <= d1]
    if b0.empty or b1.empty or series.iloc[start] <= 0 or b0.iloc[-1] <= 0:
        return None, None
    stock = float(series.iloc[end]) / float(series.iloc[start]) - 1
    market = float(b1.iloc[-1]) / float(b0.iloc[-1]) - 1
    return stock - market, d1


def score_events(events: pd.DataFrame, prices: pd.DataFrame, spy: pd.Series,
                 member: pd.DataFrame, window: tuple[int, int],
                 horizons: list[int]) -> pd.DataFrame:
    """``events`` ``[symbol, date]`` → plus ``reaction`` and ``drift_{h}`` (excess over SPY
    from the end of the reaction window). Only while the company was a member."""
    rows = []
    for row in events.itertuples():
        ticker, day = row.symbol, pd.Timestamp(row.date)
        if ticker not in prices.columns or ticker not in member.columns:
            continue
        on = member.index[member.index <= day]
        if not len(on) or not bool(member.at[on[-1], ticker]):
            continue
        value, end = reaction(prices[ticker], spy, day, window)
        if value is None:
            continue
        out = {"symbol": ticker, "date": day, "reaction": value, "entry": end}
        for h in horizons:
            stock, market = forward_return(prices[ticker], end, h), forward_return(spy, end, h)
            out[f"excess_{h}"] = None if stock is None or market is None else stock - market
        rows.append(out)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Result:
    signal: str
    horizon: int
    n: int
    n_base: int
    mean: float | None
    base_mean: float | None
    median: float | None
    hit_rate: float | None
    p: float | None
    q: float | None = None
    significant: bool = False
    halves: tuple[float | None, float | None] = (None, None)


def signal_events(scored: pd.DataFrame, signals: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """The events of each signal, by the fixed thresholds of the protocol."""
    out = {}
    if "positive" in signals:
        out["positive"] = scored[scored["reaction"] >= float(signals["positive"]["min_reaction"])]
    if "negative" in signals:
        out["negative"] = scored[scored["reaction"] <= float(signals["negative"]["max_reaction"])]
    return out


def run(groups: Mapping[str, pd.DataFrame], base: pd.DataFrame,
        cfg: Mapping[str, Any]) -> list[Result]:
    """Every signal × horizon against the baseline of the same companies; BH across all."""
    split = pd.Timestamp(cfg["halves_split"])
    raw: list[Result] = []
    for signal, ev in groups.items():
        bs = base[base["symbol"].isin(set(ev["symbol"]))]
        for h in [int(x) for x in cfg["horizons_days"]]:
            x = pd.to_numeric(ev[f"excess_{h}"], errors="coerce").dropna()
            b = pd.to_numeric(bs[f"excess_{h}"], errors="coerce").dropna()
            p = permutation_pvalue(x, b, n=int(cfg["permutations"]), seed=int(cfg["seed"])) \
                if len(x) and len(b) else None
            halves = []
            for side_ev, side_b in ((ev["date"] <= split, bs["date"] <= split),
                                    (ev["date"] > split, bs["date"] > split)):
                xs = pd.to_numeric(ev.loc[side_ev, f"excess_{h}"], errors="coerce").dropna()
                bh = pd.to_numeric(bs.loc[side_b, f"excess_{h}"], errors="coerce").dropna()
                halves.append(float(xs.mean() - bh.mean()) if len(xs) and len(bh) else None)
            raw.append(Result(signal=signal, horizon=h, n=len(x), n_base=len(b),
                              mean=float(x.mean()) if len(x) else None,
                              base_mean=float(b.mean()) if len(b) else None,
                              median=float(x.median()) if len(x) else None,
                              hit_rate=float((x > 0).mean()) if len(x) else None,
                              p=p, halves=(halves[0], halves[1])))
    tested = [r for r in raw if r.p is not None]
    adjusted = benjamini_hochberg([r.p for r in tested], float(cfg["fdr_alpha"]))
    qmap = {id(r): a for r, a in zip(tested, adjusted)}
    return [Result(**{**r.__dict__, "q": qmap[id(r)][0], "significant": qmap[id(r)][1]})
            if id(r) in qmap else r for r in raw]


def decision(results: list[Result], cfg: Mapping[str, Any]) -> dict[str, bool]:
    """The rule written before running: significant after BH at a decision horizon, the
    drift in the direction of the reaction against the baseline, and in both halves."""
    out: dict[str, bool] = {}
    for r in results:
        sign = 1 if r.signal == "positive" else -1
        ok = (r.horizon in [int(h) for h in cfg["decision_horizons"]] and r.significant
              and r.mean is not None and r.base_mean is not None
              and sign * (r.mean - r.base_mean) > 0
              and all(v is not None and sign * v > 0 for v in r.halves))
        out[r.signal] = out.get(r.signal, False) or ok
    return out


def study(dates_by_symbol: Mapping[str, list[str]], closes: pd.DataFrame, actions: Any,
          intervals: Any, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Events → reaction and drift → battery → decision. Pure: every input is passed in.

    Args:
        dates_by_symbol: ``{ticker: [8-K 2.02 filing dates]}``.
        closes: Raw closes of the members and SPY, long format.
        actions: ``corporate_actions`` rows, for the split adjustment.
        intervals: ``universe_membership`` rows.
    """
    from transform import breadth as br  # noqa: PLC0415

    wide = br.adjusted_closes(br.wide_closes(closes), br.splits_by_ticker(actions))
    spy = wide.pop(str(cfg["benchmark"])).dropna()
    member = br.membership_mask(intervals, wide.index, list(wide.columns))
    member = member.reindex(columns=wide.columns, fill_value=False)
    first = max(pd.Timestamp(cfg["first_date"]), wide.index.min())
    events = pd.DataFrame([{"symbol": s, "date": pd.Timestamp(d)}
                           for s, days in dates_by_symbol.items() for d in days
                           if pd.Timestamp(d) >= first])
    horizons = [int(h) for h in cfg["horizons_days"]]
    window = tuple(int(x) for x in cfg["reaction_window"])
    scored = score_events(events, wide, spy, member, window, horizons)
    usable = scored.dropna(subset=[f"excess_{horizons[0]}"]) if not scored.empty else scored
    base = excess_returns(wide, spy, member,
                          baseline_points(usable, member, step_days=int(cfg["baseline_step_days"]),
                                          away_days=int(cfg["away_days"])), horizons)
    groups = signal_events(usable, cfg["signals"])
    results = run(groups, base, cfg)
    counts = {
        "companies": len(dates_by_symbol),
        "announcements": int(len(events)),
        "scored": int(len(usable)),
        "by_signal": {k: int(len(v)) for k, v in groups.items()},
        "baseline_points": int(base[f"excess_{horizons[0]}"].notna().sum()),
        "reaction_median_abs": float(usable["reaction"].abs().median()) if len(usable) else None,
        "first_date": str(usable["date"].min().date()) if len(usable) else None,
        "last_date": str(usable["date"].max().date()) if len(usable) else None,
    }
    return {"results": results, "decision": decision(results, cfg), "counts": counts,
            "events": usable}


def report(outcome: Mapping[str, Any], cfg: Mapping[str, Any], generated: str) -> str:
    """The Markdown report, what does not work first (section 8, phase 4)."""
    def pct(x: float | None) -> str:
        return "—" if x is None or pd.isna(x) else f"{x * 100:+.2f} %".replace(".", ",")

    def num(x: float | None, d: int = 3) -> str:
        return "—" if x is None or pd.isna(x) else f"{x:.{d}f}".replace(".", ",")

    counts, results = outcome["counts"], outcome["results"]
    lines = [f"# Deriva tras resultados (PEAD) en el S&P 500 ({generated})", "",
             "Protocolo y regla de decisión escritos en `settings.yaml` (`pead_study`) y "
             "guardados en git antes de descargar datos; una sola ejecución.", "",
             "## Veredicto", ""]
    for signal, ok in outcome["decision"].items():
        lines.append(f"- **{signal}**: " + ("VALIDADA según la regla escrita." if ok else
                                          "**no validada** según la regla escrita."))
    lines += ["", "## Pruebas (deriva = exceso sobre SPY desde el cierre tras la ventana de "
              "reacción; frente a las mismas empresas en días normales)", "",
              "| Señal | Horizonte | n | Deriva media | Base | Diferencia | Mediana | "
              "% positivos | p | q (BH) | 1.ª mitad | 2.ª mitad |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        diff = None if r.mean is None or r.base_mean is None else r.mean - r.base_mean
        lines.append(
            f"| {r.signal} | {r.horizon} d | {r.n} | {pct(r.mean)} | {pct(r.base_mean)} | "
            f"{pct(diff)} | {pct(r.median)} | {num(r.hit_rate, 2)} | {num(r.p)} | "
            f"{num(r.q)}{' ✓' if r.significant else ''} | {pct(r.halves[0])} | "
            f"{pct(r.halves[1])} |")
    lines += ["", "## Datos", "",
              f"- Empresas con precio y CIK: {counts['companies']}; anuncios (8-K 2.02): "
              f"{counts['announcements']:,}; con reacción y deriva medibles dentro del índice: "
              f"{counts['scored']:,} ({counts['first_date']} → {counts['last_date']}).",
              f"- Por señal: {counts['by_signal']}. Reacción absoluta mediana: "
              f"{pct(counts['reaction_median_abs'])}.",
              f"- Puntos de la base: {counts['baseline_points']:,}.",
              "", "## Límites", "",
              "- **Sorpresa por la reacción del precio**, no por el consenso de analistas (de "
              "pago). Una reacción grande puede deberse a la guía, no a la cifra.",
              "- **Universo: S&P 500**, donde la anomalía es más débil por la atención de los "
              "analistas; las pequeñas empresas no se prueban aquí.",
              "- **Supervivencia:** faltan los miembros que salieron sin precio o con un ticker "
              "que ya no existe; muchas salidas vienen tras malas noticias, así que el sesgo "
              "juega en contra de la señal negativa.",
              "- Algún 8-K 2.02 es un avance de cifras preliminares, no un resultado.",
              "- Rentabilidades de precio, sin dividendos, en los dos lados."]
    return "\n".join(lines)
