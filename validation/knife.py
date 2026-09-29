"""Does buying a falling knife (or an overextended stock) do worse than the rest? (§15.5 p. 12)

The user's rule of thumb — do not buy a stock falling hard unless there is a good reason —
has support elsewhere in this project: insiders buying after deep drops saw the stocks keep
underperforming (RESEARCH.md §2.42, §2.62). This study asks it directly. The protocol lives in
``settings.yaml`` (``knife_study``), written before anything was computed; this module only
executes it.

**Conditions**, on split-adjusted closes known at the date:

- ``knife``: close below its ``sma_sessions`` average **and** a return of ``max_return`` or
  worse over the last ``return_days`` calendar days.
- ``overextended``: close at least ``min_ratio_to_sma`` times its average.

**Comparison within each date.** Knives crowd into market sell-offs (March 2020, 2022), so
comparing them with "ordinary days" would measure the market, not the rule. On a quarterly
grid, each date gives one number: the mean excess return over SPY of the members meeting the
condition minus that of the other members, from the first session after the date to ``h``
calendar days later. The test is a sign-flip permutation across dates (each date one
observation, since stocks on the same date share the market's move), Benjamini-Hochberg over
the battery, and the sign in each half.

Pure: every input is passed in; no network, no database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from validation.metrics import benjamini_hochberg, sign_flip_pvalue


def _row_at(frame: pd.DataFrame, day: pd.Timestamp, *, after: bool = False) -> pd.Series | None:
    """The row of the last session on or before ``day`` (or the first strictly after)."""
    index = frame.index
    pos = index.searchsorted(day, side="right")
    pos = pos if after else pos - 1
    if pos < 0 or pos >= len(index):
        return None
    return frame.iloc[pos]


def flags_at(closes: pd.DataFrame, sma: pd.DataFrame, day: pd.Timestamp,
             cfg: Mapping[str, Any]) -> pd.DataFrame:
    """``[knife, overextended]`` booleans per ticker, from what was known on ``day``.
    A ticker without a close, an average or the earlier close is left out (no flag)."""
    knife_cfg, over_cfg = cfg["conditions"]["knife"], cfg["conditions"]["overextended"]
    price, average = _row_at(closes, day), _row_at(sma, day)
    earlier = _row_at(closes, day - pd.Timedelta(days=int(knife_cfg["return_days"])))
    if price is None or average is None or earlier is None:
        return pd.DataFrame(columns=["knife", "overextended"])
    ret = price / earlier - 1
    known = price.notna() & average.notna() & earlier.notna() & (earlier > 0)
    out = pd.DataFrame({
        "knife": (price < average) & (ret <= float(knife_cfg["max_return"])),
        "overextended": price >= float(over_cfg["min_ratio_to_sma"]) * average,
    })
    return out[known]


def forward_excess(closes: pd.DataFrame, bench: pd.Series, day: pd.Timestamp,
                   horizon: int) -> pd.Series:
    """Excess return over the benchmark per ticker, from the close of the first session
    after ``day`` to the first close at least ``horizon`` days after that entry."""
    index = closes.index
    entry_pos = index.searchsorted(day, side="right")
    if entry_pos >= len(index):
        return pd.Series(dtype=float)
    entry = index[entry_pos]
    exit_pos = index.searchsorted(entry + pd.Timedelta(days=horizon), side="left")
    if exit_pos >= len(index):
        return pd.Series(dtype=float)
    leaving = index[exit_pos]
    stock = closes.loc[leaving] / closes.loc[entry] - 1
    b_in, b_out = bench.asof(entry), bench.asof(leaving)
    if pd.isna(b_in) or pd.isna(b_out) or b_in <= 0:
        return pd.Series(dtype=float)
    return (stock - (b_out / b_in - 1)).dropna()


def spreads(closes: pd.DataFrame, bench: pd.Series, member: pd.DataFrame,
            cfg: Mapping[str, Any]) -> pd.DataFrame:
    """``[date, condition, horizon, spread, n_condition, n_rest]`` over the grid."""
    windows = {int(c["sma_sessions"]) for c in cfg["conditions"].values()}
    sma = {w: closes.rolling(w, min_periods=w).mean() for w in windows}
    start = max(pd.Timestamp(cfg["first_date"]), closes.index.min())
    rows = []
    for day in pd.date_range(start, closes.index.max(), freq=f"{int(cfg['grid_step_days'])}D"):
        on = _row_at(member, day)
        if on is None:
            continue
        members = set(on.index[on.to_numpy(dtype=bool)])
        for name, spec in cfg["conditions"].items():
            flags = flags_at(closes, sma[int(spec["sma_sessions"])], day, cfg)
            flags = flags[flags.index.isin(members)]
            if flags.empty:
                continue
            for h in [int(x) for x in cfg["horizons_days"]]:
                excess = forward_excess(closes[flags.index], bench, day, h)
                if excess.empty:
                    continue
                inside = excess[flags.loc[excess.index, name].astype(bool)]
                rest = excess[~flags.loc[excess.index, name].astype(bool)]
                if len(inside) < int(cfg["min_names"]) or len(rest) < int(cfg["min_names"]):
                    continue
                rows.append({"date": day, "condition": name, "horizon": h,
                             "spread": float(inside.mean() - rest.mean()),
                             "n_condition": len(inside), "n_rest": len(rest)})
    return pd.DataFrame(rows, columns=["date", "condition", "horizon", "spread",
                                       "n_condition", "n_rest"])


@dataclass(frozen=True)
class Result:
    condition: str
    horizon: int
    dates: int
    mean_spread: float | None
    median_names: float | None
    negative_share: float | None
    p: float | None
    q: float | None = None
    significant: bool = False
    halves: tuple[float | None, float | None] = (None, None)


def run(table: pd.DataFrame, cfg: Mapping[str, Any]) -> list[Result]:
    split = pd.Timestamp(cfg["halves_split"])
    raw = []
    for (name, h), group in table.groupby(["condition", "horizon"]):
        x = group["spread"].to_numpy(dtype=float)
        halves = tuple(float(g["spread"].mean()) if len(g) else None
                       for g in (group[group["date"] <= split], group[group["date"] > split]))
        raw.append(Result(condition=name, horizon=int(h), dates=len(x),
                          mean_spread=float(x.mean()) if len(x) else None,
                          median_names=float(group["n_condition"].median()) if len(x) else None,
                          negative_share=float((x < 0).mean()) if len(x) else None,
                          p=sign_flip_pvalue(x, n=int(cfg["permutations"]), seed=int(cfg["seed"])),
                          halves=halves))
    tested = [r for r in raw if r.p is not None]
    adjusted = benjamini_hochberg([r.p for r in tested], float(cfg["fdr_alpha"]))
    qmap = {id(r): a for r, a in zip(tested, adjusted)}
    return [Result(**{**r.__dict__, "q": qmap[id(r)][0], "significant": qmap[id(r)][1]})
            if id(r) in qmap else r for r in raw]


def decision(results: list[Result], cfg: Mapping[str, Any]) -> dict[str, bool]:
    """The rule written before running: a gate only if, at a decision horizon, the condition
    does significantly WORSE than the rest, and worse in both halves."""
    out: dict[str, bool] = {}
    for r in results:
        ok = (r.horizon in [int(h) for h in cfg["decision_horizons"]] and r.significant
              and r.mean_spread is not None and r.mean_spread < 0
              and all(v is not None and v < 0 for v in r.halves))
        out[r.condition] = out.get(r.condition, False) or ok
    return out


def study(closes: pd.DataFrame, actions: Any, intervals: Any,
          cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Closes → flags → within-date spreads → battery → decision."""
    from transform import breadth as br  # noqa: PLC0415

    wide = br.adjusted_closes(br.wide_closes(closes), br.splits_by_ticker(actions))
    bench = wide.pop(str(cfg["benchmark"])).dropna()
    member = br.membership_mask(intervals, wide.index, list(wide.columns))
    member = member.reindex(columns=wide.columns, fill_value=False)
    table = spreads(wide, bench, member, cfg)
    results = run(table, cfg)
    counts = {
        "tickers": int(wide.shape[1]),
        "dates": int(table["date"].nunique()) if len(table) else 0,
        "first_date": str(table["date"].min().date()) if len(table) else None,
        "last_date": str(table["date"].max().date()) if len(table) else None,
    }
    return {"results": results, "decision": decision(results, cfg), "counts": counts,
            "table": table}


def report(outcome: Mapping[str, Any], cfg: Mapping[str, Any], generated: str) -> str:
    def pct(x: float | None) -> str:
        return "—" if x is None or pd.isna(x) else f"{x * 100:+.2f} %".replace(".", ",")

    def num(x: float | None, d: int = 3) -> str:
        return "—" if x is None or pd.isna(x) else f"{x:.{d}f}".replace(".", ",")

    c = outcome["counts"]
    lines = [f"# «Cuchillo cayendo» y sobreextensión en el S&P 500 ({generated})", "",
             "Protocolo y regla escritos en `settings.yaml` (`knife_study`) antes de calcular "
             "nada; una sola ejecución.", "", "## Veredicto", ""]
    for name, ok in outcome["decision"].items():
        lines.append(f"- **{name}**: " + ("PUERTA según la regla escrita." if ok else
                                        "**sin puerta** según la regla escrita."))
    lines += ["", "## Pruebas (por fecha: exceso sobre SPY de las que cumplen la condición "
              "menos el del resto de miembros ese día)", "",
              "| Condición | Horizonte | Fechas | Diferencia media | Acciones por fecha "
              "(mediana) | % fechas negativas | p | q (BH) | 1.ª mitad | 2.ª mitad |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in outcome["results"]:
        lines.append(f"| {r.condition} | {r.horizon} d | {r.dates} | {pct(r.mean_spread)} | "
                     f"{num(r.median_names, 0)} | {num(r.negative_share, 2)} | {num(r.p)} | "
                     f"{num(r.q)}{' ✓' if r.significant else ''} | {pct(r.halves[0])} | "
                     f"{pct(r.halves[1])} |")
    lines += ["", "## Datos", "",
              f"- Valores con precio: {c['tickers']}; fechas de la rejilla con datos: "
              f"{c['dates']} ({c['first_date']} → {c['last_date']}).",
              "", "## Límites", "",
              "- **S&P 500 solamente** (precios desde 2017): unas 36 fechas trimestrales, y los "
              "horizontes de 180 días se solapan entre fechas vecinas.",
              "- **Supervivencia en contra de la puerta:** faltan los cuchillos que acabaron sin "
              "precio (quiebras, exclusiones).",
              "- Rentabilidades de precio, sin dividendos, en los dos lados."]
    return "\n".join(lines)
