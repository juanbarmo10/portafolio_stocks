"""Do insider purchases precede returns above the market? (CLAUDE.md §15.1.6)

The user's rule: the signal is validated before the panel shows it. The protocol and the
decision rule live in ``settings.yaml`` (``insider_study``) and were written before any
data was downloaded; this module only executes them.

**Events** (from ``ingest/insiders.purchases``), each dated by its **filing date** — when it
became public (§9.4) — with one event per company per ``cooldown_days`` so the samples are
close to independent:

- ``cluster``: ``min_insiders`` distinct insiders filing purchases within ``window_days``.
  A joint filing (a fund and its partners reporting the same purchase) counts once.
- ``executive``: a purchase by the CEO or CFO worth at least ``min_value_usd``.

**Measure.** Excess return over SPY (price returns, both) from the close of the first
session after the filing to the first close ``h`` calendar days later
(``metrics.forward_return``). Only while the company was an S&P 500 member and priced.

**Baseline.** The same companies on a monthly grid, away from their own events by more than
``cooldown_days`` — "what these stocks do on an ordinary day", so the test isolates the
purchase, not the kind of company that has insiders buying.

**Test.** Two-sided permutation test of the difference in means, Benjamini-Hochberg over
the six signal × horizon pairs, and the sign in each half of the sample.

Known limits, said in the report: the universe is the S&P 500 (where the literature finds
the weakest effect; small caps are not tested); members that left the index without prices
are missing — and a company where insiders bought and that then collapsed is exactly that,
so the bias **favours** the signal; code P also covers private placements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from validation.metrics import benjamini_hochberg, forward_return, permutation_pvalue


def _cooldown(dates: list[pd.Timestamp], days: int) -> list[pd.Timestamp]:
    kept: list[pd.Timestamp] = []
    for day in sorted(dates):
        if not kept or (day - kept[-1]).days >= days:
            kept.append(day)
    return kept


def cluster_events(purchases: pd.DataFrame, *, min_insiders: int, window_days: int,
                   cooldown_days: int) -> pd.DataFrame:
    """``[signal, issuer_cik, symbol, date]`` — the day the ``min_insiders``-th distinct
    buyer's filing made the group public."""
    frame = purchases.assign(day=pd.to_datetime(purchases["filing_date"]))
    # One buyer per filing: the smallest owner CIK stands for a joint filing.
    filers = frame.groupby(["issuer_cik", "accession"]).agg(
        day=("day", "min"), buyer=("owner_cik", "min"), symbol=("symbol", "first")).reset_index()
    rows = []
    for cik, group in filers.groupby("issuer_cik"):
        group = group.sort_values("day")
        candidates = []
        for day in group["day"].unique():
            window = group[(group["day"] > day - pd.Timedelta(days=window_days))
                           & (group["day"] <= day)]
            if window["buyer"].nunique() >= min_insiders:
                candidates.append(pd.Timestamp(day))
        symbol = group["symbol"].iloc[-1]
        rows += [{"signal": "cluster", "issuer_cik": cik, "symbol": symbol, "date": d}
                 for d in _cooldown(candidates, cooldown_days)]
    return pd.DataFrame(rows, columns=["signal", "issuer_cik", "symbol", "date"])


def executive_events(purchases: pd.DataFrame, *, title_pattern: str, min_value_usd: float,
                     cooldown_days: int) -> pd.DataFrame:
    """``[signal, issuer_cik, symbol, date]`` — a CEO or CFO purchase of at least the value."""
    pattern = re.compile(title_pattern, re.IGNORECASE)
    frame = purchases[purchases["title"].fillna("").map(lambda t: bool(pattern.search(t)))]
    frame = frame.assign(day=pd.to_datetime(frame["filing_date"]))
    # A filing's value is summed over its lines once, not once per joint owner.
    per_filing = frame.drop_duplicates(["accession", "shares", "price", "day"]).groupby(
        ["issuer_cik", "accession"]).agg(day=("day", "min"), value=("value", "sum"),
                                         symbol=("symbol", "first")).reset_index()
    per_filing = per_filing[per_filing["value"] >= min_value_usd]
    rows = []
    for cik, group in per_filing.groupby("issuer_cik"):
        symbol = group.sort_values("day")["symbol"].iloc[-1]
        rows += [{"signal": "executive", "issuer_cik": cik, "symbol": symbol, "date": d}
                 for d in _cooldown(list(group["day"]), cooldown_days)]
    return pd.DataFrame(rows, columns=["signal", "issuer_cik", "symbol", "date"])


def normalize_symbol(symbol: str) -> str:
    """Form 4 writes class shares as ``BRK-B``, ``BRKB`` or ``BRK.B``; the membership list
    as ``BRK.B``. Only the separator is normalized — nothing is guessed."""
    return str(symbol).upper().strip().replace("-", ".").replace("/", ".")


def excess_returns(prices: pd.DataFrame, spy: pd.Series, member: pd.DataFrame,
                   points: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """For ``points`` ``[symbol, date, ...]``: add ``excess_{h}`` per horizon.

    ``None`` when the company was not an S&P 500 member that day, has no price, or the
    window runs past the data. ``prices`` are split-adjusted closes (dates × tickers).
    """
    out = points.copy()
    for h in horizons:
        out[f"excess_{h}"] = None
    for idx, row in out.iterrows():
        ticker, day = row["symbol"], pd.Timestamp(row["date"])
        if ticker not in prices.columns or ticker not in member.columns:
            continue
        on = member.index[member.index <= day]
        if not len(on) or not bool(member.at[on[-1], ticker]):
            continue
        for h in horizons:
            stock = forward_return(prices[ticker], day, h)
            market = forward_return(spy, day, h)
            if stock is not None and market is not None:
                out.at[idx, f"excess_{h}"] = stock - market
    return out


def baseline_points(events: pd.DataFrame, member: pd.DataFrame, *, step_days: int,
                    away_days: int) -> pd.DataFrame:
    """Monthly grid dates of the event companies, farther than ``away_days`` from any of
    their own events and while they were members."""
    rows = []
    for symbol, group in events.groupby("symbol"):
        if symbol not in member.columns:
            continue
        days = member.index[member[symbol].to_numpy()]
        if not len(days):
            continue
        dates = pd.date_range(days.min(), days.max(), freq=f"{step_days}D")
        own = pd.to_datetime(group["date"]).to_numpy()
        for day in dates:
            if np.min(np.abs((own - day.to_datetime64()) / np.timedelta64(1, "D"))) > away_days:
                rows.append({"symbol": symbol, "date": day})
    return pd.DataFrame(rows, columns=["symbol", "date"])


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
    base_hit_rate: float | None
    p: float | None
    q: float | None = None
    significant: bool = False
    halves: tuple[float | None, float | None] = (None, None)


def run(events: pd.DataFrame, base: pd.DataFrame, cfg: Mapping[str, Any]) -> list[Result]:
    """The battery: every signal × horizon, BH across all of them."""
    horizons = list(cfg["horizons_days"])
    split = pd.Timestamp(cfg["halves_split"])
    raw: list[Result] = []
    for signal in sorted(events["signal"].unique()):
        ev = events[events["signal"] == signal]
        symbols = set(ev["symbol"])
        bs = base[base["symbol"].isin(symbols)]
        for h in horizons:
            x = pd.to_numeric(ev[f"excess_{h}"], errors="coerce").dropna()
            b = pd.to_numeric(bs[f"excess_{h}"], errors="coerce").dropna()
            p = permutation_pvalue(x, b, n=int(cfg["permutations"]), seed=int(cfg["seed"])) \
                if len(x) and len(b) else None
            # Each half against the baseline of the same half: markets differ between them.
            halves = []
            for ev_side, base_side in ((ev["date"] <= split, bs["date"] <= split),
                                       (ev["date"] > split, bs["date"] > split)):
                xs = pd.to_numeric(ev.loc[ev_side, f"excess_{h}"], errors="coerce").dropna()
                bh = pd.to_numeric(bs.loc[base_side, f"excess_{h}"], errors="coerce").dropna()
                halves.append(float(xs.mean() - bh.mean()) if len(xs) and len(bh) else None)
            raw.append(Result(
                signal=signal, horizon=h, n=len(x), n_base=len(b),
                mean=float(x.mean()) if len(x) else None,
                base_mean=float(b.mean()) if len(b) else None,
                median=float(x.median()) if len(x) else None,
                hit_rate=float((x > 0).mean()) if len(x) else None,
                base_hit_rate=float((b > 0).mean()) if len(b) else None,
                p=p, halves=(halves[0], halves[1])))
    tested = [r for r in raw if r.p is not None]
    adjusted = benjamini_hochberg([r.p for r in tested], float(cfg["fdr_alpha"]))
    qmap = {id(r): a for r, a in zip(tested, adjusted)}
    return [Result(**{**r.__dict__, "q": qmap[id(r)][0], "significant": qmap[id(r)][1]})
            if id(r) in qmap else r for r in raw]


def decision(results: list[Result], cfg: Mapping[str, Any]) -> dict[str, bool]:
    """The rule written before running: significant after BH at a decision horizon, a
    positive mean difference, and the same sign in both halves."""
    out: dict[str, bool] = {}
    for r in results:
        ok = (r.horizon in cfg["decision_horizons"] and r.significant
              and r.mean is not None and r.base_mean is not None and r.mean > r.base_mean
              and all(v is not None and v > 0 for v in r.halves))
        out[r.signal] = out.get(r.signal, False) or ok
    return out


# --- The study, end to end ----------------------------------------------------------------


def study(purchases: pd.DataFrame, closes: pd.DataFrame, actions: Any, intervals: Any,
          cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Events → excess returns → battery → decision. Pure: every input is passed in.

    Args:
        purchases: ``ingest.insiders.COLUMNS`` rows, every quarter.
        closes: Raw closes of the S&P 500 members and SPY, long format.
        actions: ``corporate_actions`` rows, for the split adjustment.
        intervals: ``universe_membership`` rows.
    """
    from transform import breadth as br  # noqa: PLC0415

    signals = cfg["signals"]
    events = pd.concat([
        cluster_events(purchases, min_insiders=int(signals["cluster"]["min_insiders"]),
                       window_days=int(signals["cluster"]["window_days"]),
                       cooldown_days=int(cfg["cooldown_days"])),
        executive_events(purchases, title_pattern=str(signals["executive"]["title_pattern"]),
                         min_value_usd=float(signals["executive"]["min_value_usd"]),
                         cooldown_days=int(cfg["cooldown_days"])),
    ], ignore_index=True)
    events["symbol"] = events["symbol"].map(normalize_symbol)
    wide = br.adjusted_closes(br.wide_closes(closes), br.splits_by_ticker(actions))
    spy = wide.pop(str(cfg["benchmark"])).dropna()
    member = br.membership_mask(intervals, wide.index, list(wide.columns))
    member = member.reindex(columns=wide.columns, fill_value=False)
    first_price = wide.index.min()
    horizons = [int(h) for h in cfg["horizons_days"]]

    in_index = events[events["symbol"].isin(wide.columns) & (events["date"] >= first_price)]
    scored = excess_returns(wide, spy, member, in_index, horizons)
    usable = scored.dropna(subset=[f"excess_{horizons[0]}"])
    base = excess_returns(wide, spy, member,
                          baseline_points(usable, member, step_days=int(cfg["baseline_step_days"]),
                                          away_days=int(cfg["cooldown_days"])), horizons)
    results = run(usable, base, cfg)
    counts = {
        "purchases": len(purchases),
        "events": events.groupby("signal").size().to_dict(),
        "events_in_sp500": usable.groupby("signal").size().to_dict(),
        "baseline_points": int(base[f"excess_{horizons[0]}"].notna().sum()),
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
    lines = [f"# Compras de directivos: ¿preceden rentabilidad sobre el S&P 500? ({generated})",
             "",
             "Protocolo y regla de decisión escritos en `settings.yaml` (`insider_study`) "
             "antes de descargar los datos; una sola ejecución.", "",
             "## Veredicto", ""]
    for signal, ok in outcome["decision"].items():
        lines.append(f"- **{signal}**: " + ("VALIDADA según la regla escrita." if ok else
                                          "**no validada** según la regla escrita."))
    lines += ["", "## Pruebas (exceso sobre SPY: eventos frente a los mismos valores en días "
              "normales)", "",
              "| Señal | Horizonte | n | Exceso medio | Base | Diferencia | Mediana | "
              "% positivos (base) | p | q (BH) | 1.ª mitad | 2.ª mitad |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        diff = None if r.mean is None or r.base_mean is None else r.mean - r.base_mean
        lines.append(
            f"| {r.signal} | {r.horizon} d | {r.n} | {pct(r.mean)} | {pct(r.base_mean)} | "
            f"{pct(diff)} | {pct(r.median)} | {num(r.hit_rate, 2)} ({num(r.base_hit_rate, 2)}) | "
            f"{num(r.p)} | {num(r.q)}{' ✓' if r.significant else ''} | {pct(r.halves[0])} | "
            f"{pct(r.halves[1])} |")
    lines += ["", "## Datos", "",
              f"- Compras de consejeros y directivos (Form 4, código P): {counts['purchases']:,}.",
              f"- Eventos: {counts['events']}; dentro del S&P 500 con precio: "
              f"{counts['events_in_sp500']} ({counts['first_date']} → {counts['last_date']}).",
              f"- Puntos de la base: {counts['baseline_points']:,}.",
              "", "## Límites", "",
              "- **Universo: S&P 500.** Es donde la literatura encuentra el efecto más débil; "
              "las pequeñas empresas no se prueban aquí.",
              "- **Supervivencia a favor de la señal:** los que salieron del índice sin precio "
              "faltan, y una empresa donde los directivos compraron y que luego se hundió es "
              "justo eso.",
              "- El código P incluye colocaciones privadas; rentabilidades de precio, sin "
              "dividendos, en los dos lados.",
              "- Eventos concentrados en el tiempo (p. ej. marzo de 2020) no son independientes "
              "entre sí aunque sean de empresas distintas; la mitad por mitad es la defensa."]
    return "\n".join(lines)
