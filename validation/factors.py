"""Quality, accruals and momentum: do they sort the S&P 500? (CLAUDE.md §15.4 point 9)

Three of the most cited anomalies that free data can compute, tested **before** the screen
may sort by any of them (section 9.7). The protocol and the decision rule are in
``settings.yaml`` (``factor_study``), written before the members' fundamentals were
downloaded; this module only executes them.

Signals, each oriented so that the **top** group is the one the literature expects to win:

- ``gross_profitability`` = gross profit / total assets, last fiscal year (Novy-Marx 2013).
  Only companies that file ``GrossProfit`` — the panel's own definition.
- ``accruals`` = (net income − operating cash flow) / average total assets, **negated**
  (Sloan 1996; the cash-flow version of Hribar and Collins 2002): profit without cash
  behind it is expected to disappoint.
- ``momentum`` = total return from twelve months to one month before the date (Jegadeesh
  and Titman 1993).

**Point-in-time.** Fundamentals are the last *annual* figures filed by the date, in the
version in force that day (``filed <= date``, section 9.4), and older than
``max_fiscal_age_days`` counts as no signal. Momentum uses closes up to a month before the
date. Returns start at the first session **after** it (``metrics.forward_return``).

**Test**, per signal and horizon, on a disjoint grid — each period starts where the
previous one ended. In each period, quintiles are formed among the members with a signal
and a forward return, and two numbers recorded: ``long`` = the top quintile's mean return
minus all of them (what an investor who only buys can use) and ``spread`` = top minus
bottom (the literature's figure). The periods are the sample — the stocks of one date share
the market's move and are not independent — and the p-value flips their signs at random
(``metrics.sign_flip_pvalue``). Benjamini-Hochberg runs across every test.

One company, one vote: share classes of the same CIK (GOOG and GOOGL) count once.

Total returns in today's share units apply every split and dividend through today; a
return between two dates is a ratio, invariant to the constant a later split introduces
(the same argument as ``transform/breadth.py``).

Pure functions; the runner (``run_validation.py --factors``) does the I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from ingest import sec_xbrl as xbrl
from transform import adjustments
from transform.screen import is_financial
from validation.metrics import benjamini_hochberg, grid_dates, sign_flip_pvalue

SIGNALS = ("gross_profitability", "accruals", "momentum")
METRICS = ("gross_profit", "net_income", "operating_cash_flow", "assets")
TESTS = ("long", "spread")
PREVIOUS_YEAR_TOLERANCE_DAYS = 45
FILL_LIMIT = 5          # sessions: a price hole is bridged, a delisting is not

RECORD_COLUMNS = ["cik", "metric", "period", "end", "filed", "value"]


# --- Fundamentals, point-in-time ---------------------------------------------------------


def fact_records(facts_by_cik: Mapping[str, Mapping[str, Any]],
                 concepts: Mapping[str, Mapping[str, Any]],
                 windows: Mapping[str, Any]) -> pd.DataFrame:
    """Every filed value of the study's metrics: :data:`RECORD_COLUMNS`.

    Uses the panel's own extraction (tag preference, duration windows), so the study reads
    each figure the way the screen would show it. ``period`` is ``fy`` for annual flows and
    ``''`` for balance-sheet instants.
    """
    rows = []
    for cik, facts in facts_by_cik.items():
        for metric in METRICS:
            spec = concepts.get(metric)
            if not spec:
                continue
            records, _stats = xbrl.extract_metric(facts, metric, spec, cik, windows)
            for record in records:
                parts = record["series_id"].split(":")
                if parts[-1] == xbrl.PROVENANCE_SUFFIX:
                    continue
                rows.append((cik, metric, parts[2] if len(parts) > 2 else "",
                             str(record["ts"])[:10], str(record["ts_release"])[:10],
                             float(record["value"])))
    frame = pd.DataFrame(rows, columns=RECORD_COLUMNS)
    frame["end"] = pd.to_datetime(frame["end"])
    frame["filed"] = pd.to_datetime(frame["filed"])
    return frame


def known(records: pd.DataFrame, metric: str, period: str, as_of: Any) -> pd.DataFrame:
    """``[cik, end, value]``: per company and period end, the version in force at ``as_of``
    — the latest filed on or before it (section 9.6: a restatement replaces the original
    only from the day it is filed)."""
    day = pd.Timestamp(as_of)
    rows = records[(records["metric"] == metric) & (records["period"] == period)
                   & (records["filed"] <= day)]
    return rows.sort_values("filed").drop_duplicates(["cik", "end"], keep="last")[
        ["cik", "end", "value"]]


def _latest_fresh(frame: pd.DataFrame, as_of: Any, max_age_days: int) -> pd.DataFrame:
    latest = frame.sort_values("end").drop_duplicates("cik", keep="last")
    return latest[latest["end"] >= pd.Timestamp(as_of) - pd.Timedelta(days=max_age_days)]


def gross_profitability(records: pd.DataFrame, as_of: Any, max_age_days: int) -> pd.Series:
    """Gross profit / total assets at the same fiscal year-end, per CIK."""
    profit = _latest_fresh(known(records, "gross_profit", "fy", as_of), as_of, max_age_days)
    assets = known(records, "assets", "", as_of).rename(columns={"value": "assets"})
    merged = profit.merge(assets, on=["cik", "end"])
    merged = merged[merged["assets"] > 0]
    return pd.Series((merged["value"] / merged["assets"]).to_numpy(),
                     index=merged["cik"].to_numpy(), dtype=float)


def accruals(records: pd.DataFrame, as_of: Any, max_age_days: int) -> pd.Series:
    """(Net income − operating cash flow) / average total assets of the fiscal year, per CIK
    (**not** negated: the orientation is the study's). The average needs the balance sheet
    a year earlier (± ``PREVIOUS_YEAR_TOLERANCE_DAYS``); without it, no value."""
    income = known(records, "net_income", "fy", as_of)
    cash = known(records, "operating_cash_flow", "fy", as_of)
    both = income.merge(cash, on=["cik", "end"], suffixes=("_ni", "_ocf"))
    both = _latest_fresh(both, as_of, max_age_days)
    assets = known(records, "assets", "", as_of)
    now = both.merge(assets.rename(columns={"value": "assets_end"}), on=["cik", "end"])
    if now.empty:
        return pd.Series(dtype=float)
    now = now.assign(target=now["end"] - pd.Timedelta(days=365)).sort_values("target")
    previous = assets.rename(columns={"end": "prev_end", "value": "assets_prev"}) \
        .sort_values("prev_end")
    merged = pd.merge_asof(now, previous, left_on="target", right_on="prev_end", by="cik",
                           direction="nearest",
                           tolerance=pd.Timedelta(days=PREVIOUS_YEAR_TOLERANCE_DAYS))
    merged = merged.dropna(subset=["assets_prev"])
    average = (merged["assets_end"] + merged["assets_prev"]) / 2
    merged, average = merged[average > 0], average[average > 0]
    return pd.Series(((merged["value_ni"] - merged["value_ocf"]) / average).to_numpy(),
                     index=merged["cik"].to_numpy(), dtype=float)


# --- Prices -------------------------------------------------------------------------------


def total_return_wide(closes: pd.DataFrame, actions: pd.DataFrame, as_of: Any,
                      calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """``sessions × tickers`` total-return indices (dividends added on their dates, section
    9.1), on ``calendar``. Built with the panel's tested :func:`adjustments.total_return_index`."""
    frame = adjustments.as_frame(actions)
    by_ticker = {t: g for t, g in frame.groupby("ticker")} if not frame.empty else {}
    empty = frame.iloc[0:0]
    columns = {}
    for series_id, group in closes.groupby("series_id"):
        ticker = str(series_id).rsplit(":", 1)[0]
        prices = group.sort_values("ts").drop_duplicates("ts", keep="last")[["ts", "value"]]
        index = adjustments.total_return_index(prices, by_ticker.get(ticker, empty), ticker,
                                               as_of)
        if index.empty:
            continue
        columns[ticker] = pd.Series(index["value"].astype(float).to_numpy(),
                                    index=pd.to_datetime(index["ts"].astype(str).str[:10]))
    wide = pd.DataFrame(columns)
    wide = wide[~wide.index.duplicated(keep="last")].sort_index()
    return wide.reindex(calendar)


def momentum(past: pd.DataFrame, as_of: Any, lookback_days: int, skip_days: int) -> pd.Series:
    """Total return from ``lookback_days`` to ``skip_days`` before ``as_of``, per ticker.
    ``past``: the total-return frame forward-filled a few sessions (only past values)."""
    day = pd.Timestamp(as_of)
    index = past.index
    end = index.searchsorted(day - pd.Timedelta(days=skip_days), side="right") - 1
    start = index.searchsorted(day - pd.Timedelta(days=lookback_days), side="right") - 1
    if start < 0 or end <= start:
        return pd.Series(dtype=float)
    first, last = past.iloc[start], past.iloc[end]
    return (last / first.where(first > 0) - 1).dropna()


def forward_returns(future: pd.DataFrame, as_of: Any, horizon_days: int) -> pd.Series:
    """Per ticker, the return from the first session after ``as_of`` to the first session
    ``horizon_days`` later — :func:`metrics.forward_return`, for every ticker at once.
    ``future``: the total-return frame back-filled a few sessions (the first price on or
    after each day). Empty when the window runs past the data."""
    index = future.index
    entry = index.searchsorted(pd.Timestamp(as_of), side="right")
    if entry >= len(index):
        return pd.Series(dtype=float)
    exit_ = index.searchsorted(index[entry] + pd.Timedelta(days=horizon_days))
    if exit_ >= len(index):
        return pd.Series(dtype=float)
    first, last = future.iloc[entry], future.iloc[exit_]
    return (last / first.where(first > 0) - 1).dropna()


# --- One period ---------------------------------------------------------------------------


def one_per_company(values: pd.Series, cik_of: Mapping[str, str]) -> pd.Series:
    """Tickers → one value per CIK (the alphabetically first ticker stands for it); tickers
    without a CIK stay as they are."""
    keys = pd.Series([cik_of.get(t, t) for t in values.index], index=values.index)
    keep = ~keys.sort_index().duplicated(keep="first")
    return values.loc[keep[keep].index]


def period_values(scores: pd.Series, returns: pd.Series, groups: int, min_names: int
                  ) -> tuple[float, float, int] | None:
    """``(long, spread, names)`` for one period, or ``None`` with fewer than ``min_names``
    stocks having both a score and a return. ``scores`` are already oriented: high = the
    group expected to win."""
    both = pd.concat({"score": scores, "ret": returns}, axis=1, join="inner").dropna()
    if len(both) < min_names:
        return None
    bucket = pd.qcut(both["score"].rank(method="first"), groups, labels=False)
    top = float(both.loc[bucket == groups - 1, "ret"].mean())
    bottom = float(both.loc[bucket == 0, "ret"].mean())
    return top - float(both["ret"].mean()), top - bottom, len(both)


# --- The battery ------------------------------------------------------------------------


@dataclass(frozen=True)
class Result:
    signal: str
    horizon: int
    test: str
    n: int
    mean: float | None
    median: float | None
    hit_rate: float | None
    names: float | None          # median stocks per period
    p: float | None
    q: float | None = None
    significant: bool = False
    halves: tuple[float | None, float | None] = (None, None)


def signal_scores(signal: str, date: pd.Timestamp, *, records: pd.DataFrame,
                  past: pd.DataFrame, members: list[str], cik_of: Mapping[str, str],
                  financial: Mapping[str, bool | None], cfg: Mapping[str, Any]) -> pd.Series:
    """Oriented scores per ticker for the members on ``date`` (high = expected to win)."""
    spec = cfg["signals"][signal]
    age = int(cfg["max_fiscal_age_days"])
    if signal == "momentum":
        raw = momentum(past, date, int(spec["lookback_days"]), int(spec["skip_days"]))
        raw = raw[raw.index.isin(members)]
    else:
        by_cik = gross_profitability(records, date, age) if signal == "gross_profitability" \
            else accruals(records, date, age)
        raw = pd.Series({t: by_cik[cik_of[t]] for t in members
                         if cik_of.get(t) in by_cik.index}, dtype=float)
    if spec.get("exclude_financials"):
        raw = raw[[financial.get(cik_of.get(t, "")) is not True for t in raw.index]]
    raw = one_per_company(raw, cik_of)
    return raw if str(spec["expected"]) == "positive" else -raw


def periods(records: pd.DataFrame, tr: pd.DataFrame, member: pd.DataFrame,
            cik_of: Mapping[str, str], financial: Mapping[str, bool | None],
            cfg: Mapping[str, Any]) -> pd.DataFrame:
    """``[signal, horizon, date, long, spread, names]`` — one row per signal, horizon and
    grid period with enough names."""
    past = tr.ffill(limit=FILL_LIMIT)
    future = tr.bfill(limit=FILL_LIMIT)
    sessions = tr.index[tr.index >= pd.Timestamp(cfg["first_date"])]
    rows = []
    for horizon in [int(h) for h in cfg["horizons_days"]]:
        for date in grid_dates(sessions, horizon):
            returns = forward_returns(future, date, horizon)
            if returns.empty:
                continue
            on = member.index[member.index <= date]
            if not len(on):
                continue
            row = member.loc[on[-1]]
            members = [t for t in row.index[row.to_numpy()] if t in tr.columns]
            for signal in cfg["signals"]:
                scores = signal_scores(signal, date, records=records, past=past,
                                       members=members, cik_of=cik_of, financial=financial,
                                       cfg=cfg)
                found = period_values(scores, returns, int(cfg["groups"]),
                                      int(cfg["min_names"]))
                if found is not None:
                    rows.append({"signal": signal, "horizon": horizon, "date": date,
                                 "long": found[0], "spread": found[1], "names": found[2]})
    return pd.DataFrame(rows, columns=["signal", "horizon", "date", *TESTS, "names"])


def run(table: pd.DataFrame, cfg: Mapping[str, Any]) -> list[Result]:
    """Every signal × horizon × test, BH across all of them."""
    split = pd.Timestamp(cfg["halves_split"])
    raw: list[Result] = []
    for signal in cfg["signals"]:
        for horizon in [int(h) for h in cfg["horizons_days"]]:
            rows = table[(table["signal"] == signal) & (table["horizon"] == horizon)]
            for test in TESTS:
                x = rows[test].astype(float)
                halves = tuple(float(x[side].mean()) if side.any() else None
                               for side in (rows["date"] <= split, rows["date"] > split))
                raw.append(Result(
                    signal=signal, horizon=horizon, test=test, n=len(x),
                    mean=float(x.mean()) if len(x) else None,
                    median=float(x.median()) if len(x) else None,
                    hit_rate=float((x > 0).mean()) if len(x) else None,
                    names=float(rows["names"].median()) if len(rows) else None,
                    p=sign_flip_pvalue(x.to_numpy(), n=int(cfg["permutations"]),
                                       seed=int(cfg["seed"])),
                    halves=halves))  # type: ignore[arg-type]
    tested = [r for r in raw if r.p is not None]
    adjusted = benjamini_hochberg([r.p for r in tested], float(cfg["fdr_alpha"]))
    qmap = {id(r): a for r, a in zip(tested, adjusted)}
    return [Result(**{**r.__dict__, "q": qmap[id(r)][0], "significant": qmap[id(r)][1]})
            if id(r) in qmap else r for r in raw]


def decision(results: list[Result], cfg: Mapping[str, Any]) -> dict[str, bool]:
    """The rule written before running: the ``decision_test`` significant after BH at a
    decision horizon, positive (the scores are oriented, so positive = the expected sign),
    and positive in both halves."""
    out = {signal: False for signal in cfg["signals"]}
    for r in results:
        if r.test != cfg["decision_test"] or r.horizon not in cfg["decision_horizons"]:
            continue
        ok = (r.significant and r.mean is not None and r.mean > 0
              and all(v is not None and v > 0 for v in r.halves))
        out[r.signal] = out[r.signal] or ok
    return out


# --- The study, end to end ----------------------------------------------------------------


def study(facts_by_cik: Mapping[str, Mapping[str, Any]], cik_of: Mapping[str, str],
          sics: Mapping[str, Any], closes: pd.DataFrame, actions: pd.DataFrame,
          intervals: pd.DataFrame, concepts: Mapping[str, Mapping[str, Any]],
          windows: Mapping[str, Any], cfg: Mapping[str, Any], as_of: Any) -> dict[str, Any]:
    """Records → total returns → periods → battery → decision. Pure: every input passed in.

    Args:
        facts_by_cik: ``ingest.member_facts`` subsets.
        cik_of: Member ticker → CIK (tickers the SEC map no longer knows are absent).
        sics: CIK → SIC code.
        closes: Raw closes of the members and of ``cfg['calendar_ticker']``, long format.
        actions: ``corporate_actions`` rows (splits and dividends).
        intervals: ``universe_membership`` rows.
    """
    from transform import breadth as br  # noqa: PLC0415

    records = fact_records(facts_by_cik, concepts, windows)
    calendar_ticker = str(cfg["calendar_ticker"])
    calendar_rows = closes[closes["series_id"] == f"{calendar_ticker}:close_raw"]
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(
        calendar_rows["ts"].astype(str).str[:10]).unique()))
    member_closes = closes[closes["series_id"] != f"{calendar_ticker}:close_raw"]
    tr = total_return_wide(member_closes, actions, as_of, calendar)
    member = br.membership_mask(intervals, tr.index, list(tr.columns))
    member = member.reindex(columns=tr.columns, fill_value=False)
    financial = {cik: is_financial(sic) for cik, sic in sics.items()}
    table = periods(records, tr, member, cik_of, financial, cfg)
    results = run(table, cfg)
    counts = {
        "companies_with_facts": int(records["cik"].nunique()) if len(records) else 0,
        "with_gross_profit": int(records.loc[records["metric"] == "gross_profit", "cik"]
                                 .nunique()) if len(records) else 0,
        "priced_members": int(tr.shape[1]),
        "members_without_cik": sorted(t for t in tr.columns if t not in cik_of),
        "financials": int(sum(1 for v in financial.values() if v)),
        "sic_unknown": int(sum(1 for v in financial.values() if v is None)),
        "first_date": str(table["date"].min().date()) if len(table) else None,
        "last_date": str(table["date"].max().date()) if len(table) else None,
    }
    return {"results": results, "decision": decision(results, cfg), "counts": counts,
            "periods": table}


# --- Report -------------------------------------------------------------------------------

LABELS = {"gross_profitability": "Rentabilidad bruta (calidad)",
          "accruals": "Devengos bajos", "momentum": "Momentum 12-1"}
TEST_LABELS = {"long": "1.er quintil − todas", "spread": "1.er − último quintil"}


def report(outcome: Mapping[str, Any], cfg: Mapping[str, Any], generated: str) -> str:
    """The Markdown report, the verdict first (section 8, phase 4)."""
    def pct(x: float | None) -> str:
        return "—" if x is None or pd.isna(x) else f"{x * 100:+.2f} %".replace(".", ",")

    def num(x: float | None, d: int = 3) -> str:
        return "—" if x is None or pd.isna(x) else f"{x:.{d}f}".replace(".", ",")

    counts, results = outcome["counts"], outcome["results"]
    lines = [f"# Calidad, devengos y momentum en el S&P 500 ({generated})", "",
             "Protocolo y regla de decisión escritos en `settings.yaml` (`factor_study`) "
             "antes de descargar los fundamentales; una sola ejecución.", "",
             "## Veredicto", ""]
    for signal, ok in outcome["decision"].items():
        lines.append(f"- **{LABELS.get(signal, signal)}**: "
                     + ("VALIDADA según la regla escrita." if ok
                        else "**no validada** según la regla escrita."))
    lines += ["", "## Pruebas (rentabilidad total por periodo; signo ya orientado: "
              "positivo = lo que la literatura espera)", "",
              "| Señal | Horizonte | Prueba | Periodos | Media | Mediana | % positivos | "
              "Empresas/periodo | p | q (BH) | 2018-2021 | 2022-hoy |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {LABELS.get(r.signal, r.signal)} | {r.horizon} d | {TEST_LABELS[r.test]} | "
            f"{r.n} | {pct(r.mean)} | {pct(r.median)} | {num(r.hit_rate, 2)} | "
            f"{num(r.names, 0)} | {num(r.p)} | {num(r.q)}{' ✓' if r.significant else ''} | "
            f"{pct(r.halves[0])} | {pct(r.halves[1])} |")
    lines += ["", "## Datos", "",
              f"- Miembros con precio: {counts['priced_members']}; sin CIK en el mapa actual "
              f"de la SEC: {len(counts['members_without_cik'])} "
              f"({', '.join(counts['members_without_cik'][:12])}…).",
              f"- Empresas con hechos XBRL: {counts['companies_with_facts']}; que presentan "
              f"`GrossProfit`: {counts['with_gross_profit']}.",
              f"- Financieras (SIC 6000-6799) excluidas de calidad y devengos: "
              f"{counts['financials']}; SIC desconocido: {counts['sic_unknown']}.",
              f"- Periodos del {counts['first_date']} al {counts['last_date']}.",
              "", "## Límites", "",
              "- **Universo: S&P 500 desde 2018**, unos 8 años. Pocas empresas grandes muy "
              "seguidas: donde la literatura encuentra los efectos más débiles.",
              "- **Supervivencia:** los miembros que dejaron de cotizar y yfinance olvidó "
              "faltan, y los que dejan de cotizar dentro de un periodo salen de él (una "
              "adquisición suele ser con prima: el sesgo va en contra de quien las tenía).",
              "- **CIK por el mapa de hoy:** un ticker reutilizado por otra empresa llevaría "
              "sus cifras; el filtro de antigüedad del ejercicio elimina la mayoría de casos.",
              "- **Calidad solo con `GrossProfit` presentado:** las empresas que no lo "
              "presentan quedan fuera de esa señal, y no son una muestra al azar.",
              "- Periodos de una misma señal a 30, 90 y 180 días cubren los mismos años: las "
              "tres pruebas no son independientes entre sí."]
    return "\n".join(lines)
