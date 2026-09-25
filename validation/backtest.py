"""Signal validation battery for the regime light (CLAUDE.md sections 8 phase 4, 9.7).

The question: **do the light and its components say anything about what the market does
next?** The light only *blocks* purchases, so the relevant cost-benefit is exactly this: if
buying during red is followed by returns no worse than buying at any other time, the block
costs returns and protects nothing.

Pre-registered before any result was looked at (2026-09-24), and not changed after:

- **Signals.** The verdict red, the verdict green, and each of the eight components voting
  risk-off — all built by :func:`transform.regime.build`, the same code the panel uses,
  point-in-time as the panel shows it.
- **Hypotheses.** Red and every component's risk-off vote precede *lower* returns than the
  baseline; green precedes *higher* ones.
- **Outcome.** The S&P 500's total return (SPY with dividends added on their dates, section
  9.1) over 30, 90 and 180 calendar days, entering at the next session's close.
- **Sampling.** A grid of non-overlapping windows, one horizon apart
  (:func:`validation.metrics.grid_dates`); the signal and the baseline are the grid dates
  where the signal is on and off. Each signal is judged over the window where it exists.
- **Test.** Two-sided permutation test on the difference of means, 10 000 shuffles;
  Benjamini-Hochberg at q = 0.10 over every test in the battery with at least eight dates
  on each side. The rest are reported as *insufficient*, never dropped (section 9.7).
- **Robustness, descriptive only.** The grid's starting day is arbitrary, so the edge is
  recomputed from ten other starting days and the share with the same sign is reported.

Also here: the one hypothesis left open by phase 3 that *can* be tested out of sample. The
RSP/SPY component did not discriminate corrections from calm in 2012-2026. It is built
from prices alone — no vintages, no revisions — so 2004-2012 is a clean out-of-sample
period for it. The NFCI hypothesis cannot be: its point-in-time history starts in 2011 and
every revised value before that would be look-ahead. Saying so is the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from transform import regime as rg
from validation.metrics import (
    benjamini_hochberg,
    forward_return,
    grid_dates,
    permutation_pvalue,
)

LOWER, HIGHER = -1, 1
# Trading sessions per calendar day, to turn a horizon in days into grid offsets in sessions.
SESSIONS_PER_DAY = 252 / 365


@dataclass(frozen=True)
class Signal:
    """A boolean series (NaN where the signal does not exist) and what it claims."""

    key: str
    label: str
    direction: int          # −1: precedes lower returns; +1: higher
    series: pd.Series


@dataclass
class TestResult:
    signal: str
    label: str
    direction: int
    horizon: int
    window: str
    n_signal: int
    n_baseline: int
    mean_signal: float | None
    mean_baseline: float | None
    edge: float | None
    pvalue: float | None
    sign_stability: float | None
    status: str                         # 'ok' | 'insufficient'
    qvalue: float | None = None
    significant: bool | None = None

    @property
    def agrees(self) -> bool | None:
        """Whether the edge has the sign the hypothesis predicted."""
        return None if self.edge is None else bool(np.sign(self.edge) == self.direction)


@dataclass
class Battery:
    results: list[TestResult]
    fdr_alpha: float
    horizons: list[int]
    notes: list[str] = field(default_factory=list)


def signals(built: rg.RegimeBuild) -> list[Signal]:
    """The battery's signals, straight from the light as the panel builds it."""
    verdict = built.frame["verdict"]
    judged = verdict != rg.INSUFFICIENT
    out = [
        Signal("verdict_red", "Veredicto rojo (risk-off)", LOWER,
               (verdict == rg.RISK_OFF).astype(float).where(judged)),
        Signal("verdict_green", "Veredicto verde (risk-on)", HIGHER,
               (verdict == rg.RISK_ON).astype(float).where(judged)),
    ]
    for rule in built.rules:
        vote = built.votes[rule.key]["vote"]
        out.append(Signal(f"off:{rule.key}", f"{rule.label}: vota risk-off", LOWER,
                          (vote == -1).astype(float).where(vote.notna())))
    return out


def _sample(signal: pd.Series, prices: pd.Series, horizon: int, offset: int,
            outcome: Callable[..., float | None] = forward_return
            ) -> tuple[list[float], list[float]]:
    exists = signal.dropna()
    on, off = [], []
    for day in grid_dates(exists.index, horizon, offset):
        r = outcome(prices, day, horizon)
        if r is None:
            continue
        (on if exists.loc[day] == 1 else off).append(r)
    return on, off


def test_signal(
    signal: Signal, prices: pd.Series, horizon: int, *, min_group: int = 8,
    permutations: int = 10_000, seed: int = 0, offsets: int = 10,
    outcome: Callable[..., float | None] = forward_return,
) -> TestResult:
    """One signal at one horizon (see the module docstring for the protocol).

    ``outcome`` is what is measured after each date: the forward return by default, or any
    function with its signature (the brake study uses the forward maximum drawdown).
    """
    exists = signal.series.dropna()
    window = (f"{exists.index[0].date()} → {exists.index[-1].date()}"
              if len(exists) else "—")
    on, off = _sample(signal.series, prices, horizon, 0, outcome)
    mean_on = float(np.mean(on)) if on else None
    mean_off = float(np.mean(off)) if off else None
    edge = None if mean_on is None or mean_off is None else mean_on - mean_off
    ok = len(on) >= min_group and len(off) >= min_group

    stability = None
    if ok and edge is not None and edge != 0:
        same = []
        for k in range(1, offsets + 1):
            offset = int(k * max(1.0, horizon * SESSIONS_PER_DAY / offsets))
            a, b = _sample(signal.series, prices, horizon, offset, outcome)
            if a and b:
                same.append(np.sign(np.mean(a) - np.mean(b)) == np.sign(edge))
        stability = float(np.mean(same)) if same else None

    return TestResult(
        signal=signal.key, label=signal.label, direction=signal.direction, horizon=horizon,
        window=window, n_signal=len(on), n_baseline=len(off),
        mean_signal=mean_on, mean_baseline=mean_off, edge=edge,
        pvalue=permutation_pvalue(on, off, n=permutations, seed=seed) if ok else None,
        sign_stability=stability, status="ok" if ok else "insufficient",
    )


def run_battery(built: rg.RegimeBuild, cfg: Mapping[str, Any]) -> Battery:
    """Every signal at every horizon, with BH over the testable ones."""
    horizons = [int(h) for h in cfg.get("horizons_days", [30, 90, 180])]
    alpha = float(cfg.get("fdr_alpha", 0.10))
    prices = built.benchmark_total.dropna()
    results = [
        test_signal(sig, prices, h, min_group=int(cfg.get("min_group", 8)),
                    permutations=int(cfg.get("permutations", 10_000)),
                    seed=int(cfg.get("seed", 0)), offsets=int(cfg.get("offsets", 10)))
        for sig in signals(built) for h in horizons
    ]
    testable = [r for r in results if r.status == "ok"]
    for result, (q, significant) in zip(
        testable, benjamini_hochberg([r.pvalue for r in testable], alpha)
    ):
        result.qvalue, result.significant = q, significant
    return Battery(results=results, fdr_alpha=alpha, horizons=horizons)


def discrimination(
    built: rg.RegimeBuild, key: str, *, start: str | None = None, end: str | None = None,
    correction: float = 0.10, calm: float = 0.05,
) -> dict[str, Any]:
    """Phase 3's discrimination statistic for one component within ``[start, end)``: its
    risk-off share on correction days against calm days.

    Same statistic, same thresholds as :func:`transform.regime.evaluate`, so the in-sample
    and out-of-sample figures compare like with like. Descriptive — days are autocorrelated
    and a p-value on them would overstate the evidence.
    """
    vote = built.votes[key]["vote"]
    prices = built.benchmark.dropna()
    common = vote.dropna().index.intersection(prices.index)
    if start:
        common = common[common >= pd.Timestamp(start)]
    if end:
        common = common[common < pd.Timestamp(end)]
    if len(common) == 0:
        return {"key": key, "window": None}
    # The drawdown is measured on the full price history, so a fall that started before the
    # window is still a fall inside it.
    dd = rg.drawdown(prices).loc[common]
    v = vote.loc[common]

    def share(mask: pd.Series) -> float | None:
        cast = v[mask]
        return None if cast.empty else float((cast == -1).mean())

    return {
        "key": key,
        "window": f"{common[0].date()} → {common[-1].date()}",
        "correction_days": int((dd <= -correction).sum()),
        "calm_days": int((dd >= -calm).sum()),
        "off_in_correction": share(dd <= -correction),
        "off_in_calm": share(dd >= -calm),
    }


# --- Report -----------------------------------------------------------------------------


def _pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{x * 100:+.{digits}f} %".replace(".", ",")


def _share(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f} %".replace(".", ",")


def _num(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}".replace(".", ",")


def report(battery: Battery, oos: Sequence[Mapping[str, Any]], in_sample: Mapping[str, Any],
           generated: str) -> str:
    """The battery as Markdown, including — first — what did not work."""
    ok = [r for r in battery.results if r.status == "ok"]
    significant = [r for r in ok if r.significant]
    lines = [
        f"# Validación del semáforo de régimen — {generated}",
        "",
        f"{len(battery.results)} pruebas (señal × horizonte {battery.horizons} días); "
        f"{len(ok)} con muestra suficiente; FDR Benjamini-Hochberg a q = "
        f"{_num(battery.fdr_alpha, 2)} sobre esas {len(ok)}. "
        f"**Significativas tras FDR: {len(significant)}.**",
        "",
        "| Señal | h | Ventana | n señal | n base | Media señal | Media base | Ventaja | "
        "¿Signo esperado? | p | q | Estable* |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in battery.results:
        agrees = "—" if r.agrees is None else ("sí" if r.agrees else "**no**")
        mark = " ✅" if r.significant else ""
        lines.append(
            f"| {r.label} | {r.horizon} | {r.window} | {r.n_signal} | {r.n_baseline} | "
            f"{_pct(r.mean_signal)} | {_pct(r.mean_baseline)} | {_pct(r.edge)} | {agrees} | "
            f"{_num(r.pvalue)} | {_num(r.qvalue)}{mark} | "
            f"{'—' if r.sign_stability is None else f'{r.sign_stability:.0%}'} |"
        )
    lines += [
        "",
        "\\* Porcentaje de diez rejillas con otro día de arranque cuya ventaja tiene el mismo "
        "signo. Descriptivo: una ventaja que cambia de signo al mover el arranque es ruido.",
        "",
        "## Fuera de muestra: RSP/SPY antes de 2012",
        "",
        "| Periodo | Días en corrección | Días en calma | Vota risk-off en corrección | "
        "Vota risk-off en calma |",
        "|---|---|---|---|---|",
    ]
    for row in [in_sample, *oos]:
        lines.append(
            f"| {row.get('window') or '—'} | {row.get('correction_days', '—')} | "
            f"{row.get('calm_days', '—')} | {_share(row.get('off_in_correction'))} | "
            f"{_share(row.get('off_in_calm'))} |"
        )
    lines += ["", *battery.notes]
    return "\n".join(lines)
