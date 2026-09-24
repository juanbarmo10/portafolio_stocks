"""Regime traffic light — levels 1 and 2 in one verdict (CLAUDE.md sections 2, 8 phase 3).

Section 2's hierarchy rule is the reason this exists: **if levels 1 and 2 are clearly red,
nothing is bought, however good level 3 looks.** An excellent company falls 40 % in a
risk-off market. This module turns the macro and market-structure readings into one
verdict that can block a purchase — and **only** block one. Green does not mean "buy"; it
means "the environment does not forbid it".

**Anti-overfitting is a rule here, not advice (section 9.7).** With enough indicators and
enough thresholds, some combination always would have called every past crash. So:

- Every threshold was written in config **before** any result was looked at, and was not
  adjusted afterwards (2026-09-24). A poor validation is reported, not tuned away.
- Two kinds of rule, so thresholds are not invented: **level** against a *natural*
  reference nobody has to choose (NFCI 0 is average conditions by construction; the curve
  and the volatility term structure invert at 0 and 1; a majority is 50 %), and **trend**
  against the series' own 200-session average when no natural level exists.
- Equal, fixed weights and a simple majority: red if more than half of the voting
  components say risk-off, green if more than half say risk-on, amber otherwise. There is
  no cut-off to choose.
- Every component's vote is visible. A verdict nobody can take apart is a verdict nobody
  can check.

**Point-in-time, per component.** A macro series is read as it was *published* by each
date (``ts_release``), never as it now reads for that date (section 9.4). The as-of
construction below takes, on every session, the latest reference period released so far.

Pure functions over stored frames: no network, no database handle (section 10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from transform.breadth import equal_weight_ratio, sector_breadth

RISK_ON = "risk_on"
NEUTRAL = "neutral"
RISK_OFF = "risk_off"
INSUFFICIENT = "insufficient"

LEVEL = "level"
TREND = "trend"


@dataclass(frozen=True)
class Rule:
    """How one component votes. Written in config before any result was seen."""

    key: str
    label: str
    kind: str                         # 'level' | 'trend'
    off_when: str                     # 'above' | 'below' — which side is risk-off
    reference: float | None = None    # level rules: the natural reference
    dead_zone: float | None = None    # level rules: the neutral band around it
    series: str | None = None         # a stored FRED series …
    source: str | None = None         # … or a derived one ('net_liquidity', …)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "Rule":
        rule = cls(
            key=str(cfg["key"]), label=str(cfg.get("label", cfg["key"])),
            kind=str(cfg["kind"]), off_when=str(cfg["off_when"]),
            reference=None if cfg.get("reference") is None else float(cfg["reference"]),
            dead_zone=None if cfg.get("dead_zone") is None else float(cfg["dead_zone"]),
            series=cfg.get("series"), source=cfg.get("source"),
        )
        if rule.kind not in (LEVEL, TREND):
            raise ValueError(f"{rule.key}: unknown rule kind {rule.kind!r}")
        if rule.off_when not in ("above", "below"):
            raise ValueError(f"{rule.key}: off_when must be 'above' or 'below'")
        if rule.kind == LEVEL and (rule.reference is None or rule.dead_zone is None):
            raise ValueError(f"{rule.key}: a level rule needs a reference and a dead zone")
        if (rule.series is None) == (rule.source is None):
            raise ValueError(f"{rule.key}: give exactly one of `series` or `source`")
        return rule


@dataclass(frozen=True)
class Vote:
    """One component on one date — the part of the verdict that can be checked."""

    key: str
    label: str
    value: float | None
    value_date: str | None        # the reference date of the value used
    reference: float | None       # what it was compared with (for trend: its average)
    vote: int | None              # +1 risk-on, 0 neutral, −1 risk-off, None abstains
    detail: str


@dataclass(frozen=True)
class RegimeReading:
    """The verdict on one date, with every vote behind it."""

    date: str
    verdict: str
    on: int
    off: int
    neutral: int
    available: int
    votes: list[Vote]


# --- Building each component as it was known on each date ---------------------------


def asof(observations: pd.DataFrame, series_id: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """A stored series as it was **published** by each date: ``[value, value_date]``.

    On every session, the latest reference period released so far — and, for that period,
    the latest version released so far. A revision of an older period does not move the
    reading; a revision of the current one replaces it from its own release date. Rows with
    an unknown release date cannot be placed in time and are left out rather than guessed.
    """
    empty = pd.DataFrame({"value": np.nan, "value_date": pd.NaT}, index=calendar)
    if observations is None or observations.empty:
        return empty
    rows = observations[
        (observations["series_id"] == series_id) & observations["value"].notna()
        & (observations["ts_release"].fillna("").str.len() >= 10)
    ]
    if rows.empty:
        return empty

    frame = pd.DataFrame({
        "ts": pd.to_datetime(rows["ts"].str[:10]),
        "released": pd.to_datetime(rows["ts_release"].str[:10]),
        "value": rows["value"].astype(float),
    }).sort_values(["released", "ts"])
    frame["frontier"] = frame["ts"].cummax()
    frontier = frame[frame["ts"] == frame["frontier"]]
    per_release = frontier.groupby("released", as_index=False).last()

    merged = pd.merge_asof(
        pd.DataFrame({"date": calendar}), per_release[["released", "ts", "value"]],
        left_on="date", right_on="released", direction="backward",
    )
    return pd.DataFrame(
        {"value": merged["value"].to_numpy(), "value_date": merged["ts"].to_numpy()},
        index=calendar,
    )


def _combine(frames: Sequence[pd.DataFrame], values: pd.Series) -> pd.DataFrame:
    """A derived component: its value, dated by the OLDEST input it rests on."""
    dates = pd.concat([f["value_date"] for f in frames], axis=1).min(axis=1)
    return pd.DataFrame({"value": values, "value_date": dates})


def component_frames(
    rules: Sequence[Rule],
    fred: pd.DataFrame,
    closes: pd.DataFrame,
    splits: Mapping[str, Mapping[Any, float]],
    calendar: pd.DatetimeIndex,
    *,
    sectors: Sequence[str],
    equal: str = "RSP",
    cap: str = "SPY",
    breadth_window: int = 200,
    min_window_fraction: float = 0.95,
) -> dict[str, pd.DataFrame]:
    """Every component's ``[value, value_date]`` on the trading calendar."""
    out: dict[str, pd.DataFrame] = {}
    for rule in rules:
        if rule.series:
            out[rule.key] = asof(fred, rule.series, calendar)
        elif rule.source == "net_liquidity":
            parts = [asof(fred, s, calendar) for s in ("WALCL", "WTREGEN", "WLRRAL")]
            # All three from the same weekly H.4.1 release, in the same unit (millions).
            value = parts[0]["value"] - parts[1]["value"] - parts[2]["value"]
            out[rule.key] = _combine(parts, value)
        elif rule.source == "vix_term":
            parts = [asof(fred, s, calendar) for s in ("VIXCLS", "VXVCLS")]
            out[rule.key] = _combine(parts, parts[0]["value"] / parts[1]["value"])
        elif rule.source == "sector_breadth":
            sb = sector_breadth(closes, splits, sectors, window=breadth_window,
                                min_window_fraction=min_window_fraction)
            out[rule.key] = _price_component(sb, "share", calendar)
        elif rule.source == "equal_weight":
            ew = equal_weight_ratio(closes, splits, equal=equal, cap=cap)
            out[rule.key] = _price_component(ew, "ratio", calendar)
        else:
            raise ValueError(f"{rule.key}: unknown source {rule.source!r}")
    return out


def _price_component(frame: pd.DataFrame, column: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """A daily price-built measure: known at the close of its own session."""
    if frame.empty:
        return pd.DataFrame({"value": np.nan, "value_date": pd.NaT}, index=calendar)
    series = pd.Series(
        pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(frame["date"]),
    ).reindex(calendar)
    dates = pd.Series(calendar, index=calendar).where(series.notna())
    return pd.DataFrame({"value": series, "value_date": dates})


# --- Votes ------------------------------------------------------------------------------


def votes(
    component: pd.DataFrame,
    rule: Rule,
    *,
    window: int = 200,
    min_window_fraction: float = 0.95,
    trend_dead_zone_sd: float = 0.25,
    max_staleness_days: int = 30,
) -> pd.DataFrame:
    """``[vote, reference, band]`` per session for one component.

    A level rule compares with its natural reference; a trend rule with the component's own
    trailing average, the neutral band being ``trend_dead_zone_sd`` standard deviations.
    The component **abstains** (``None``) without a value, without a full enough window,
    or when its latest value is older than ``max_staleness_days`` — a dead series must not
    keep voting with its last word.
    """
    value = component["value"].astype(float)
    if rule.kind == LEVEL:
        reference = pd.Series(rule.reference, index=value.index, dtype=float)
        band = pd.Series(rule.dead_zone, index=value.index, dtype=float)
    else:
        periods = max(1, math.ceil(window * min_window_fraction))
        reference = value.rolling(window, min_periods=periods).mean()
        band = value.rolling(window, min_periods=periods).std() * trend_dead_zone_sd

    above = value > reference + band
    below = value < reference - band
    off, on = (above, below) if rule.off_when == "above" else (below, above)
    vote = pd.Series(0.0, index=value.index).mask(off, -1.0).mask(on, 1.0)

    age = (pd.Series(value.index, index=value.index)
           - pd.to_datetime(component["value_date"])).dt.days
    abstain = value.isna() | reference.isna() | band.isna() | (age > max_staleness_days)
    vote = vote.mask(abstain)
    return pd.DataFrame({"vote": vote, "reference": reference, "band": band})


def regime_frame(
    all_votes: Mapping[str, pd.DataFrame], *, min_components: int = 4
) -> pd.DataFrame:
    """Per session: every component's vote, the counts, and the verdict.

    Equal weights, simple majority. Fewer than ``min_components`` voting → no verdict.
    """
    grid = pd.DataFrame({key: frame["vote"] for key, frame in all_votes.items()})
    available = grid.notna().sum(axis=1)
    on = (grid == 1).sum(axis=1)
    off = (grid == -1).sum(axis=1)
    neutral = (grid == 0).sum(axis=1)

    verdict = pd.Series(NEUTRAL, index=grid.index)
    verdict = verdict.mask(on > available / 2, RISK_ON)
    verdict = verdict.mask(off > available / 2, RISK_OFF)
    verdict = verdict.mask(available < min_components, INSUFFICIENT)
    return grid.assign(on=on, off=off, neutral=neutral, available=available, verdict=verdict)


def reading_at(
    date: Any,
    rules: Sequence[Rule],
    components: Mapping[str, pd.DataFrame],
    all_votes: Mapping[str, pd.DataFrame],
    frame: pd.DataFrame,
) -> RegimeReading | None:
    """The verdict on (or before) ``date``, with each component's reason in Spanish."""
    if frame.empty:
        return None
    day = frame.index[frame.index <= pd.Timestamp(date)]
    if len(day) == 0:
        return None
    day = day[-1]
    row = frame.loc[day]

    out: list[Vote] = []
    for rule in rules:
        comp, vt = components[rule.key].loc[day], all_votes[rule.key].loc[day]
        value = None if pd.isna(comp["value"]) else float(comp["value"])
        reference = None if pd.isna(vt["reference"]) else float(vt["reference"])
        vote = None if pd.isna(vt["vote"]) else int(vt["vote"])
        value_date = None if pd.isna(comp["value_date"]) else pd.Timestamp(
            comp["value_date"]).date().isoformat()
        out.append(Vote(rule.key, rule.label, value, value_date, reference, vote,
                        _detail(rule, value, reference, vote)))
    return RegimeReading(
        date=day.date().isoformat(), verdict=str(row["verdict"]),
        on=int(row["on"]), off=int(row["off"]), neutral=int(row["neutral"]),
        available=int(row["available"]), votes=out,
    )


def _detail(rule: Rule, value: float | None, reference: float | None, vote: int | None) -> str:
    if vote is None:
        return "Se abstiene: sin dato reciente o sin ventana suficiente."
    side = {1: "risk-on", 0: "neutro", -1: "risk-off"}[vote]
    if rule.kind == LEVEL:
        return (f"{side}: nivel frente a su referencia natural {rule.reference:g} "
                f"(zona neutra ±{rule.dead_zone:g}).")
    trend = "por encima de" if value is not None and reference is not None \
        and value > reference else "por debajo de"
    return f"{side}: {trend} su media de 200 sesiones."


# --- Validation against the market's own record -----------------------------------------


def drawdown(prices: pd.Series) -> pd.Series:
    """Fall from the running peak: 0 at a new high, −0.10 ten percent below it."""
    return prices / prices.cummax() - 1.0


def evaluate(
    frame: pd.DataFrame,
    prices: pd.Series,
    *,
    correction: float = 0.10,
    calm: float = 0.05,
) -> dict[str, Any]:
    """How the verdicts line up with the S&P 500's corrections — descriptive, not a test.

    A correction is the market's standard definition: 10 % below the prior peak. It is
    chosen for being conventional, not for flattering the result. Descriptive on purpose:
    the forward-return study with multiple-comparison correction is phase 4.

    Returns shares of red / amber / green on correction days and on calm days (within 5 %
    of the peak), per-component off-vote shares on each, and one row per episode.
    """
    common = frame.index.intersection(prices.dropna().index)
    fr = frame.loc[common]
    dd = drawdown(prices.loc[common])
    judged = fr["verdict"] != INSUFFICIENT
    in_correction = (dd <= -correction) & judged
    is_calm = (dd >= -calm) & judged

    def share(mask: pd.Series, verdict: str) -> float | None:
        n = int(mask.sum())
        return None if n == 0 else float((fr.loc[mask, "verdict"] == verdict).sum() / n)

    components = [c for c in fr.columns
                  if c not in ("on", "off", "neutral", "available", "verdict")]
    per_component = {}
    for key in components:
        col = fr[key]
        per_component[key] = {
            "off_in_correction": _off_share(col[in_correction]),
            "off_in_calm": _off_share(col[is_calm]),
        }

    return {
        "correction_days": int(in_correction.sum()),
        "calm_days": int(is_calm.sum()),
        "red_in_correction": share(in_correction, RISK_OFF),
        "amber_in_correction": share(in_correction, NEUTRAL),
        "green_in_correction": share(in_correction, RISK_ON),
        "red_in_calm": share(is_calm, RISK_OFF),
        "green_in_calm": share(is_calm, RISK_ON),
        "per_component": per_component,
        "episodes": _episodes(fr, dd, correction),
    }


def _off_share(votes_: pd.Series) -> float | None:
    cast = votes_.dropna()
    return None if cast.empty else float((cast == -1).mean())


def _episodes(fr: pd.DataFrame, dd: pd.Series, correction: float) -> list[dict[str, Any]]:
    """Each fall of at least ``correction`` from a peak, and what the light said in it."""
    episodes: list[dict[str, Any]] = []
    at_peak = dd == 0
    peaks = list(dd.index[at_peak])
    for i, peak in enumerate(peaks):
        end = peaks[i + 1] if i + 1 < len(peaks) else dd.index[-1]
        span = dd.loc[peak:end]
        if span.min() > -correction:
            continue
        trough = span.idxmin()
        falling = fr.loc[peak:trough]
        judged = falling[falling["verdict"] != INSUFFICIENT]
        red = judged.index[judged["verdict"] == RISK_OFF]
        episodes.append({
            "peak": peak.date().isoformat(),
            "trough": trough.date().isoformat(),
            "depth": float(span.min()),
            "first_red": red[0].date().isoformat() if len(red) else None,
            "red_share_to_trough": (float(len(red) / len(judged)) if len(judged) else None),
            # Over the whole fall, not only the judged days: an episode with no verdict still
            # had components voting, just fewer than the minimum — and saying so is the point.
            "components_voting": int(falling["available"].median()) if len(falling) else 0,
            "judged": bool(len(judged)),
        })
    return episodes


def rules_from_config(cfg: Mapping[str, Any]) -> list[Rule]:
    """The configured components, validated at load: a malformed rule fails loudly."""
    return [Rule.from_config(c) for c in cfg.get("components", [])]


def verdict_label(verdict: str) -> str:
    """Spanish label for the panel."""
    return {
        RISK_ON: "🟢 Risk-on — el entorno no bloquea",
        NEUTRAL: "🟡 Mixto — sin mayoría clara",
        RISK_OFF: "🔴 Risk-off — no se compra (§2)",
        INSUFFICIENT: "⚪ Insuficiente — menos de la mitad de los componentes tiene dato",
    }.get(verdict, verdict)
