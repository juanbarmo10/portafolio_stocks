"""Sell discipline and "reinforce before opening" — 📋 Desplegar (CLAUDE.md §15.5, point 1).

The behaviour mirror (``transform/behavior.py``) measured the account's pattern: winners
sold within weeks and losers kept, which is the disposition effect, and the one habit that
rules out the few large winners a growth portfolio lives on. The mirror shows it after the
fact, on the journal page. This module puts the same evidence **at the moment of deciding**:

**Before selling** (:func:`before_selling`): what the user wrote for that position against
what is true today — whether the structured invalidation rule has been crossed, which exit
rules have triggered, how long the position has been held against the written horizon, and
the open result. It returns whether a **written rule backs the sale**. It never says "sell"
or "hold": a sale with no written backing may still be right, and the user decides
(sections 2, 12). The panel cannot block an order in IBKR; it can make the pattern visible.

**Reinforce before opening** (:func:`reinforce`): the held positions ordered by whether the
thesis is intact and its own metric is moving **away** from the invalidation threshold since
the previous filing. The metric is the one the user chose in ``invalidation_rule``: the
panel never picks which figure matters, and the criterion is the thesis, **never the
price** — "it went up" is not a reason to add and "it went down" is not a reason to average
down.

Unknown stays unknown (section 12): a position with no card, no structured rule or no SEC
figures (an IFRS filer such as NU) is reported as not evaluable, never as intact.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from transform import thesis as th

# Operators whose breach is a value that is too LOW: moving up is moving away from it.
_LOW_IS_BAD = frozenset({"<", "<="})

BACKED = "rule"           # a written rule backs the sale
UNBACKED = "none"         # card written, nothing in it backs the sale
NO_CARD = "no_card"       # no card: there is no written criterion to sell or to keep


def position_metrics(valued: pd.DataFrame | None, ticker: str) -> dict[str, float | None]:
    """``price``, ``weight`` and ``return_pct`` of one held position — the metrics an
    ``exit_ladder`` rule may use besides the fundamentals. ``weight`` is over the priced
    stocks, as in ``portfolio.valuation``."""
    empty = {"price": None, "weight": None, "return_pct": None}
    if valued is None or valued.empty:
        return empty
    rows = valued[valued["ticker"] == ticker]
    if rows.empty:
        return empty
    row = rows.iloc[0]

    def clean(value: Any) -> float | None:
        return None if pd.isna(value) else float(value)

    return {"price": clean(row["price"]), "weight": clean(row["weight"]),
            "return_pct": clean(row["unrealized_return"])}


@dataclass(frozen=True)
class SellCheck:
    """Everything written about one position, against today (see :func:`before_selling`).

    Attributes:
        backing: ``BACKED`` (the invalidation rule is crossed or an exit rule triggered),
            ``UNBACKED`` (a card exists and nothing in it is met) or ``NO_CARD``.
        invalidation: The written criterion, verbatim ('' without a card).
        invalidation_breached: ``True`` (crossed: bad news), ``False``, ``None`` unknown.
        exit_triggered: Structured exit rules met today: ``rule_id, kind, trigger,
            action, detail``.
        exit_prose: Exit rules written only in words — the user judges those.
        exit_unevaluable: ``rule_id`` of structured exit rules that could not be evaluated.
        held_days: Days since the **oldest** open lot; ``newest_lot_days`` for the newest.
        unrealized_return: Open result over FIFO cost.
        reasons: Why ``backing`` is what it is, in Spanish, for the panel.
    """

    ticker: str
    backing: str
    has_card: bool
    invalidation: str
    invalidation_breached: bool | None
    invalidation_detail: str
    exit_rules: int
    exit_triggered: list[dict[str, Any]] = field(default_factory=list)
    exit_prose: list[dict[str, Any]] = field(default_factory=list)
    exit_unevaluable: list[str] = field(default_factory=list)
    held_days: int | None = None
    newest_lot_days: int | None = None
    min_holding_days: int = 90
    unrealized_return: float | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def before_horizon(self) -> bool | None:
        """Held for less than the written minimum horizon (``None`` = unknown)."""
        return None if self.held_days is None else self.held_days < self.min_holding_days


def _lot_ages(open_lots: pd.DataFrame | None, ticker: str,
              as_of: pd.Timestamp) -> tuple[int | None, int | None]:
    if open_lots is None or open_lots.empty:
        return None, None
    days = pd.to_datetime(open_lots.loc[open_lots["ticker"] == ticker, "ts"].astype(str)
                          .str[:10])
    if days.empty:
        return None, None
    return int((as_of - days.min()).days), int((as_of - days.max()).days)


def before_selling(
    ticker: str,
    card: Mapping[str, Any] | None,
    fundamentals: Mapping[str, float | None],
    valued: pd.DataFrame | None,
    open_lots: pd.DataFrame | None,
    as_of: Any,
    *,
    min_holding_days: int = 90,
) -> SellCheck:
    """What the written plan says about selling ``ticker`` today.

    ``card`` is the thesis card (``None`` if the position has none). ``fundamentals`` is
    ``thesis.evaluable_metrics`` for its CIK — empty for a company with no SEC figures.
    Exit rules are evaluated on those plus :func:`position_metrics`, exactly as the alert
    ``exit_ladder_triggered`` does, so the page and Telegram never disagree.
    """
    day = pd.Timestamp(as_of).normalize()
    held, newest = _lot_ages(open_lots, ticker, day)
    position = position_metrics(valued, ticker)
    common = dict(ticker=ticker, held_days=held, newest_lot_days=newest,
                  min_holding_days=min_holding_days, unrealized_return=position["return_pct"])
    if not card:
        return SellCheck(
            backing=NO_CARD, has_card=False, invalidation="", invalidation_breached=None,
            invalidation_detail="", exit_rules=0,
            reasons=["Sin ficha de tesis: no hay criterio escrito ni para vender ni para "
                     "mantener. La decisión se tomaría por impulso, que es justo lo que la "
                     "ficha existe para evitar (§5.2)."],
            **common)

    breached, detail = th.evaluate_rule(card.get("invalidation_rule"), fundamentals)
    metrics = {**fundamentals, **position}
    ladder = list(card.get("exit_ladder") or [])
    triggered: list[dict[str, Any]] = []
    prose: list[dict[str, Any]] = []
    unevaluable: list[str] = []
    for rule in ladder:
        if not rule.get("rule"):
            prose.append(dict(rule))
            continue
        hit, rule_detail = th.evaluate_rule(rule["rule"], metrics)
        if hit:
            triggered.append({k: rule.get(k) for k in ("rule_id", "kind", "trigger", "action")}
                             | {"detail": rule_detail})
        elif hit is None:
            unevaluable.append(str(rule.get("rule_id")))

    reasons: list[str] = []
    if breached:
        reasons.append(f"Tu criterio de invalidación está cruzado: {detail}")
    for rule in triggered:
        reasons.append(f"Regla de salida `{rule['rule_id']}` ({rule['kind']}) alcanzada: "
                       f"«{rule['action']}».")
    backed = bool(breached) or bool(triggered)
    if not backed:
        reasons.append("Nada de lo que escribiste para esta posición respalda vender hoy: "
                       "la regla de invalidación no está cruzada"
                       + (" (o no se puede evaluar)" if breached is None else "")
                       + " y ninguna regla de salida estructurada se ha alcanzado.")
        if prose:
            reasons.append(f"{len(prose)} regla(s) de salida escritas solo en texto: esas "
                           "las juzgas tú, releyéndolas abajo.")
    return SellCheck(
        backing=BACKED if backed else UNBACKED, has_card=True,
        invalidation=str(card.get("invalidation") or ""), invalidation_breached=breached,
        invalidation_detail=detail, exit_rules=len(ladder), exit_triggered=triggered,
        exit_prose=prose, exit_unevaluable=unevaluable, reasons=reasons, **common)


# --- Reinforce before opening -------------------------------------------------------------

REINFORCE_COLUMNS = ["ticker", "status", "metric", "operator", "threshold", "value_now",
                     "value_before", "trend", "headroom", "weight", "room", "note"]

# Order of the table: what the thesis supports adding to first, what it argues against last.
STATUS_ORDER = ("improving", "flat", "unknown", "worsening", "at_limit", "no_card", "breached")


def _trend(operator: str, now: float | None, before: float | None) -> str | None:
    if now is None or before is None:
        return None
    if abs(now - before) <= 1e-9 * max(1.0, abs(before)):
        return "flat"
    up = now > before
    return "improving" if up == (operator in _LOW_IS_BAD) else "worsening"


def _headroom(operator: str, value: float | None, threshold: float) -> float | None:
    """Relative distance to the threshold, positive on the safe side. ``None`` with a zero
    threshold: a relative distance to zero is not defined (section 12)."""
    if value is None or threshold == 0:
        return None
    gap = (value - threshold) if operator in _LOW_IS_BAD else (threshold - value)
    return gap / abs(threshold)


def reinforce(
    held: Sequence[str],
    cards: Mapping[str, Mapping[str, Any]],
    ciks: Mapping[str, str],
    observations: pd.DataFrame,
    valued: pd.DataFrame | None,
    as_of: Any,
    *,
    lookback_days: int = 100,
    hurdle_rate: float | None = None,
    nav_total: float | None = None,
    max_position: float | None = None,
) -> pd.DataFrame:
    """Held positions, ordered by what the thesis says about adding to them.

    For each ticker in ``held``: its ``invalidation_rule`` metric today and
    ``lookback_days`` earlier, both **point-in-time** (only figures filed by each date,
    section 9.4) — 100 days so a quarterly filing falls in between. ``trend`` is
    ``improving`` when the metric moved away from the threshold. ``weight`` is over the
    whole account (``nav_total``, cash included, the basis of ``max_position``) and
    ``room`` is what is left under ``max_position``.

    ``status``: ``improving`` · ``flat`` (no new filing, or no change) · ``unknown`` (no
    structured rule, or no figure) · ``worsening`` · ``at_limit`` (no room under
    ``max_position``) · ``no_card`` · ``breached``. The order is :data:`STATUS_ORDER`, and
    it is the only opinion expressed: the choice is the user's.
    """
    day = pd.Timestamp(as_of).normalize()
    before_day = (day - pd.Timedelta(days=lookback_days)).date().isoformat()
    rows = []
    for ticker in held:
        card = cards.get(ticker)
        position = position_metrics(valued, ticker)
        weight = None
        if valued is not None and not valued.empty and nav_total:
            value = valued.loc[valued["ticker"] == ticker, "market_value"]
            if not value.empty and pd.notna(value.iloc[0]):
                weight = float(value.iloc[0]) / nav_total
        room = None if max_position is None or weight is None else float(max_position) - weight
        row = dict(ticker=ticker, status="no_card", metric=None, operator=None, threshold=None,
                   value_now=None, value_before=None, trend=None, headroom=None,
                   weight=weight if nav_total else position["weight"], room=room, note="")
        if not card:
            row["note"] = "Sin ficha de tesis: nada escrito que diga si reforzar."
            rows.append(row)
            continue
        rule = card.get("invalidation_rule") or {}
        cik = str(ciks.get(ticker) or card.get("cik") or "").zfill(10)
        now = th.evaluable_metrics(observations, cik, day, hurdle_rate) if cik.strip("0") \
            else {}
        breached, detail = th.evaluate_rule(rule or None, now)
        metric, operator = rule.get("metric"), str(rule.get("operator", ""))
        row.update(metric=metric, operator=operator or None, threshold=rule.get("threshold"))
        if breached:
            row.update(status="breached", value_now=now.get(metric),
                       note="Criterio de invalidación CRUZADO: reforzar iría contra lo que "
                            "escribiste.")
            rows.append(row)
            continue
        if breached is None:
            row.update(status="unknown", value_now=now.get(metric) if metric else None,
                       note=detail.replace("**", ""))
        else:
            before = th.evaluable_metrics(observations, cik, before_day, hurdle_rate)
            value_now, value_before = now.get(metric), before.get(metric)
            trend = _trend(operator, value_now, value_before)
            row.update(value_now=value_now, value_before=value_before, trend=trend,
                       headroom=_headroom(operator, value_now, float(rule["threshold"])),
                       status=trend or "unknown",
                       note="" if trend else "Sin cifra de hace un trimestre para comparar.")
        if room is not None and room <= 0:
            row.update(status="at_limit",
                       note=f"Ya pesa {weight:.1%} con un tope de {float(max_position):.0%}."
                       .replace(".", ","))
        rows.append(row)
    frame = pd.DataFrame(rows, columns=REINFORCE_COLUMNS)
    if frame.empty:
        return frame
    rank = {s: i for i, s in enumerate(STATUS_ORDER)}
    return frame.assign(_rank=frame["status"].map(rank)).sort_values(
        ["_rank", "ticker"], kind="stable").drop(columns="_rank").reset_index(drop=True)
