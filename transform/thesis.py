"""Thesis invalidation board — level 3, point 7 (CLAUDE.md sections 5.2, 8 phase 2).

The rule this implements is the one that makes the whole universe honest: **no company
enters without a written criterion that would kill the thesis**. Without falsification
there is no way to know when to sell, and a position sustains itself by inertia.

``invalidation`` is free text by design (section 5.2), and free text cannot be evaluated by
a program. So this module does two separate things and never confuses them:

**It shows the written criterion verbatim, for a human to judge.** That is the point of
having written it in the first place. Nothing here paraphrases it, scores it or decides it
has been met.

**It runs the checks that really are mechanical**, and those it does decide:

- the ``review_date`` has passed;
- earnings are within the window section 2 makes a hard rule ("no position is opened in the
  five days before results without an explicit decision");
- the company filed an amended financial statement — a governance flag (section 9.6);
- the fundamentals it would be judged on are stale or missing.

**Optionally**, a card may add a structured ``invalidation_rule`` next to the prose:
``{metric: cash_conversion, operator: "<", threshold: 0.6}``. When present it is evaluated
against the point-in-time figures and its verdict is reported alongside the text. It never
replaces the prose, and a rule naming a metric this project cannot compute reports that
plainly instead of quietly passing.

A verdict of ``None`` means *unknown*, and unknown is never rendered as "fine" (section 12).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from transform import fundamentals as fun
from transform import value_accrual as va

# Comparisons a structured rule may use. Deliberately four: a rule needing more expressive
# power than this is a rule that belongs in prose, where a human reads it (section 9.7 —
# every extra free parameter is another way to fit noise).
OPERATORS = {
    "<": lambda value, threshold: value < threshold,
    "<=": lambda value, threshold: value <= threshold,
    ">": lambda value, threshold: value > threshold,
    ">=": lambda value, threshold: value >= threshold,
}

# Days before results inside which section 2 forbids opening a position without an
# explicit decision. Config overrides it; this is the documented default.
EARNINGS_WINDOW_DAYS = 5

FINANCIAL_FORMS = frozenset({"10-K", "10-Q", "20-F", "40-F", "6-K"})


@dataclass(frozen=True)
class ThesisStatus:
    """One company's thesis and everything mechanically knowable about its health.

    Attributes:
        ticker / cik: Identity. The CIK is the key (section 9.3).
        thesis / value_accrual / key_metric / invalidation: The written card, verbatim.
        review_date: When the thesis is due for review.
        review_overdue: Whether that date has passed.
        earnings_date / earnings_is_estimated / days_to_earnings: Next results. The
            estimated flag survives to here because an estimated date must never be shown
            as a confirmed one (RESEARCH.md section 2.14).
        earnings_window: Inside the section-2 blackout before results.
        amended_filings: Amended financial filings — a possible restatement (section 9.6).
        metrics: The point-in-time figures the rule could be evaluated against.
        rule: The structured criterion, if the card carries one.
        rule_breached: ``True`` (thesis criterion met, i.e. **bad news**), ``False``, or
            ``None`` when it cannot be evaluated.
        rule_detail: Why, in Spanish, for the panel.
        flags: Everything mechanical that deserves attention, in plain language.
    """

    ticker: str
    cik: str
    thesis: str
    value_accrual: str
    key_metric: str
    invalidation: str
    review_date: str | None
    review_overdue: bool
    earnings_date: str | None
    earnings_is_estimated: bool
    days_to_earnings: int | None
    earnings_window: bool
    amended_filings: list[dict[str, Any]]
    metrics: dict[str, float | None]
    rule: Mapping[str, Any] | None
    rule_breached: bool | None
    rule_detail: str
    flags: list[str] = field(default_factory=list)


def evaluable_metrics(
    observations: pd.DataFrame, cik: str, as_of: Any, hurdle_rate: float | None = None
) -> dict[str, float | None]:
    """Every figure a structured rule may be written against, point-in-time.

    Keeping the list explicit is the point: a rule can only name something this project
    actually computes, so a typo surfaces as "unknown metric" instead of as silence.
    """
    snapshot = fun.snapshot(observations, cik, as_of)
    accrual = va.assess(observations, cik, as_of, hurdle_rate=hurdle_rate)
    return {
        "revenue_ttm": snapshot.revenue_ttm,
        "revenue_growth_yoy": snapshot.revenue_growth_yoy,
        "net_income_ttm": snapshot.net_income_ttm,
        "operating_margin": snapshot.operating_margin,
        "net_margin": snapshot.net_margin,
        "fcf_ttm": snapshot.fcf_ttm,
        "fcf_margin": snapshot.fcf_margin,
        "cash_conversion": snapshot.cash_conversion,
        "dilution_yoy": accrual.dilution_yoy,
        "sbc_over_revenue": accrual.sbc_over_revenue,
        "sbc_over_fcf": accrual.sbc_over_fcf,
        "net_buybacks_ttm": accrual.net_buybacks_ttm,
        "roic": accrual.roic,
        "long_term_debt": snapshot.long_term_debt,
        "equity": snapshot.equity,
        "cash": snapshot.cash,
    }


def evaluate_rule(
    rule: Mapping[str, Any] | None, metrics: Mapping[str, float | None]
) -> tuple[bool | None, str]:
    """Evaluate a structured invalidation rule.

    Returns ``(breached, detail)`` where ``breached`` is ``True`` when the criterion the
    user wrote as fatal has been met — that is, **bad news** — and ``None`` when the rule
    cannot be evaluated at all. Unknown is reported as unknown; a rule that silently passed
    because its metric was missing would be worse than having no rule (section 12).
    """
    if not rule:
        return None, "Sin regla estructurada: el criterio escrito lo juzgas tú."

    metric = str(rule.get("metric", ""))
    operator = str(rule.get("operator", ""))
    threshold = rule.get("threshold")

    if metric not in metrics:
        return None, (
            f"La regla nombra `{metric}`, que este panel no calcula. Métricas "
            f"disponibles: {', '.join(sorted(metrics))}."
        )
    if operator not in OPERATORS:
        return None, f"Operador `{operator}` no reconocido. Usa uno de: {', '.join(OPERATORS)}."
    if threshold is None:
        return None, "La regla no tiene umbral."

    value = metrics[metric]
    if value is None:
        return None, f"`{metric}` no tiene valor publicado todavía; no se puede evaluar."

    breached = OPERATORS[operator](float(value), float(threshold))
    verdict = "CRUZADO" if breached else "no cruzado"
    return breached, f"{metric} = {value:,.4g} {operator} {threshold} → **{verdict}**."


def _days_until(target: str | None, as_of: str) -> int | None:
    if not target:
        return None
    try:
        return (pd.Timestamp(target) - pd.Timestamp(as_of)).days
    except (ValueError, TypeError):
        return None


def status(
    card: Mapping[str, Any],
    observations: pd.DataFrame,
    filings: Sequence[Mapping[str, Any]] | pd.DataFrame | None,
    events: Sequence[Mapping[str, Any]] | pd.DataFrame | None,
    as_of: Any,
    *,
    hurdle_rate: float | None = None,
    earnings_window_days: int = EARNINGS_WINDOW_DAYS,
) -> ThesisStatus:
    """Assemble one company's board row. Pure: everything comes in as data."""
    as_of_iso = pd.Timestamp(as_of).date().isoformat()
    cik = str(card.get("cik", "")).zfill(10) if card.get("cik") else ""
    ticker = str(card.get("ticker", "?"))

    metrics = evaluable_metrics(observations, cik, as_of, hurdle_rate) if cik else {}
    rule = card.get("invalidation_rule")
    breached, detail = evaluate_rule(rule, metrics)

    review_date = str(card["review_date"]) if card.get("review_date") else None
    days_to_review = _days_until(review_date, as_of_iso)
    review_overdue = days_to_review is not None and days_to_review < 0

    earnings_date, estimated, days_to_earnings = None, False, None
    events_frame = _as_frame(events)
    if not events_frame.empty and cik:
        upcoming = events_frame[
            (events_frame["cik"] == cik)
            & (events_frame["category"] == "earnings")
            & (events_frame["ts"] >= as_of_iso)
        ].sort_values("ts")
        if not upcoming.empty:
            row = upcoming.iloc[0]
            earnings_date = str(row["ts"])
            estimated = bool(row.get("is_estimated", 0))
            days_to_earnings = _days_until(earnings_date, as_of_iso)

    amended: list[dict[str, Any]] = []
    filings_frame = _as_frame(filings)
    if not filings_frame.empty and cik:
        rows = filings_frame[
            (filings_frame["cik"] == cik) & (filings_frame["is_amended"] == 1)
        ]
        amended = [
            row for row in rows.to_dict("records")
            if str(row["form"])[:-2].upper() in FINANCIAL_FORMS
        ]

    in_window = days_to_earnings is not None and 0 <= days_to_earnings <= earnings_window_days

    flags: list[str] = []
    if review_overdue:
        flags.append(
            f"Revisión vencida hace {abs(days_to_review)} días "
            f"({review_date}). Una tesis sin revisar se sostiene por inercia."
        )
    if in_window:
        flags.append(
            f"Resultados en {days_to_earnings} días"
            + (" (fecha estimada)" if estimated else " (fecha confirmada)")
            + f". §2: no se abre posición dentro de {earnings_window_days} días sin "
            "decisión explícita."
        )
    if amended:
        flags.append(
            "Estados financieros enmendados: "
            + ", ".join(f"{f['form']} ({f['filed_date']})" for f in amended[:3])
            + ". Posible reexpresión — bandera de gobernanza a investigar (§9.6)."
        )
    if breached:
        flags.append(f"Criterio de invalidación CRUZADO: {detail}")
    if cik and metrics.get("revenue_ttm") is None:
        flags.append(
            "Sin fundamentales calculables todavía: faltan filings ingeridos o cuatro "
            "trimestres contiguos. No se estima nada (§12)."
        )

    return ThesisStatus(
        ticker=ticker,
        cik=cik,
        thesis=str(card.get("thesis", "")),
        value_accrual=str(card.get("value_accrual", "")),
        key_metric=str(card.get("key_metric", "")),
        invalidation=str(card.get("invalidation", "")),
        review_date=review_date,
        review_overdue=review_overdue,
        earnings_date=earnings_date,
        earnings_is_estimated=estimated,
        days_to_earnings=days_to_earnings,
        earnings_window=in_window,
        amended_filings=amended,
        metrics=metrics,
        rule=rule,
        rule_breached=breached,
        rule_detail=detail,
        flags=flags,
    )


def board(
    cards: Sequence[Mapping[str, Any]],
    observations: pd.DataFrame,
    filings: Sequence[Mapping[str, Any]] | pd.DataFrame | None = None,
    events: Sequence[Mapping[str, Any]] | pd.DataFrame | None = None,
    as_of: Any = None,
    *,
    hurdle_rate: float | None = None,
    earnings_window_days: int = EARNINGS_WINDOW_DAYS,
) -> list[ThesisStatus]:
    """The whole board, companies with something flagged first.

    Ordering is the only opinion this module expresses about priority, and it is a weak
    one: what needs attention floats up. It does not score theses or rank companies —
    that would be a verdict, and the verdict is the user's (sections 2, 12).
    """
    as_of = as_of or pd.Timestamp.today().date().isoformat()
    rows = [
        status(card, observations, filings, events, as_of,
               hurdle_rate=hurdle_rate, earnings_window_days=earnings_window_days)
        for card in cards or []
    ]
    return sorted(rows, key=lambda row: (-len(row.flags), row.ticker))


def _as_frame(data: Sequence[Mapping[str, Any]] | pd.DataFrame | None) -> pd.DataFrame:
    """Accept records or a frame, so callers can pass whatever the database handed them."""
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(list(data))
