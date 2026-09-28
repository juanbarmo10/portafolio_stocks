"""Scenarios and size by tolerable loss (🏢 Empresa, 📋 Desplegar; CLAUDE.md §15.5, point 3).

A growth portfolio lives on a few positions that multiply, and dies on sizing one wrong
thesis as if it could not fail. This module turns what the user writes in the card —
three scenarios with their probabilities — into the numbers a sizing decision needs:
what the position is expected to return, how likely it is to lose money, how much it can
lose, and therefore how large it can be for a loss the user has decided to accept.

**Every number comes from the card; the panel invents none.** A scenario is one of:

- ``growth``: the yearly growth of free cash flow **per share** for 10 years; its annual
  return comes from the implied-return engine (``transform.valuation.implied_return``) at
  today's price. Needs positive FCF and SEC figures.
- ``multiple``: how many times today's price the share is worth after ``years``. Works for
  any company — one that burns cash (IBRX) or one the SEC cannot read (NU).

Every scenario is brought to the same horizon (``years``) as a **wealth multiple**. The
expected return is the annualized **expected wealth** — ``(Σ p·M)^(1/years) − 1`` — never
the probability-weighted average of annual returns, which misstates it for the same reason
an arithmetic mean of growth rates does (section 9.10).

Sizing, three ceilings, none of them a target:

- **Loss budget**: the share of the whole account the user accepts losing on one thesis
  (``portfolio.loss_budget``). Divided by the worst written loss, it gives the largest size
  at which the bear case costs exactly that; divided by 1, the size if the thesis goes to
  zero.
- **Fractional Kelly**: the fraction that maximizes the expected logarithm of wealth over
  the written scenarios, times ``kelly_fraction`` (a quarter by default). Full Kelly with
  probabilities that are guesses is how an optimistic investor goes broke; it is shown as a
  ceiling only.
- **``portfolio.max_position``**, when written.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from transform.valuation import Valuation, implied_return

NAMES = ("bear", "base", "bull")
LABELS = {"bear": "Pesimista", "base": "Base", "bull": "Optimista"}


@dataclass(frozen=True)
class Scenario:
    name: str
    prob: float
    growth: float | None          # FCF per share, yearly, when written that way
    multiple_written: float | None
    annual_return: float | None
    multiple: float | None        # wealth multiple over the horizon
    note: str                     # why a value is missing, or the user's own note


@dataclass(frozen=True)
class Assessment:
    """See the module docstring. ``None`` fields are unknown, never zero."""

    years: int
    scenarios: list[Scenario]
    expected_multiple: float | None
    expected_return: float | None
    loss_probability: float | None
    worst_loss: float | None             # 1 − smallest multiple, 0 when no scenario loses
    kelly: float | None                  # full Kelly fraction, 0..1
    kelly_ceiling: float | None          # × kelly_fraction
    size_by_budget: float | None         # loss_budget / worst_loss
    size_if_zero: float | None           # loss_budget
    ceiling: float | None                # the smallest of the ceilings that exist
    notes: list[str] = field(default_factory=list)


def kelly_fraction(probs: list[float], multiples: list[float], steps: int = 1000) -> float:
    """The fraction of wealth ``f`` in ``[0, 1]`` maximizing ``Σ p·log(1 + f·(M − 1))``.

    A grid is enough at this precision, and unlike a derivative it handles a scenario that
    goes to zero (``log 0`` at ``f = 1``) without special cases.
    """
    best, best_value = 0.0, 0.0
    for i in range(1, steps + 1):
        f = i / steps
        values = [1 + f * (m - 1) for m in multiples]
        if min(values) <= 0:
            break
        value = sum(p * math.log(v) for p, v in zip(probs, values))
        if value > best_value:
            best, best_value = f, value
    return best


def assess(spec: Mapping[str, Any] | None, valuation: Valuation | None, *,
           years: int = 5, terminal_growth: float = 0.025, dcf_years: int = 10,
           kelly_share: float = 0.25, loss_budget: float | None = None,
           max_position: float | None = None) -> Assessment | None:
    """``spec`` is the card's ``scenarios`` block (validated at load, ``core.config``).
    ``None`` when the card has none."""
    if not spec:
        return None
    years = int(spec.get("years") or years)
    scenarios: list[Scenario] = []
    for name in NAMES:
        s = dict(spec.get(name) or {})
        prob, note = float(s.get("prob", 0.0)), str(s.get("note") or "")
        growth, written = s.get("growth"), s.get("multiple")
        rate = multiple = None
        if written is not None:
            multiple = float(written)
            rate = multiple ** (1 / years) - 1 if multiple > 0 else -1.0
        elif growth is not None:
            implied = (implied_return(valuation.market_cap, valuation.fcf_ttm, float(growth),
                                      terminal_growth=terminal_growth, years=dcf_years)
                       if valuation is not None else None)
            if implied is None or implied.rate is None:
                note = (implied.note if implied is not None else "sin valoración") \
                    + " — escribe este escenario como `multiple`"
            else:
                rate = implied.rate
                multiple = (1 + rate) ** years
        scenarios.append(Scenario(name, prob, None if growth is None else float(growth),
                                  None if written is None else float(written), rate,
                                  multiple, note))

    known = all(s.multiple is not None for s in scenarios)
    probs = [s.prob for s in scenarios]
    multiples = [s.multiple for s in scenarios] if known else []
    expected = sum(p * m for p, m in zip(probs, multiples)) if known else None
    worst = max(0.0, 1 - min(multiples)) if known else None
    kelly = kelly_fraction(probs, multiples) if known else None
    ceiling_kelly = None if kelly is None else kelly * kelly_share
    by_budget = (None if loss_budget is None or worst is None or worst == 0
                 else min(1.0, float(loss_budget) / worst))
    ceilings = [c for c in (by_budget, ceiling_kelly,
                            None if max_position is None else float(max_position))
                if c is not None]
    notes = []
    if not known:
        notes.append("Algún escenario no tiene cifra: sin él no hay rentabilidad esperada ni "
                     "tamaño. El motivo está en su fila.")
    if loss_budget is None:
        notes.append("Sin presupuesto de pérdida escrito (`portfolio.loss_budget`), el tamaño "
                     "por pérdida tolerable no se calcula.")
    elif worst == 0:
        notes.append("Ningún escenario pierde dinero: el presupuesto de pérdida no limita. "
                     "¿Seguro que el pesimista es pesimista?")
    return Assessment(
        years=years, scenarios=scenarios, expected_multiple=expected,
        expected_return=None if expected is None or expected <= 0
        else expected ** (1 / years) - 1,
        loss_probability=sum(p for p, m in zip(probs, multiples) if m < 1) if known else None,
        worst_loss=worst, kelly=kelly, kelly_ceiling=ceiling_kelly,
        size_by_budget=by_budget,
        size_if_zero=None if loss_budget is None else float(loss_budget),
        ceiling=min(ceilings) if ceilings else None, notes=notes,
    )
