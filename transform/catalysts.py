"""Catalysts written by hand: FDA decisions, trial readouts, lockup expiries (CLAUDE.md §15.5,
point 9).

A biotech's price moves on dates the SEC calendar does not know: the FDA's action date under
PDUFA, the readout of a pivotal trial, the day insiders may start selling. The panel reads
them from ``catalysts`` in ``settings.local.yaml``, each with the **source** of its date
(section 9.8: every number says where it came from). No free structured source was usable
(verified 2026-09-29): the FDA does not publish action dates; ClinicalTrials.gov has an API
whose ``robots.txt`` disallows ``/api/``, and its completion dates are the sponsor's own
estimates; lockups live in the prospectus text.

**The config is the only truth, and nothing is stored.** A copy in ``events`` would outlive
a date the user moved or deleted — the table is upserted, never pruned — and the public copy
would not carry it anyway. Read at the moment of use, a changed date is simply the new date
(the same choice as ``needs_review``, derived and not stored).

A date is either a day (``date``) or a window (``from`` … ``date``): "data expected in Q4"
is a window, and treating its end as the date would warn after the window had opened. Every
countdown is to the **start**.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

KINDS = {
    "pdufa": "Decisión de la FDA (PDUFA)",
    "readout": "Resultados de un ensayo",
    "lockup": "Fin del lockup",
    "other": "Otro",
}


@dataclass(frozen=True)
class Catalyst:
    ticker: str
    kind: str
    start: dt.date
    end: dt.date
    label: str
    source: str

    @property
    def is_window(self) -> bool:
        return self.start != self.end

    @property
    def kind_label(self) -> str:
        return KINDS.get(self.kind, self.kind)

    @property
    def when(self) -> str:
        """The date, or the window, in words."""
        if not self.is_window:
            return self.end.isoformat()
        return f"entre el {self.start.isoformat()} y el {self.end.isoformat()}"

    @property
    def key(self) -> str:
        """Stable while the catalyst and its dates stay the same; a moved date is new."""
        return f"{self.ticker}:{self.kind}:{self.start.isoformat()}:{self.end.isoformat()}"

    def days_to(self, today: dt.date) -> int:
        """Days to the start; negative once the window has opened."""
        return (self.start - today).days


def _day(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def problems(raw: Any) -> list[str]:
    """Why ``catalysts`` cannot be read; empty when it can. ``None`` is no catalysts."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        return ["must be a list of catalysts"]
    out: list[str] = []
    for i, entry in enumerate(raw):
        name = f"catalysts[{i}]"
        if not isinstance(entry, dict):
            out.append(f"{name}: must be a mapping")
            continue
        name = f"{name} ({entry.get('ticker', '?')})"
        if not isinstance(entry.get("ticker"), str) or not entry["ticker"].strip():
            out.append(f"{name}: ticker missing")
        if entry.get("kind") not in KINDS:
            out.append(f"{name}: kind {entry.get('kind')!r} must be one of {sorted(KINDS)}")
        end = _day(entry.get("date"))
        if end is None:
            out.append(f"{name}: date {entry.get('date')!r} must be YYYY-MM-DD")
        if "from" in entry:
            start = _day(entry.get("from"))
            if start is None:
                out.append(f"{name}: from {entry.get('from')!r} must be YYYY-MM-DD")
            elif end is not None and start > end:
                out.append(f"{name}: from is after date")
        for field in ("label", "source"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                out.append(f"{name}: {field} is required"
                           + (" — where the date comes from (section 9.8)"
                              if field == "source" else ""))
    return out


def parse(raw: Any) -> list[Catalyst]:
    """The catalysts of a valid ``catalysts`` list (``problems`` is empty), by start date."""
    out = []
    for entry in raw or []:
        end = _day(entry["date"])
        start = _day(entry["from"]) if entry.get("from") is not None else end
        out.append(Catalyst(ticker=str(entry["ticker"]).strip().upper(), kind=entry["kind"],
                            start=start, end=end, label=str(entry["label"]).strip(),
                            source=str(entry["source"]).strip()))
    return sorted(out, key=lambda c: (c.start, c.ticker))


def upcoming(catalysts: Iterable[Catalyst], today: dt.date, *, within_days: int | None = None,
             tickers: Iterable[str] | None = None) -> list[Catalyst]:
    """Catalysts not yet over (their end is today or later), optionally only those starting
    within ``within_days`` and only for ``tickers``."""
    wanted = None if tickers is None else {str(t).upper() for t in tickers}
    return [c for c in catalysts
            if c.end >= today
            and (within_days is None or c.days_to(today) <= within_days)
            and (wanted is None or c.ticker in wanted)]


def countdown(catalyst: Catalyst, today: dt.date) -> str:
    """"en 12 días", "hoy", or "ventana abierta" once a window has started."""
    days = catalyst.days_to(today)
    if days > 1:
        return f"en {days} días"
    if days == 1:
        return "mañana"
    if days == 0:
        return "hoy"
    return "ventana abierta"


def from_settings(raw: Mapping[str, Any]) -> list[Catalyst]:
    """The catalysts of a loaded settings ``raw`` mapping."""
    return parse(raw.get("catalysts"))
