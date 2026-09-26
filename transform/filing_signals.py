"""What a company's recent filings say, before the numbers do (CLAUDE.md §15.4, point 4).

The SEC records every filing the day it lands, for free. Some forms are news on their own:
a company that files its annual report late, that says its past statements can no longer be
relied on, or that is told it may be delisted. Others are facts whose meaning depends on the
company: a sale of securities under a shelf (``424B``) is a bond for Amazon and new shares
for a biotech burning cash. This module labels each filing for what it certainly is and
says "act" only for what is actionable on its own or in context:

====================  =========  =====================================================
Signal                Severity   Why
====================  =========  =====================================================
NT 10-K / NT 10-Q     red        the company could not file on time
8-K item 4.02         red        past statements can no longer be relied on
8-K item 3.01         red        delisting notice / listing standards not met
8-K item 4.01         yellow     auditor changed (read why)
8-K item 2.06         yellow     material impairment
shelf (S-3, S-1, F-3) info       the company can now sell securities (shares or debt)
sale (424B*)          info       it sold securities under a shelf — which ones, the
                                 document says; the panel does not read it
employee plan (S-8)   info       shares registered for compensation: SBC dilution
13D                   info       someone holds ≥ 5 % with intent to influence (activist,
                                 or a controlling shareholder like TMUS's parent)
8-K item 5.02         info       a director or officer arrived or left
====================  =========  =====================================================

**The one combination promoted to a warning: a shelf or a sale by a company that burns
cash.** For a company with negative free cash flow, an offering is almost always new
shares — the dilution §2 is about. For one that generates cash it is usually debt, and the
panel does not pretend to know.

Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

RED, YELLOW, INFO = "red", "yellow", "info"

ITEMS = {
    "4.02": (RED, "Los estados financieros anteriores ya no son fiables (reexpresión)"),
    "3.01": (RED, "Aviso de exclusión de cotización o de incumplimiento de requisitos"),
    "4.01": (YELLOW, "Cambio de auditor"),
    "2.06": (YELLOW, "Deterioro material de activos"),
    "5.02": (INFO, "Llegada o salida de consejeros o directivos"),
}
SHELF = {"S-3", "S-3ASR", "S-1", "F-3"}
LATE = {"NT 10-K", "NT 10-Q"}
STAKE = {"SC 13D", "SCHEDULE 13D"}

COLUMNS = ["date", "form", "severity", "kind", "label", "url"]


def _base(form: str) -> str:
    form = (form or "").upper()
    return form[:-2] if form.endswith("/A") else form


def classify(form: str, items: str | None) -> list[tuple[str, str, str]]:
    """``[(severity, kind, label)]`` for one filing; empty when it signals nothing."""
    base = _base(form)
    out: list[tuple[str, str, str]] = []
    if base in LATE:
        out.append((RED, "late", "Presentación tardía: no pudo presentar a tiempo "
                                 + ("el informe anual" if base == "NT 10-K" else
                                    "el trimestral")))
    elif base in SHELF:
        out.append((INFO, "shelf", "Registro para vender valores (acciones o deuda)"))
    elif base.startswith("424B"):
        out.append((INFO, "offering", "Venta de valores bajo un registro (acciones o deuda: "
                                      "léelo)"))
    elif base == "S-8":
        out.append((INFO, "employee_plan", "Acciones registradas para planes de empleados"))
    elif base in STAKE:
        out.append((INFO, "stake", "Participación ≥ 5 % con intención de influir"
                                   + (" (actualización)" if form.upper().endswith("/A")
                                      else "")))
    elif base == "8-K" and items:
        for code in (c.strip() for c in str(items).split(",")):
            if code in ITEMS:
                severity, label = ITEMS[code]
                out.append((severity, f"item_{code}", f"8-K {code}: {label}"))
    return out


def signals(filings: pd.DataFrame, cik: str, since: Any, until: Any | None = None
            ) -> pd.DataFrame:
    """The company's signals filed in ``[since, until]``, newest first (:data:`COLUMNS`)."""
    if filings is None or filings.empty:
        return pd.DataFrame(columns=COLUMNS)
    rows = filings[(filings["cik"] == cik)
                   & (filings["filed_date"].astype(str) >= str(since)[:10])]
    if until is not None:
        rows = rows[rows["filed_date"].astype(str) <= str(until)[:10]]
    out = []
    for row in rows.to_dict("records"):
        for severity, kind, label in classify(row["form"], row.get("items")):
            out.append({"date": str(row["filed_date"])[:10], "form": row["form"],
                        "severity": severity, "kind": kind, "label": label,
                        "url": row.get("url")})
    return pd.DataFrame(out, columns=COLUMNS).sort_values("date", ascending=False) \
        .reset_index(drop=True)


@dataclass(frozen=True)
class Warning_:
    """An actionable signal: ``severity`` red or yellow, with the reason in words."""

    date: str
    form: str
    severity: str
    text: str
    url: str | None


def warnings(found: pd.DataFrame, burns_cash: bool | None) -> list[Warning_]:
    """Red and yellow signals, plus a shelf or sale when the company burns cash (the case in
    which it is almost always new shares). ``burns_cash`` ``None`` = unknown: no promotion."""
    out = []
    for row in found.to_dict("records"):
        if row["severity"] in (RED, YELLOW):
            out.append(Warning_(row["date"], row["form"], row["severity"], row["label"],
                                row["url"]))
        elif row["kind"] in ("shelf", "offering") and burns_cash:
            out.append(Warning_(row["date"], row["form"], YELLOW,
                                f"{row['label']} — y la empresa quema caja: casi siempre "
                                "son acciones nuevas (dilución)", row["url"]))
    return out
