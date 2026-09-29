"""Financial health of a sector group, on each company's latest fiscal year (CLAUDE.md §15.5,
point 14 — "a silver miner that is financially sound").

For a miner, health is mostly the **balance sheet and the cash it generates at today's metal
price**: a miner with net debt and negative free cash flow has to issue shares or borrow when
the metal falls, and that is when it can least afford to. So the table puts side by side:

- **cash, debt and net cash** at the fiscal year's end; **net debt / EBITDA** (EBITDA =
  operating income + depreciation, depletion and amortisation) when there is net debt;
- **free cash flow** (operating cash flow − purchases of property, plant and equipment) and
  its margin; revenue growth and operating margin;
- **dilution**: basic weighted shares against the year before;
- **market cap** from the last close × the cover page's shares outstanding, and from it the
  FCF yield and EV/EBITDA.

Reading rules, each a trap measured on the real filings (2026-09-29):

- **Only the latest fiscal year.** Several IFRS tags exist only in old years (one company's
  gross profit last tagged in 2017, others' debt in 2018-2021). A figure missing in the
  latest year is ``None`` — never the value of another year.
- **Untagged debt is unknown, not zero.** Debt is the tagged total, or long-term + current
  **only when the long-term part is tagged**: a current portion of 0 said "no debt" for a
  company with 268.6 M of notes under a tag the synonyms lacked. Otherwise debt, net cash,
  EV and net debt/EBITDA stay ``None``, with a note. Only standard tags count: a loan or a
  convertible under a company's own tag is not seen, which the page says.
- **Share counts in another scale.** One company files its weighted average a thousand times
  too large. The cover page's count (``dei``) is the reference: a weighted average more than
  five times away from it is flagged and its dilution left out.
- **A fiscal year that is not the calendar year** (one ends in March) is said by its end date.

Nothing is scored or ranked: which company is "healthy" is the reader's call (section 12).
Leases are not debt here. Pure functions; no network, no database (section 10).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from transform.macro import point_in_time
from transform.screen import _ratio

YEAR_DAYS = (330, 400)
SCALE_LIMIT = 5.0           # weighted average vs cover-page count, either way
JUMP_LIMIT = 0.5            # a share count up >50 % in a year is a merger, not dilution
STALE_DAYS = 480            # a fiscal year older than this is flagged

COLUMNS = ["cik", "ticker", "name", "basis", "fy_end", "filed", "revenue", "revenue_growth",
           "gross_margin", "operating_margin", "net_margin", "ebitda", "fcf", "fcf_margin",
           "cash", "debt", "net_cash", "net_debt_to_ebitda", "debt_to_equity", "dilution",
           "shares", "price", "market_cap", "fcf_yield", "ev_to_ebitda", "notes"]


def _series(known: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    out: dict[str, dict[str, pd.Series]] = {}
    for sid, rows in known.groupby("series_id"):
        parts = str(sid).split(":")
        if parts[-1] == "src":
            continue
        cik, key = parts[0], ":".join(parts[1:])
        values = pd.Series(rows["value"].astype(float).to_numpy(),
                           index=pd.to_datetime(rows["ts"].astype(str).str[:10]))
        out.setdefault(cik, {})[key] = values[~values.index.duplicated(keep="last")].sort_index()
    return out


def health_table(observations: pd.DataFrame, tickers: Sequence[str],
                 registry: Mapping[str, tuple[str, str]], prices: Mapping[str, float | None],
                 as_of: Any) -> pd.DataFrame:
    """One row per ticker of the group, in the order given. ``registry`` maps CIK → (ticker,
    name); ``prices`` ticker → last close (split-adjusted to today, so it matches the latest
    cover-page count). A ticker with no data keeps a row with a note."""
    known = point_in_time(observations, as_of) if observations is not None and \
        not observations.empty else pd.DataFrame(columns=["series_id", "ts", "value"])
    data = _series(known)
    by_ticker = {t: cik for cik, (t, _) in registry.items()}
    today = pd.Timestamp(as_of)
    rows = []
    for ticker in tickers:
        cik = by_ticker.get(ticker)
        m = data.get(cik or "", {})
        notes: list[str] = []
        revenue = m.get("revenue:fy", pd.Series(dtype=float))
        row: dict[str, Any] = dict.fromkeys(COLUMNS)
        row.update({"cik": cik, "ticker": ticker,
                    "name": registry.get(cik or "", (None, None))[1]})
        if revenue.empty:
            row["notes"] = "sin cifras anuales en la SEC"
            rows.append(row)
            continue
        end = revenue.index[-1]
        prev = [d for d in revenue.index if YEAR_DAYS[0] <= (end - d).days <= YEAR_DAYS[1]]
        prev_end = prev[-1] if prev else None

        def at(key: str, day: pd.Timestamp | None = end) -> float | None:
            values = m.get(key)
            return None if values is None or day is None or day not in values.index \
                else float(values[day])

        rev = at("revenue:fy")
        ebit, dep = at("operating_income:fy"), at("depreciation:fy")
        ocf, capex = at("operating_cash_flow:fy"), at("capex:fy")
        fcf = None if ocf is None or capex is None else ocf - abs(capex)
        ebitda = None if ebit is None or dep is None else ebit + abs(dep)
        cash = at("cash")
        # Debt: the total when tagged; else long-term + current, but only if the long-term
        # part is tagged — a current portion of 0 does not prove there is no long-term debt
        # (Hecla: current 0, notes of 268.6 M under a tag first missing from the synonyms).
        total, lt, cur = at("debt_total"), at("long_term_debt"), at("debt_current")
        if total is not None:
            debt = total
        elif lt is not None:
            debt = lt + (cur or 0.0)
        else:
            debt = None
            notes.append("deuda a largo sin etiquetar en el ejercicio: puede ser cero o "
                         "etiquetas propias")
        net_cash = None if debt is None or cash is None else cash - debt
        if ocf is None:
            notes.append("flujo operativo del ejercicio sin etiquetar")
        elif capex is None:
            notes.append("inversión en activos sin etiqueta estándar: FCF desconocido")

        cover = m.get("shares_outstanding", pd.Series(dtype=float))
        shares = float(cover.iloc[-1]) if not cover.empty else None
        basic_now, basic_prev = at("basic_shares:fy"), at("basic_shares:fy", prev_end)
        scale_ok = (basic_now is not None and shares is not None and shares > 0
                    and 1 / SCALE_LIMIT <= basic_now / shares <= SCALE_LIMIT)
        dilution = None
        if basic_now is not None and basic_prev:
            change = basic_now / basic_prev - 1
            if not scale_ok and shares is not None:
                notes.append("recuento de acciones medio en otra escala que el de portada: "
                             "dilución sin cifra")
            elif change > JUMP_LIMIT:
                notes.append(f"acciones +{change:.0%} en un año: fusión o compra, no "
                             "dilución corriente".replace(".", ","))
            else:
                dilution = change

        price = prices.get(ticker)
        market_cap = None if price is None or shares is None else price * shares
        ev = None if market_cap is None or net_cash is None else market_cap - net_cash
        if (today - end).days > STALE_DAYS:
            notes.append(f"último ejercicio presentado cerró el {end.date()}")
        basis = m.get("ifrs", pd.Series(dtype=float))
        row.update({
            "basis": ("IFRS" if float(basis.iloc[-1]) == 1.0 else "US GAAP")
            if not basis.empty else None,
            "fy_end": end.date().isoformat(),
            "filed": str(known[(known["series_id"] == f"{cik}:revenue:fy")
                               & (known["ts"].astype(str).str[:10] == end.date().isoformat())]
                         ["ts_release"].max())[:10] if "ts_release" in known else None,
            "revenue": rev,
            "revenue_growth": None if prev_end is None else
            (lambda r: None if r is None else r - 1)(_ratio(rev, at("revenue:fy", prev_end))),
            "gross_margin": _ratio(at("gross_profit:fy"), rev),
            "operating_margin": _ratio(ebit, rev),
            "net_margin": _ratio(at("net_income:fy"), rev),
            "ebitda": ebitda, "fcf": fcf, "fcf_margin": _ratio(fcf, rev),
            "cash": cash, "debt": debt, "net_cash": net_cash,
            "net_debt_to_ebitda": (-net_cash / ebitda
                                   if net_cash is not None and net_cash < 0
                                   and ebitda is not None and ebitda > 0 else None),
            "debt_to_equity": _ratio(debt, at("equity"), positive_base=True)
            if debt is not None else None,
            "dilution": dilution, "shares": shares, "price": price,
            "market_cap": market_cap, "fcf_yield": _ratio(fcf, market_cap),
            "ev_to_ebitda": _ratio(ev, ebitda, positive_base=True) if ev is not None else None,
            "notes": "; ".join(notes),
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)
