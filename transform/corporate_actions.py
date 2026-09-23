"""Corporate actions: IBKR against yfinance, and the ``needs_review`` flag (section 9.2).

Splits, spin-offs, mergers and delistings break the continuity of quantity, average cost
and PnL. The two providers this project stores both describe them, they disagree, and the
disagreement is the point — which is why ``ingest/prices.py`` and ``ingest/ibkr_flex.py``
write into the same table with a ``source`` column instead of one overwriting the other.

**IBKR is authoritative about what happened to *this* portfolio.** It is the broker that
actually processed the event on the account. yfinance describes the security in general,
from endpoints with no contract (section 4.3). So a disagreement is never resolved in
yfinance's favour, and it is never resolved automatically at all.

The measured case that motivates the module: **yfinance encodes spin-offs as splits.** On
2016-09-19, when GICS separated real estate and XLF distributed XLRE, the library reports a
"split" of ratio 1.231 and the price drops from 23.62 to 19.31. There was no split. The
arithmetic happens to be right for the *price* series, so ``close_raw`` survives it — but a
spin-off **divides the cost base between two entities**, and recording it as a split leaves
the base whole on the original and falsifies the PnL permanently.

**Nothing here corrects anything.** Section 9.2 is explicit: on an unrecognized corporate
action the system fails loudly and marks the position ``needs_review``, never infers. This
module produces that flag and the evidence behind it.

**Where the flag lives — decided here.** It is *derived*, not stored: a pure function of
the ``corporate_actions`` rows and the open positions, recomputed on read. Storing it would
create a second source of truth that goes stale the moment either side is re-ingested, and
it would have to be cleared by hand after a review — a piece of mutable state nobody would
remember to maintain. Derived, it disappears exactly when the underlying disagreement does.

Pure functions, no network, no database handle (section 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from transform.adjustments import as_date, as_frame, unclean_split_ratios

IBKR = "ibkr"
YFINANCE = "yfinance"

# Providers stamp an ex-date from different systems, so the same event can land a day or
# two apart. Wider than this and they are different events, not one event seen twice.
MATCH_WINDOW_DAYS = 5

# Kinds that change what a share *is*, so a position cannot be carried across one without
# a decision. A cash dividend is not among them: it is income, handled in level 4.
STRUCTURAL = frozenset({"split", "spinoff", "merger", "delisting"})


@dataclass(frozen=True)
class ActionReview:
    """One corporate action that a human has to look at, with both witnesses attached.

    Attributes:
        ticker: Affected ticker.
        ex_date: Ex-date, as the reporting source stamped it.
        reason: Machine-readable cause — ``'kind_mismatch'``, ``'unconfirmed'``,
            ``'unclean_ratio'`` or ``'unmapped'``.
        ibkr: What IBKR reported, or ``None`` when it reported nothing.
        yfinance: What yfinance reported, or ``None``.
        detail: One line for the panel, in Spanish.
    """

    ticker: str
    ex_date: str | None
    reason: str
    ibkr: dict[str, Any] | None
    yfinance: dict[str, Any] | None
    detail: str


@dataclass(frozen=True)
class PositionReview:
    """Everything blocking a clean read of one position (section 9.2).

    Attributes:
        ticker: The held ticker.
        needs_review: Whether anything at all was found. **Never inferred away.**
        reviews: The findings, most structural first.
    """

    ticker: str
    needs_review: bool
    reviews: list[ActionReview] = field(default_factory=list)


def _rows_for(frame: pd.DataFrame, ticker: str, source: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows = frame[(frame["ticker"] == ticker) & (frame["source"] == source)]
    return rows.to_dict("records")


def _near(left: Any, right: Any, days: int = MATCH_WINDOW_DAYS) -> bool:
    """Whether two ex-dates are close enough to be the same event."""
    a, b = as_date(left), as_date(right)
    if a is None or b is None:
        return False
    return abs((a - b).days) <= days


def _summary(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "kind": row.get("kind"),
        "ex_date": row.get("ex_date"),
        "ratio": None if row.get("ratio") is None or pd.isna(row.get("ratio"))
        else float(row["ratio"]),
        "amount": None if row.get("amount") is None or pd.isna(row.get("amount"))
        else float(row["amount"]),
    }


def reconcile_ticker(
    actions: Sequence[Mapping[str, Any]] | pd.DataFrame | None, ticker: str
) -> list[ActionReview]:
    """Compare both providers' structural actions for one ticker.

    Cash dividends are excluded: they do not change what a share is, and they reconcile
    against ``cash_transactions`` in level 4 instead, against the amount actually received.

    Returns:
        Findings, ``[]`` when the two sources agree or when only one has anything to say
        about events neither calls structural.
    """
    frame = as_frame(actions)
    if frame.empty:
        return []

    broker = [r for r in _rows_for(frame, ticker, IBKR) if r.get("kind") in STRUCTURAL]
    market = [r for r in _rows_for(frame, ticker, YFINANCE) if r.get("kind") in STRUCTURAL]

    findings: list[ActionReview] = []
    matched_broker: list[int] = []

    for quote in market:
        partner = next(
            (
                index for index, row in enumerate(broker)
                if index not in matched_broker and _near(row.get("ex_date"),
                                                         quote.get("ex_date"))
            ),
            None,
        )
        if partner is None:
            findings.append(ActionReview(
                ticker=ticker, ex_date=str(quote.get("ex_date")), reason="unconfirmed",
                ibkr=None, yfinance=_summary(quote),
                detail=(
                    f"yfinance reporta un **{quote.get('kind')}** el "
                    f"{quote.get('ex_date')} que IBKR no corrobora. IBKR es la fuente "
                    "autorizada de lo que le pasó a TU cartera (§9.2); si tenías la "
                    "posición, revisa qué ocurrió antes de fiarte de cantidad o coste."
                ),
            ))
            continue

        row = broker[partner]
        matched_broker.append(partner)
        if row.get("kind") != quote.get("kind"):
            findings.append(ActionReview(
                ticker=ticker, ex_date=str(row.get("ex_date")), reason="kind_mismatch",
                ibkr=_summary(row), yfinance=_summary(quote),
                detail=(
                    f"IBKR dice **{row.get('kind')}** y yfinance dice "
                    f"**{quote.get('kind')}** para el mismo evento "
                    f"({quote.get('ex_date')}). Manda IBKR. Un spin-off tratado como "
                    "split deja la base de coste entera en la entidad original y falsea "
                    "el PnL para siempre (§9.2) — no se corrige solo."
                ),
            ))

    for index, row in enumerate(broker):
        if index in matched_broker:
            continue
        findings.append(ActionReview(
            ticker=ticker, ex_date=str(row.get("ex_date")), reason="unconfirmed",
            ibkr=_summary(row), yfinance=None,
            detail=(
                f"IBKR reporta un **{row.get('kind')}** el {row.get('ex_date')} que "
                "yfinance no recoge. La serie de precios puede no reflejarlo, así que la "
                "valoración de esa posición queda en duda hasta comprobarlo (§9.8)."
            ),
        ))

    for flagged in unclean_split_ratios(frame[frame["ticker"] == ticker]):
        findings.append(ActionReview(
            ticker=ticker, ex_date=str(flagged["ex_date"]), reason="unclean_ratio",
            ibkr=None, yfinance=_summary(flagged) if flagged["source"] == YFINANCE else None,
            detail=flagged["detail"],
        ))

    order = {"kind_mismatch": 0, "unclean_ratio": 1, "unconfirmed": 2, "unmapped": 3}
    return sorted(findings, key=lambda f: (order.get(f.reason, 9), str(f.ex_date)))


def review_positions(
    actions: Sequence[Mapping[str, Any]] | pd.DataFrame | None,
    positions: pd.DataFrame | Sequence[str] | None,
    *,
    unmapped: Sequence[str] = (),
) -> list[PositionReview]:
    """The ``needs_review`` flag of section 9.2, per held position.

    Args:
        actions: Rows of ``corporate_actions``, both sources.
        positions: Output of ``transform.portfolio.latest_positions`` or a plain list of
            tickers. Only held tickers are reviewed: a discrepancy on a security nobody
            owns is a data-quality note, not a blocked position.
        unmapped: Labels the IBKR ingester could not map to a known kind, from its
            ``partial_failures()``. They arrive already flagged as needing review, and are
            carried through rather than re-derived so the two paths cannot disagree.

    Returns:
        One :class:`PositionReview` per held ticker that has something to look at.
        Positions that reconcile cleanly are **not** included — the list is what needs
        attention, not a roll call.
    """
    if positions is None:
        held: list[str] = []
    elif isinstance(positions, pd.DataFrame):
        held = [] if positions.empty else sorted({str(t) for t in positions["ticker"]})
    else:
        held = sorted({str(t) for t in positions})

    carried: dict[str, list[ActionReview]] = {}
    for label in unmapped:
        ticker = next((t for t in held if t and t in str(label)), None)
        if ticker is None:
            continue
        carried.setdefault(ticker, []).append(ActionReview(
            ticker=ticker, ex_date=None, reason="unmapped", ibkr=None, yfinance=None,
            detail=(
                f"IBKR reportó una reorganización que el panel no sabe clasificar: "
                f"{label}. No se infiere el tipo (§9.2)."
            ),
        ))

    out: list[PositionReview] = []
    for ticker in held:
        findings = [*carried.get(ticker, []), *reconcile_ticker(actions, ticker)]
        if findings:
            out.append(PositionReview(ticker=ticker, needs_review=True, reviews=findings))
    return out


def review_index(reviews: Sequence[PositionReview]) -> dict[str, PositionReview]:
    """Index the reviews by ticker, for a page that renders one position at a time."""
    return {review.ticker: review for review in reviews}


def data_quality_notes(
    actions: Sequence[Mapping[str, Any]] | pd.DataFrame | None,
) -> list[ActionReview]:
    """Findings on securities that are **not** held — the market-reference ETFs, mostly.

    Kept apart from :func:`review_positions` on purpose. XLF's 2016 spin-off shows up here
    forever and blocks nothing; letting it into the position flags would train the reader
    to ignore them, which is how a real flag gets missed.
    """
    frame = as_frame(actions)
    if frame.empty:
        return []
    notes: list[ActionReview] = []
    for flagged in unclean_split_ratios(frame):
        notes.append(ActionReview(
            ticker=str(flagged["ticker"]), ex_date=str(flagged["ex_date"]),
            reason="unclean_ratio", ibkr=None, yfinance=None, detail=flagged["detail"],
        ))
    return notes
