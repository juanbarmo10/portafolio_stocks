"""IBKR Activity Flex Query: the real account, read-only (CLAUDE.md sections 4.1, 9.2, 9.3).

The Flex Web Service token is *structurally* incapable of placing, modifying or cancelling
an order, or of moving funds. That is a limitation of IBKR's system, not a setting anyone
has to remember keeping — which is why this project reads the account through Flex and not
through the TWS API (section 4.1). Nothing in this module writes to the account.

Two HTTP calls, verified end to end against the real account on 2026-09-16
(RESEARCH.md section 2.7):

1. ``SendRequest`` with token + query id returns a *reference code* and, in ``<Url>``, the
   host to collect the report from. That host is **gdcdyn**, not the **ndcdyn** the request
   went to, so the returned URL is used rather than a hardcoded one.
2. ``GetStatement`` with the reference code returns the XML. Generation is asynchronous:
   a not-ready answer is normal and is retried with backoff.

⚠️ **A failure arrives with HTTP 200.** The error is inside the body
(``<Status>Fail</Status>`` plus ``ErrorCode``/``ErrorMessage``), so a client that only
checks ``raise_for_status()`` treats an error as a statement. Every response goes through
:func:`_classify` first.

**Personal data.** ``AccountInformation`` carries the holder's name, full home address,
date of birth and email. :func:`account_summary` rebuilds it from a whitelist, so only the
operational fields can ever reach a log, the database or a traceback. An attribute nobody
put on the list is an attribute nobody reviewed.

**What is never inferred** (sections 9.2, 9.3, 12): a ticker that does not resolve to a CIK
is stored as ``NULL`` and counted; a corporate-action code outside the documented map is
skipped and reported, never bucketed as a split; an unknown trade side or cash type is
reported rather than guessed. All of those land in :meth:`partial_failures`, so the run
loads what it understood and still exits non-zero.

fetch() -> DataFrame[source, series_id, ts, ts_release, value]
    series_id = "TICKER:position_qty" | "TICKER:position_cost_basis"
                | "NAV:total" | "NAV:cash" | "NAV:stock" | "NAV:dividend_accruals"

fetch_tables() -> {"trades": [...], "cash_transactions": [...], "corporate_actions": [...]}
"""

from __future__ import annotations

import datetime as dt
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from core.config import Settings
from core.logging_setup import get_logger
from db.loader import CASH_TRANSACTION_COLUMNS, TRADE_COLUMNS
from ingest.base import Ingester, empty_observations, retry

log = get_logger(__name__)

SOURCE = "ibkr"

# Defaults; the effective values come from sources.ibkr_flex in settings.yaml (section 10).
FLEX_BASE = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"
FLEX_VERSION = "3"
USER_AGENT = "equitydash/0.1"

# AccountInformation fields this project may read. Everything else — name, street, city,
# state, postalCode, country, dateOfBirth, primaryEmail and their residential twins — is
# dropped before the data leaves this module (see the module docstring).
#
# Read from the XML rather than from the parsed object on purpose: ibflex 1.1 does not
# model taxLotMatchingMethod or dividendReinvestmentEnabled, and silently drops them. The
# first one matters — it is IBKR confirming it matches lots FIFO, the same convention the
# fiscal layer assumes (section 11) — and a log line reading "lot matching=None" when the
# account says FIFO is precisely the plausible-and-wrong output this project refuses.
ACCOUNT_WHITELIST = (
    "accountId", "accountType", "customerType", "accountCapabilities", "currency",
    "ibEntity", "dateOpened", "dateFunded", "dateClosed", "lastTradedDate",
    "taxLotMatchingMethod", "dividendReinvestmentEnabled", "tradingPermissions",
)

# IBKR cash-transaction types -> this project's vocabulary. Payment in lieu keeps its own
# kind: it is not a dividend and may not be taxed like one (section 11), so lumping the
# two would quietly corrupt the only withholding figure the panel is allowed to use.
CASH_KINDS = {
    "DIVIDEND": "dividend",
    "PAYMENTINLIEU": "payment_in_lieu",
    "WHTAX": "withholding_tax",
    "BROKERINTRCVD": "interest",
    "BROKERINTPAID": "interest",
    "BONDINTRCVD": "interest",
    "BONDINTPAID": "interest",
    "FEES": "fee",
    "COMMADJ": "fee",
    "ADVISORFEES": "fee",
    "DEPOSITWITHDRAW": "deposit",
}

# IBKR reorg codes -> this project's vocabulary. Deliberately partial: a code that is not
# here is *reported*, never mapped to the nearest neighbour. A spin-off splits the cost
# basis across two entities and a split does not, so one wrong bucket falsifies the PnL
# for good (section 9.2).
CORPORATE_ACTION_KINDS = {
    "FORWARDSPLIT": "split",
    "REVERSESPLIT": "split",
    "SPINOFF": "spinoff",
    "MERGER": "merger",
    "DELISTWORTHLESS": "delisting",
    "CASHDIV": "dividend",
}

TRADE_SIDES = {"BUY": "buy", "SELL": "sell"}


class FlexRequestError(RuntimeError):
    """The Flex Web Service reported a failure, or answered something unparseable."""


class FlexNotReady(RuntimeError):
    """The statement is still being generated. Retryable by design (section 4.1)."""


# --- HTTP client --------------------------------------------------------------


def _classify(body: str) -> ET.Element:
    """Parse a Flex response and raise on anything that is not a usable answer.

    Args:
        body: Raw response text.

    Returns:
        The parsed root element.

    Raises:
        FlexNotReady: The service says the statement is not generated yet.
        FlexRequestError: The service reported a failure, or the body is not XML.
    """
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        # Truncated: the body may be a login page or an error blob; it is not a secret,
        # but it is not worth dumping a megabyte into the log either.
        raise FlexRequestError(f"Flex response is not XML: {body[:200]!r}") from exc

    status = (root.findtext("Status") or "").strip()
    if not status:
        return root  # a generated report (<FlexQueryResponse>) carries no Status

    if status.lower() == "success":
        return root

    code = (root.findtext("ErrorCode") or "").strip()
    message = (root.findtext("ErrorMessage") or "").strip()
    # "Warn" + a generation-in-progress message is the documented not-ready answer.
    if status.lower() == "warn" or "progress" in message.lower():
        raise FlexNotReady(f"Statement not ready yet (code {code}): {message}")
    raise FlexRequestError(f"Flex Web Service failed (code {code}): {message}")


class FlexClient:
    """Two-step Activity Flex download. Read-only by construction.

    The token never reaches a log line or an exception message: it travels only in the
    query string of the request itself. Anything this class reports names the endpoint,
    never the URL it called.
    """

    def __init__(
        self,
        token: str,
        query_id: str,
        *,
        base_url: str = FLEX_BASE,
        version: str = FLEX_VERSION,
        timeout_s: float = 60.0,
        retry_kwargs: dict[str, Any] | None = None,
        session: Any = None,
    ) -> None:
        self._token = token
        self._query_id = query_id
        self._base_url = base_url.rstrip("/")
        self._version = str(version)
        self._timeout_s = timeout_s
        self._retry_kwargs = retry_kwargs or {}
        self._session = session or requests

    def _get(self, url: str, params: dict[str, str]) -> str:
        response = self._session.get(
            url, params=params, headers={"User-Agent": USER_AGENT}, timeout=self._timeout_s
        )
        response.raise_for_status()  # necessary but NOT sufficient: see _classify
        return response.text

    def send_request(self) -> tuple[str, str]:
        """Ask for the statement.

        Returns:
            ``(reference_code, statement_url)``. The URL comes from the response because
            the collection host differs from the request host.
        """
        root = _classify(
            self._get(
                f"{self._base_url}/SendRequest",
                {"t": self._token, "q": self._query_id, "v": self._version},
            )
        )
        reference = (root.findtext("ReferenceCode") or "").strip()
        if not reference:
            raise FlexRequestError("SendRequest succeeded but returned no ReferenceCode.")
        url = (root.findtext("Url") or "").strip() or f"{self._base_url}/GetStatement"
        log.info("Flex statement requested; collecting from %s.", url, extra={"source": SOURCE})
        return reference, url

    def get_statement(self, reference: str, url: str) -> str:
        """Collect the generated statement, retrying while it is not ready."""

        def call() -> str:
            body = self._get(url, {"t": self._token, "q": reference, "v": self._version})
            _classify(body)  # raises FlexNotReady while generation is in progress
            return body

        return retry(call, exceptions=(FlexNotReady,), **self._retry_kwargs)

    def download(self) -> str:
        """Full handshake: request, then collect. Returns the statement XML."""
        reference, url = self.send_request()
        return self.get_statement(reference, url)


# --- Parsing ------------------------------------------------------------------


def _looks_like_xml(source: str | Path) -> bool:
    """Whether ``source`` is the XML itself rather than a path to it."""
    return isinstance(source, str) and source.lstrip().startswith("<")


def parse_statement(source: str | Path) -> tuple[Any, int]:
    """Parse a Flex XML statement into ibflex objects.

    Args:
        source: Path to an XML file, or the XML text itself.

    Returns:
        ``(statement, warning_count)`` for the single account in the report.

    Raises:
        FlexRequestError: If the report carries no statement, or carries more than one
            account. Merging several accounts into one set of ``NAV:*`` series would be a
            silent decision about whose money is whose; it is refused until someone
            decides on purpose (section 12).
    """
    from ibflex import parser  # noqa: PLC0415 — optional extra, imported at call time

    payload: Any = source.encode("utf-8") if _looks_like_xml(source) else source

    # ibflex 1.x warns and skips XML attributes it does not know, and a real statement
    # carries ~1300 of them. They are summarized, never dumped: a log with 1365 warning
    # lines in it is a log nobody reads (RESEARCH.md section 2.7).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response = parser.parse(payload)

    statements = list(response.FlexStatements or [])
    if not statements:
        raise FlexRequestError("The Flex report contains no statement.")
    if len(statements) > 1:
        raise FlexRequestError(
            f"The Flex report contains {len(statements)} accounts. This ingester writes one "
            "set of NAV series and would silently merge them; configure one query per "
            "account, or decide on a per-account series naming first."
        )
    return statements[0], len(caught)


def account_facts(source: str | Path) -> dict[str, str]:
    """Operational facts about the account, rebuilt from :data:`ACCOUNT_WHITELIST`.

    The whitelist is the mechanism, not a convention: name, address, date of birth and
    email are simply not in the returned mapping, so they cannot leak from here into a
    log, the database or a traceback. (They still exist in memory inside the parsed
    statement — what this controls is what *leaves*.)

    Args:
        source: Path to the statement, or the XML text itself.

    Returns:
        Whitelisted attributes present in the report, as strings. A field the report does
        not carry is absent from the mapping rather than present-and-empty, so a caller
        can tell "not reported" from "reported as nothing".
    """
    root = ET.fromstring(source) if _looks_like_xml(source) else ET.parse(source).getroot()
    element = root.find(".//AccountInformation")
    if element is None:
        return {}
    return {
        field: value
        for field, value in element.attrib.items()
        if field in ACCOUNT_WHITELIST and value not in (None, "")
    }


def _iso_date(value: dt.date | dt.datetime | None) -> str | None:
    """Midnight-UTC ISO8601 stamp for a date, matching the rest of the project."""
    if value is None:
        return None
    day = value.date() if isinstance(value, dt.datetime) else value
    return f"{day.isoformat()}T00:00:00+00:00"


def _iso_instant(value: dt.datetime | None, timezone: ZoneInfo) -> str | None:
    """Convert a statement timestamp to ISO8601 UTC.

    IBKR stamps executions in the report's configured timezone and the XML does not say
    which one it is, so the zone comes from ``sources.ibkr.statement_timezone`` in
    settings.yaml. That makes the assumption a visible parameter the operator can check
    against the portal instead of a constant buried here (section 10).
    """
    if value is None:
        return None
    local = value if value.tzinfo else value.replace(tzinfo=timezone)
    return local.astimezone(dt.timezone.utc).isoformat()


# --- Row builders (pure) ------------------------------------------------------


def trade_rows(
    statement: Any, resolve_cik: Callable[[str], str | None], timezone: ZoneInfo
) -> tuple[list[dict[str, Any]], list[str]]:
    """Executions -> ``trades`` rows, plus the labels of what could not be mapped.

    ``quantity`` is stored unsigned; direction lives in ``side``. ``commission`` keeps
    IBKR's sign convention, where a cost is negative, so a plain sum is what was paid.
    """
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for trade in statement.Trades or ():
        side = TRADE_SIDES.get(getattr(trade.buySell, "name", ""))
        if side is None:
            # 'BUY (Ca.)' and friends: a cancelled execution is not a position change,
            # and the query excludes them, so one showing up is news, not noise.
            failures.append(f"trade {trade.tradeID}: unmapped side {trade.buySell!r}")
            continue
        rows.append({
            "trade_id": str(trade.tradeID),
            "conid": None if trade.conid is None else str(trade.conid),
            "cik": resolve_cik(trade.symbol),
            "ticker": trade.symbol,
            "ts": _iso_instant(trade.dateTime, timezone) or _iso_date(trade.tradeDate),
            "side": side,
            "quantity": abs(float(trade.quantity)),
            "price": float(trade.tradePrice),
            "currency": trade.currency,
            "commission": None if trade.ibCommission is None else float(trade.ibCommission),
            "fx_rate": None if trade.fxRateToBase is None else float(trade.fxRateToBase),
        })
    return rows, failures


def cash_transaction_rows(
    statement: Any, resolve_cik: Callable[[str], str | None], timezone: ZoneInfo
) -> tuple[list[dict[str, Any]], list[str]]:
    """Dividends, withholding, interest, fees and deposits -> ``cash_transactions`` rows.

    Withholding rows carry the figure IBKR actually withheld. No rate is ever assumed on
    the user's behalf (section 11), which is why the observed amount is stored verbatim,
    sign included.
    """
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for tx in statement.CashTransactions or ():
        kind = CASH_KINDS.get(getattr(tx.type, "name", ""))
        if kind is None:
            failures.append(f"cash transaction {tx.transactionID}: unmapped type {tx.type!r}")
            continue
        tx_id = str(tx.transactionID) if tx.transactionID else None
        if tx_id is None:
            # Derived, not invented: the same upstream row always yields the same key, so
            # idempotency holds. Reported because a Flex row without a transactionID is
            # unusual enough to be worth a human look.
            tx_id = f"{tx.accountId}:{kind}:{tx.dateTime}:{tx.symbol or ''}:{tx.amount}"
            failures.append(f"cash transaction with no transactionID; derived key {tx_id}")
        rows.append({
            "tx_id": tx_id,
            "conid": None if tx.conid is None else str(tx.conid),
            "cik": resolve_cik(tx.symbol) if tx.symbol else None,
            "ticker": tx.symbol,
            "ts": _iso_instant(tx.dateTime, timezone) or _iso_date(tx.reportDate),
            "kind": kind,
            "amount": float(tx.amount),
            "currency": tx.currency,
        })
    return rows, failures


def security_rows(
    statement: Any, resolve_cik: Callable[[str], str | None]
) -> tuple[list[dict[str, Any]], list[str]]:
    """``SecuritiesInfo`` -> ``securities`` rows: the security master (section 9.3).

    This is the only stable key the **account** side of the project has. Everywhere else it
    keys on the ticker, which section 9.3 says is not a key: it gets renamed under the same
    company (FB -> META) and reassigned between companies. ``conid`` is IBKR's permanent
    contract identifier, and the ISIN, CUSIP and FIGI beside it are the industry-standard
    ones that survive a rename too.

    ``issuerCountryCode`` is carried because it determines an asset's **situs**, which
    section 11 needs in order to show US-situs exposure as a question for an adviser. It
    appears nowhere else in the statement. Nothing is computed from it.

    ``first_seen`` / ``last_seen`` bracket when this conid-to-ticker mapping was observed,
    so a rename is visible rather than an overwrite nobody notices.

    Returns:
        ``(rows, failures)`` — a security with no conid is skipped and reported, because
        without the key the row could only be identified by the ticker.
    """
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    observed = _statement_period(statement)

    for info in statement.SecuritiesInfo or ():
        conid = getattr(info, "conid", None)
        if not conid:
            failures.append(
                f"security {info.symbol!r}: no conid, so it has no stable key (§9.3)"
            )
            continue
        rows.append({
            "conid": str(conid),
            "ticker": info.symbol,
            "cik": resolve_cik(info.symbol) if info.symbol else None,
            "name": info.description,
            "isin": getattr(info, "isin", None),
            "cusip": getattr(info, "cusip", None),
            "figi": getattr(info, "figi", None),
            "asset_category": getattr(info.assetCategory, "value", None)
            or (str(info.assetCategory) if info.assetCategory else None),
            "sub_category": getattr(info, "subCategory", None),
            "listing_exchange": getattr(info, "listingExchange", None),
            "issuer_country": getattr(info, "issuerCountryCode", None),
            "currency": info.currency,
            "multiplier": None if info.multiplier is None else float(info.multiplier),
            "first_seen": observed[0],
            "last_seen": observed[1],
            "source": SOURCE,
        })
    return rows, failures


def _statement_period(statement: Any) -> tuple[str | None, str | None]:
    """``(fromDate, toDate)`` of the statement as ISO dates, for first_seen / last_seen."""
    return (
        _iso_date(getattr(statement, "fromDate", None)),
        _iso_date(getattr(statement, "toDate", None)),
    )


def corporate_action_rows(
    statement: Any, resolve_cik: Callable[[str], str | None]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Corporate actions as *IBKR* reports them -> ``corporate_actions`` rows.

    This is the authoritative account of what happened to **this** portfolio (section 9.2).
    It is stored with ``source='ibkr'`` next to the yfinance rows precisely so the two can
    disagree in the open: yfinance encodes spin-offs as splits, and the reconciliation that
    catches that needs both witnesses present.

    An unmapped reorg code is skipped and reported. Guessing is what section 9.2 forbids.
    """
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for action in statement.CorporateActions or ():
        code = getattr(action.type, "name", "")
        kind = CORPORATE_ACTION_KINDS.get(code)
        if kind is None:
            failures.append(
                f"corporate action {action.transactionID or action.actionID} on "
                f"{action.symbol}: unmapped type {action.type!r} — needs review"
            )
            continue
        ex_date = _iso_date(action.dateTime or action.reportDate)
        rows.append({
            "action_id": f"ibkr:{action.transactionID or action.actionID}",
            "cik": resolve_cik(action.symbol) if action.symbol else None,
            "ticker": action.symbol,
            "kind": kind,
            "ex_date": ex_date,
            # Ratio is not reported as such by Flex; it is derivable from quantity only
            # for some codes, so it stays NULL rather than being reconstructed by guess.
            "ratio": None,
            "amount": None if action.amount is None else float(action.amount),
            "currency": action.currency,
            "source": SOURCE,
        })
    return rows, failures


def unmapped_action_rows(statement: Any, today: str) -> list[dict[str, Any]]:
    """The reorganizations :func:`corporate_action_rows` refused to classify, as rows of
    ``unmapped_actions`` — so the portfolio page can flag the position, not only the log.

    IBKR's own description travels verbatim; the kind is never guessed (section 9.2).
    """
    rows = []
    for action in statement.CorporateActions or ():
        code = getattr(action.type, "name", "")
        if code in CORPORATE_ACTION_KINDS:
            continue
        rows.append({
            "action_id": f"ibkr:{action.transactionID or action.actionID}",
            "ticker": action.symbol,
            "conid": None if getattr(action, "conid", None) is None else str(action.conid),
            "ex_date": _iso_date(action.dateTime or action.reportDate),
            "code": code or repr(action.type),
            "description": getattr(action, "description", None),
            "source": SOURCE,
            "first_seen": today,
        })
    return rows


def verify_against_funds(
    statement: Any,
    trades: Sequence[Mapping[str, Any]],
    cash: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 0.005,
) -> list[str]:
    """Check what this module parsed against IBKR's own cash ledger. **Verify, not store.**

    ``StmtFunds`` (Statement of Funds) is every cash movement of the account in order, each
    row carrying a running ``balance``. It contains no fact that ``trades`` and
    ``cash_transactions`` do not already hold — the two are slices of it by type — so
    storing it would duplicate the database. What it adds is **the balance chain**: an
    arithmetic sequence that has to tie out from the first row to the last.

    So it is used as a third witness instead, the same argument phase 1 made for the NAV:
    two independent derivations that agree are evidence, and a copied number is not. Since
    nothing is stored, the check has to run here, where the statement is in hand.

    Three checks, in increasing value:

    1. **The ledger is internally consistent** — each balance equals the previous one plus
       that row's amount, per currency. A break means the section came back truncated or
       out of order, which would make checks 2 and 3 meaningless.
    2. **Every execution's cash matches, trade by trade.** For each ledger row carrying a
       ``tradeID``, the cash implied by the parsed trade — ``±quantity × price +
       commission`` — must equal the amount the broker moved. This is what actually tests
       the parser: a dropped commission, a sign flip or a wrong side shows up here.
    3. **Non-trade cash matches in aggregate**: dividends, withholding, interest, fees and
       deposits, against ``cash_transactions``.

    Measured against the real statement on 2026-09-23: 0 breaks in 46 rows, 25 of 25 trades
    matching to the cent, and 549.76 against 549.76 of non-trade cash.

    Returns:
        Discrepancy labels for :meth:`Ingester.partial_failures`, so a mismatch reaches the
        exit code. Empty when everything ties out. A disagreement here means this module
        parsed something differently from the broker, which is never cosmetic.
    """
    lines = list(statement.StmtFunds or ())
    if not lines:
        # Not a failure: the section is optional in the query. But silence would be
        # indistinguishable from "verified", so say which one it is.
        log.info(
            "StmtFunds is absent from the statement; the cash ledger cross-check did not "
            "run.", extra={"source": SOURCE},
        )
        return []

    problems: list[str] = []

    # 1. The balance chain, per currency.
    previous: dict[str, float] = {}
    breaks = 0
    for line in lines:
        currency = str(line.currency or "")
        amount, balance = line.amount, line.balance
        if amount is None or balance is None:
            continue
        before = previous.get(currency)
        if before is not None and abs(before + float(amount) - float(balance)) > tolerance:
            breaks += 1
        previous[currency] = float(balance)
    if breaks:
        problems.append(
            f"StmtFunds: {breaks} break(s) in the running balance — the ledger is "
            "truncated or out of order, so the cash cross-check cannot be trusted"
        )

    # 2. Per-execution cash. The check that actually exercises the trade parser.
    by_id = {str(row["trade_id"]): row for row in trades}
    mismatched: list[str] = []
    missing: list[str] = []
    for line in lines:
        if not line.tradeID or line.amount is None:
            continue
        trade = by_id.get(str(line.tradeID))
        if trade is None:
            missing.append(str(line.tradeID))
            continue
        sign = 1.0 if trade["side"] == "sell" else -1.0
        expected = (
            sign * float(trade["quantity"]) * float(trade["price"])
            + float(trade["commission"] or 0.0)
        )
        if abs(expected - float(line.amount)) > tolerance:
            mismatched.append(
                f"{line.tradeID} (computed {expected:.6f} vs ledger {float(line.amount):.6f})"
            )
    if missing:
        problems.append(
            f"StmtFunds moved cash for {len(missing)} execution(s) this module did not "
            f"parse into trades: {', '.join(missing[:5])}"
        )
    if mismatched:
        problems.append(
            f"StmtFunds disagrees with the parsed cash of {len(mismatched)} trade(s): "
            + "; ".join(mismatched[:5])
        )

    # 3. Non-trade cash, in aggregate.
    ledger_cash = sum(
        float(line.amount) for line in lines if not line.tradeID and line.amount is not None
    )
    parsed_cash = sum(float(row["amount"]) for row in cash)
    if abs(ledger_cash - parsed_cash) > tolerance:
        problems.append(
            f"StmtFunds non-trade cash {ledger_cash:.4f} vs parsed cash_transactions "
            f"{parsed_cash:.4f} (difference {ledger_cash - parsed_cash:+.4f})"
        )

    if not problems:
        log.info(
            "StmtFunds cross-check passed: %d ledger rows, %d execution(s) matched, "
            "non-trade cash %.2f.", len(lines), len(by_id), parsed_cash,
            extra={"source": SOURCE},
        )
    return problems


def observation_records(statement: Any) -> list[dict[str, Any]]:
    """Positions and daily NAV as long-format observations (RESEARCH.md section 2.1).

    Positions go to ``observations`` rather than a dedicated table so that a new field in
    the IBKR report never becomes a migration, and so the position's history comes for
    free. The NAV series is the second witness the phase-1 acceptance criterion needs:
    ``NAV:total`` is what IBKR says the account is worth, against which the panel's own
    valuation is reconciled.
    """
    records: list[dict[str, Any]] = []

    for position in statement.OpenPositions or ():
        stamp = _iso_date(position.reportDate)
        for series, value in (
            ("position_qty", position.position),
            ("position_cost_basis", position.costBasisMoney),
        ):
            if value is None:
                continue
            records.append({
                "source": SOURCE, "series_id": f"{position.symbol}:{series}",
                "ts": stamp, "ts_release": stamp, "value": float(value),
            })

    for day in statement.EquitySummaryInBase or ():
        stamp = _iso_date(day.reportDate)
        for series, value in (
            ("NAV:total", day.total),
            ("NAV:cash", day.cash),
            ("NAV:stock", day.stock),
            ("NAV:dividend_accruals", day.dividendAccruals),
        ):
            if value is None:
                continue
            records.append({
                "source": SOURCE, "series_id": series,
                "ts": stamp, "ts_release": stamp, "value": float(value),
            })

    return records


# --- Ingester -----------------------------------------------------------------


class IbkrFlexIngester(Ingester):
    """Downloads, parses and maps one Activity Flex Query. Never writes to the account."""

    source = SOURCE

    def __init__(self, settings: Settings, statement: str | Path | None = None) -> None:
        """
        Args:
            settings: Effective configuration.
            statement: An already-downloaded statement (path or XML text). When given, no
                network call is made — which is how the tests run against the frozen
                fixture, and how a saved statement can be re-ingested offline.
        """
        cfg = settings.source("ibkr_flex")
        self._statement_source = statement
        self._token = settings.secret("IBKR_FLEX_TOKEN")
        self._query_id = settings.secret("IBKR_FLEX_QUERY_ID")
        self._base_url = str(cfg.get("base_url", FLEX_BASE))
        self._version = str(cfg.get("version", FLEX_VERSION))
        self._timezone = ZoneInfo(str(cfg.get("statement_timezone", "America/New_York")))
        self._timeout_s = float(cfg.get("request_timeout_s", 60.0))
        self._retry_kwargs = {
            "attempts": int(cfg.get("poll_attempts", 6)),
            "base_delay_s": float(cfg.get("poll_base_delay_s", 3.0)),
            "max_delay_s": float(cfg.get("poll_max_delay_s", 60.0)),
        }
        # ticker -> CIK from everything written down (tracked AND under study), completed in
        # attach_database() with the `companies` registry the SEC filings ingester keeps —
        # which resolves the held positions through the SEC's own ticker map. Until
        # 2026-09-25 only `tracked` was read, and with no thesis written yet every trade,
        # dividend and security was stored with a NULL CIK. A ticker in none of them still
        # resolves to NULL, which section 9.3 prefers to a guess.
        # zfill to ten digits, exactly as the SEC ingesters normalize it: otherwise the
        # same company is keyed "789019" in trades and "0000789019" in observations, and
        # nothing joins (section 9.3).
        self._cik_by_ticker = {
            str(company["ticker"]): str(company["cik"]).zfill(10)
            for company in settings.researched_companies
            if company.get("ticker") and company.get("cik")
        }
        # Cash tolerance for the StmtFunds cross-check. Half a cent: the ledger and the
        # executions are both exact figures, so anything above rounding is a real
        # disagreement (section 10 — a threshold belongs in config).
        self._funds_tolerance = float(cfg.get("funds_tolerance", 0.005))
        self._xml: str | Path = ""
        self._tables: dict[str, list[dict[str, Any]]] = {}
        self._failures: list[str] = []
        self._unresolved: set[str] = set()

    def attach_database(self, conn: Any) -> None:
        """Complete the ticker -> CIK map with the ``companies`` registry (section 9.3).

        That table holds the SEC-resolved CIK of every researched company and every held
        position (``ingest/sec_filings.py``). What config names wins on a conflict: it is the
        user's explicit statement. A fresh database without the table is not a failure.
        """
        try:
            rows = conn.execute("SELECT ticker, cik FROM companies").fetchall()
        except Exception as exc:  # noqa: BLE001 — no registry yet means nothing to add
            log.debug("No companies registry to read: %s", exc)
            return
        for ticker, cik in rows:
            if ticker and cik:
                self._cik_by_ticker.setdefault(str(ticker), str(cik).zfill(10))

    @staticmethod
    def is_available(settings: Settings) -> bool:
        """Whether ibflex is installed and both Flex credentials are configured."""
        try:
            import ibflex  # noqa: F401, PLC0415 — optional extra
        except ImportError:
            return False
        return bool(settings.secret("IBKR_FLEX_TOKEN") and settings.secret("IBKR_FLEX_QUERY_ID"))

    def _resolve_cik(self, ticker: str | None) -> str | None:
        """CIK for a ticker, or None — never a guess (section 9.3)."""
        if not ticker:
            return None
        cik = self._cik_by_ticker.get(str(ticker))
        if cik is None:
            self._unresolved.add(str(ticker))
        return cik

    def _load_statement(self) -> Any:
        """The parsed statement, from the given source or from the Flex Web Service."""
        if self._statement_source is not None:
            self._xml = self._statement_source
            statement, warned = parse_statement(self._statement_source)
        else:
            if not (self._token and self._query_id):
                raise FlexRequestError(
                    "IBKR_FLEX_TOKEN and IBKR_FLEX_QUERY_ID are required to download a "
                    "statement. The token expires and is regenerated in the portal."
                )
            client = FlexClient(
                self._token, self._query_id,
                base_url=self._base_url, version=self._version,
                timeout_s=self._timeout_s, retry_kwargs=self._retry_kwargs,
            )
            self._xml = client.download()
            statement, warned = parse_statement(self._xml)
        if warned:
            log.info(
                "Flex parser skipped %d unknown XML attribute(s); IBKR has added fields "
                "ibflex does not model yet.", warned, extra={"source": SOURCE},
            )
        return statement

    def fetch(self) -> pd.DataFrame:
        """Download (or read) the statement and return positions and NAV as observations."""
        statement = self._load_statement()

        # accountId is deliberately left out of the line: it is an identifier, and a log
        # is the easiest place for one to end up somewhere it should not.
        facts = account_facts(self._xml)
        log.info(
            "Flex statement %s to %s: %s", statement.fromDate, statement.toDate,
            ", ".join(f"{k}={v}" for k, v in facts.items() if k != "accountId") or "no detail",
            extra={"source": SOURCE},
        )

        trades, trade_failures = trade_rows(statement, self._resolve_cik, self._timezone)
        cash, cash_failures = cash_transaction_rows(statement, self._resolve_cik, self._timezone)
        actions, action_failures = corporate_action_rows(statement, self._resolve_cik)
        securities, security_failures = security_rows(statement, self._resolve_cik)

        # StmtFunds is verified, never stored: it holds no fact trades and cash_transactions
        # do not, only the running balance that ties them together. A disagreement means
        # this module parsed something differently from the broker.
        funds_problems = verify_against_funds(
            statement, trades, cash, tolerance=self._funds_tolerance
        )

        self._tables = {
            "trades": trades,
            "cash_transactions": cash,
            "corporate_actions": actions,
            "securities": securities,
            "unmapped_actions": unmapped_action_rows(statement, dt.date.today().isoformat()),
        }
        self._failures = [
            *trade_failures, *cash_failures, *action_failures, *security_failures,
            *funds_problems,
        ]

        if self._unresolved:
            # One summary line, not one per row: a per-row warning for an empty universe
            # would bury the run. The CIK stays NULL in the database either way.
            log.warning(
                "%d ticker(s) did not resolve to a CIK and were stored as NULL: %s. Add "
                "them to universe.watchlist (ticker and CIK) in settings.local.yaml, or let "
                "sec_filings register a held one (section 9.3).",
                len(self._unresolved), sorted(self._unresolved), extra={"source": SOURCE},
            )

        records = observation_records(statement)
        if not records:
            log.warning("The statement carried no positions and no NAV.", extra={"source": SOURCE})
            return empty_observations()
        return self.validate(self.observation_frame(records))

    def fetch_tables(self) -> dict[str, list[dict[str, Any]]]:
        """Trades, cash, corporate actions and the security master, from :meth:`fetch`."""
        return {table: rows for table, rows in self._tables.items() if rows}

    def partial_failures(self) -> list[str]:
        """Rows the parser understood but refused to map. See the module docstring."""
        return list(self._failures)


def trades_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Loader-shaped frame for ``trades`` rows (the loader takes a frame, not records)."""
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def cash_transactions_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Loader-shaped frame for ``cash_transactions`` rows."""
    return pd.DataFrame(rows, columns=CASH_TRANSACTION_COLUMNS)
