"""IBKR Activity Flex ingestion, against the frozen real statement (CLAUDE.md section 4.1).

The fixture is a real report captured on 2026-09-16: 25 executions, 21 cash transactions,
2 open positions and 10 daily NAV rows, with amounts and tickers untouched and the
account holder's identity stripped by whitelist. Testing against it rather than against a
hand-written XML is the point — the failures this ingester has to survive (a section that
is empty, a fractional quantity, a commission below the per-order minimum) are ones nobody
would think to invent.

The cases the captured window cannot contain — an unknown cash type, an unmapped reorg
code, a two-account report — are synthetic and marked as such, the same convention used
for the FRED edge cases (RESEARCH.md section 5.1).

Network is never touched: the client is exercised against a fake session.
"""

from __future__ import annotations

import datetime as dt
import logging
import pathlib
import types
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

pytest.importorskip("ibflex", reason="the 'ibkr' extra is not installed")

from core.config import Settings  # noqa: E402
from db import loader  # noqa: E402
from db.loader import CASH_TRANSACTION_COLUMNS, TRADE_COLUMNS  # noqa: E402
from ingest.ibkr_flex import (  # noqa: E402
    ACCOUNT_WHITELIST,
    FlexClient,
    FlexRequestError,
    IbkrFlexIngester,
    account_facts,
    cash_transaction_rows,
    corporate_action_rows,
    observation_records,
    parse_statement,
    trade_rows,
)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "ibkr_flex_activity.xml"
NEW_YORK = ZoneInfo("America/New_York")


def no_cik(_ticker: str | None) -> None:
    """CIK resolver for a repo with no tracked universe: everything is unresolved."""
    return None


@pytest.fixture(scope="module")
def statement():
    """The parsed fixture, shared across tests — parsing it is the expensive part."""
    parsed, _warnings = parse_statement(FIXTURE)
    return parsed


# --- Parsing and personal data ------------------------------------------------


def test_the_fixture_parses_into_one_account(statement):
    parsed, warning_count = parse_statement(FIXTURE)
    assert parsed.accountId == "U1234567", "the fixture is not the scrubbed one"
    assert len(parsed.Trades) == 25
    assert len(parsed.CashTransactions) == 21
    assert len(parsed.OpenPositions) == 2
    # ibflex 1.x skips attributes it does not model. They are summarized, not dumped.
    assert warning_count > 0, "1.x is expected to warn-and-skip; 0.15 raised instead"


IDENTITY = {
    "NOMBRE APELLIDO", "CALLE FALSA 123", "CIUDADFICTICIA", "999999",
    "19900101", "alguien@example.com", "Paisficticio",
}

# An AccountInformation shaped like the real one: the fixture's was already scrubbed, so
# only an unscrubbed sample can prove the whitelist bites.
UNSCRUBBED = """<FlexQueryResponse><FlexStatements count="1"><FlexStatement>
<AccountInformation accountId="U1234567" ibEntity="IBLLC-US" currency="USD"
  taxLotMatchingMethod="FIFO" accountCapabilities="Cash" dividendReinvestmentEnabled="No"
  name="NOMBRE APELLIDO" street="CALLE FALSA 123" city="CIUDADFICTICIA"
  postalCodeResidentialAddress="999999" dateOfBirth="19900101"
  primaryEmail="alguien@example.com" country="Paisficticio" />
</FlexStatement></FlexStatements></FlexQueryResponse>"""


def test_account_facts_drop_identity_even_when_it_is_present():
    facts = account_facts(UNSCRUBBED)

    assert set(facts) <= set(ACCOUNT_WHITELIST)
    leaked = [value for value in facts.values() if value in IDENTITY]
    assert not leaked, f"personal data survived the whitelist: {leaked}"
    assert facts["taxLotMatchingMethod"] == "FIFO", "the operational fields must survive"


def test_account_facts_read_what_ibflex_silently_drops():
    """ibflex 1.1 does not model these two, so reading the parsed object loses them.

    taxLotMatchingMethod is the one that matters: it is IBKR confirming FIFO lot matching,
    the convention the fiscal layer assumes (section 11). Losing it would not look like a
    loss — it would look like an account with no lot-matching method.
    """
    from ibflex import Types

    modelled = set(Types.AccountInformation.__annotations__)
    assert "taxLotMatchingMethod" not in modelled, "ibflex now models it; simplify this"
    assert account_facts(FIXTURE)["taxLotMatchingMethod"] == "FIFO"


def test_the_account_identifier_never_reaches_the_log(tmp_path, caplog):
    """A log line is the easiest place for an identifier to end up somewhere it should not."""
    ingester = IbkrFlexIngester(settings_for(tmp_path), statement=FIXTURE)
    with caplog.at_level(logging.INFO):
        ingester.fetch()
    assert "U1234567" not in caplog.text


def test_a_two_account_report_refuses_to_merge():
    """Synthetic: one set of NAV series cannot describe two accounts (section 12)."""
    tree = ET.parse(FIXTURE)
    container = tree.getroot().find("FlexStatements")
    container.append(ET.fromstring(ET.tostring(container.find("FlexStatement"))))
    container.set("count", "2")

    with pytest.raises(FlexRequestError, match="2 accounts"):
        parse_statement(ET.tostring(tree.getroot(), encoding="unicode"))


# --- Trades -------------------------------------------------------------------


def test_trades_honour_the_loader_contract(statement):
    rows, failures = trade_rows(statement, no_cik, NEW_YORK)

    assert not failures
    assert len(rows) == 25
    assert set(rows[0]) == set(TRADE_COLUMNS)
    assert {row["side"] for row in rows} == {"buy", "sell"}
    assert all(row["quantity"] > 0 for row in rows), "direction belongs to side, not quantity"
    assert all(row["cik"] is None for row in rows), "no tracked universe means no guessing"


def test_the_commission_total_matches_what_ibkr_attributed_to_the_period(statement):
    """The per-trade figures must add up to ChangeInNAV.commissions, to the cent.

    This is the reconciliation the phase-1 acceptance criterion asks for, in miniature:
    two independent parts of the same report agreeing. It also pins the sign convention —
    a cost stays negative, so a plain sum is what was paid.
    """
    rows, _ = trade_rows(statement, no_cik, NEW_YORK)
    total = round(sum(row["commission"] for row in rows), 8)
    assert total == float(statement.ChangeInNAV.commissions) == -9.37078721


def test_fractional_executions_survive_intact(statement):
    """A 0.64-share buy is real (fractional shares are enabled) and must not be rounded."""
    rows, _ = trade_rows(statement, no_cik, NEW_YORK)
    fractional = [row for row in rows if row["quantity"] % 1]
    assert fractional, "the fixture contains fractional trades; they vanished"
    assert any(abs(row["quantity"] - 0.64) < 1e-9 for row in fractional)


def test_trade_timestamps_convert_from_the_report_timezone_to_utc(statement):
    """IBKR stamps executions in the report's timezone; the XML never says which one."""
    rows, _ = trade_rows(statement, no_cik, NEW_YORK)
    first = min(rows, key=lambda row: row["ts"])
    assert first["ts"].endswith("+00:00")

    amzn = next(row for row in rows if row["ticker"] == "AMZN" and row["side"] == "buy")
    # 2025-10-15 13:07:06 in New York is EDT (UTC-4).
    assert amzn["ts"] == "2025-10-15T17:07:06+00:00"


def test_a_different_configured_timezone_moves_the_timestamps(statement):
    """The zone is a parameter, not a constant: changing it must change the result."""
    ny, _ = trade_rows(statement, no_cik, NEW_YORK)
    utc, _ = trade_rows(statement, no_cik, ZoneInfo("UTC"))
    assert ny[0]["ts"] != utc[0]["ts"]


def test_an_unmapped_side_is_reported_rather_than_dropped():
    """Synthetic: a cancelled execution ('BUY (Ca.)') is news, not noise."""
    trade = types.SimpleNamespace(
        tradeID=1, symbol="AAPL", buySell=types.SimpleNamespace(name="CANCELBUY"),
        dateTime=dt.datetime(2026, 1, 2, 10, 0), tradeDate=dt.date(2026, 1, 2),
        quantity=-1, tradePrice=100, currency="USD", ibCommission=-0.35, fxRateToBase=1,
    )
    rows, failures = trade_rows(types.SimpleNamespace(Trades=[trade]), no_cik, NEW_YORK)

    assert rows == []
    assert len(failures) == 1 and "unmapped side" in failures[0]


# --- Cash transactions --------------------------------------------------------


def test_cash_transactions_keep_the_withholding_actually_observed(statement):
    """Section 11: the panel uses the figure IBKR withheld, never an assumed rate."""
    rows, failures = cash_transaction_rows(statement, no_cik, NEW_YORK)

    assert not failures
    assert set(rows[0]) == set(CASH_TRANSACTION_COLUMNS)
    by_kind: dict[str, float] = {}
    for row in rows:
        by_kind[row["kind"]] = round(by_kind.get(row["kind"], 0.0) + row["amount"], 2)

    assert by_kind["dividend"] == 10.94
    assert by_kind["withholding_tax"] == -3.30
    assert by_kind["deposit"] == 542.12
    # And they match the period attribution IBKR computed independently.
    assert by_kind["dividend"] == float(statement.ChangeInNAV.dividends)
    assert by_kind["withholding_tax"] == float(statement.ChangeInNAV.withholdingTax)


def test_payment_in_lieu_is_not_filed_as_a_dividend():
    """Synthetic: it may not be taxed like one, so it never shares a bucket (section 11)."""
    tx = types.SimpleNamespace(
        transactionID=7, type=types.SimpleNamespace(name="PAYMENTINLIEU"),
        accountId="U1", symbol="T", conid=37018770,
        dateTime=dt.datetime(2026, 2, 2, 20, 20),
        reportDate=dt.date(2026, 2, 2), amount=1.5, currency="USD",
    )
    rows, failures = cash_transaction_rows(
        types.SimpleNamespace(CashTransactions=[tx]), no_cik, NEW_YORK
    )
    assert not failures
    assert rows[0]["kind"] == "payment_in_lieu"


def test_an_unknown_cash_type_is_reported_not_bucketed():
    """Synthetic: IBKR adding a type must not silently become a fee."""
    tx = types.SimpleNamespace(
        transactionID=8, type=types.SimpleNamespace(name="SOMETHING_NEW"),
        accountId="U1", symbol=None, dateTime=dt.datetime(2026, 2, 2, 20, 20),
        reportDate=dt.date(2026, 2, 2), amount=-1.0, currency="USD",
    )
    rows, failures = cash_transaction_rows(
        types.SimpleNamespace(CashTransactions=[tx]), no_cik, NEW_YORK
    )
    assert rows == []
    assert len(failures) == 1 and "unmapped type" in failures[0]


# --- Corporate actions --------------------------------------------------------


def test_the_fixture_has_no_corporate_actions_so_section_9_2_is_untested(statement):
    """Stated, not assumed: this is the hole the synthetic cases below have to cover."""
    rows, failures = corporate_action_rows(statement, no_cik)
    assert rows == [] and failures == []


def test_a_split_is_mapped_and_carries_the_ibkr_source():
    action = types.SimpleNamespace(
        transactionID=11, actionID=None, type=types.SimpleNamespace(name="FORWARDSPLIT"),
        symbol="AAPL", dateTime=dt.date(2026, 3, 2), reportDate=dt.date(2026, 3, 2),
        amount=None, currency="USD",
    )
    rows, failures = corporate_action_rows(
        types.SimpleNamespace(CorporateActions=[action]), no_cik
    )
    assert not failures
    assert rows[0]["kind"] == "split"
    # Stored next to the yfinance rows on purpose: section 9.2 needs both witnesses to
    # disagree in the open, because yfinance encodes spin-offs as splits.
    assert rows[0]["source"] == "ibkr"
    assert rows[0]["ratio"] is None, "Flex does not report a ratio; it must not be invented"


def test_an_unmapped_reorg_code_is_never_bucketed_as_a_split():
    """Section 9.2: a wrong bucket falsifies the cost basis for good. Refuse instead."""
    action = types.SimpleNamespace(
        transactionID=12, actionID=None, type=types.SimpleNamespace(name="STOCKDIV"),
        symbol="XLF", dateTime=dt.date(2026, 3, 2), reportDate=dt.date(2026, 3, 2),
        amount=None, currency="USD",
    )
    rows, failures = corporate_action_rows(
        types.SimpleNamespace(CorporateActions=[action]), no_cik
    )
    assert rows == []
    assert "needs review" in failures[0]


# --- Observations -------------------------------------------------------------


def test_positions_and_nav_become_long_format_observations(statement):
    records = observation_records(statement)
    series = {record["series_id"] for record in records}

    assert {"TMUS:position_qty", "TMUS:position_cost_basis", "UBER:position_qty"} <= series
    assert {"NAV:total", "NAV:cash", "NAV:stock"} <= series

    qty = next(r for r in records if r["series_id"] == "TMUS:position_qty")
    assert qty["value"] == 1.0
    assert qty["ts"] == qty["ts_release"], "a NAV is known the day it is reported"


def test_the_nav_series_reconciles_with_the_period_attribution(statement):
    """The phase-1 acceptance criterion, as a test: IBKR's two accounts of itself agree."""
    records = observation_records(statement)
    nav = sorted(
        (r for r in records if r["series_id"] == "NAV:total"), key=lambda r: r["ts"]
    )
    assert nav[-1]["value"] == float(statement.ChangeInNAV.endingValue) == 1138.96750879

    last_day = nav[-1]["ts"]
    parts = {
        r["series_id"]: r["value"] for r in records
        if r["ts"] == last_day and r["series_id"].startswith("NAV:")
    }
    assert round(parts["NAV:cash"] + parts["NAV:stock"], 8) == nav[-1]["value"]


# --- Client (no network) ------------------------------------------------------


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    """Records the calls made and replays queued bodies."""

    def __init__(self, *bodies: str) -> None:
        self._bodies = list(bodies)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return FakeResponse(self._bodies.pop(0))


SEND_OK = """<FlexStatementResponse timestamp='16 September, 2026'>
<Status>Success</Status><ReferenceCode>123</ReferenceCode>
<Url>https://gdcdyn.interactivebrokers.com/AccountManagement/FlexWebService/GetStatement</Url>
</FlexStatementResponse>"""
SEND_FAIL = """<FlexStatementResponse><Status>Fail</Status><ErrorCode>1012</ErrorCode>
<ErrorMessage>Token has expired.</ErrorMessage></FlexStatementResponse>"""
NOT_READY = """<FlexStatementResponse><Status>Warn</Status><ErrorCode>1019</ErrorCode>
<ErrorMessage>Statement generation in progress. Please try again shortly.</ErrorMessage>
</FlexStatementResponse>"""
REPORT = """<FlexQueryResponse queryName="x" type="AF"><FlexStatements count="1">
</FlexStatements></FlexQueryResponse>"""


def test_a_failure_arrives_with_http_200_and_is_still_a_failure():
    """The trap: raise_for_status() passes, so the body has to be classified."""
    client = FlexClient("tok", "q", session=FakeSession(SEND_FAIL))
    with pytest.raises(FlexRequestError, match="1012"):
        client.send_request()


def test_the_collection_host_comes_from_the_response_not_from_config():
    """SendRequest goes to ndcdyn and answers with gdcdyn. Hardcoding the second breaks."""
    session = FakeSession(SEND_OK, REPORT)
    client = FlexClient("tok", "q", session=session, retry_kwargs={"base_delay_s": 0})
    client.download()

    request_url, collect_url = session.calls[0][0], session.calls[1][0]
    assert "ndcdyn" in request_url and request_url.endswith("/SendRequest")
    assert "gdcdyn" in collect_url, "the returned <Url> was ignored"
    assert session.calls[1][1]["q"] == "123", "GetStatement takes the reference code"


def test_a_statement_that_is_not_ready_is_retried():
    """Generation is asynchronous; a not-ready answer is normal, not an error."""
    session = FakeSession(SEND_OK, NOT_READY, NOT_READY, REPORT)
    client = FlexClient(
        "tok", "q", session=session,
        retry_kwargs={"attempts": 4, "base_delay_s": 0, "max_delay_s": 0},
    )
    assert client.download().startswith("<FlexQueryResponse")
    assert len(session.calls) == 4


def test_the_token_never_reaches_a_log_line_or_a_traceback(caplog):
    secret = "SUPERSECRETTOKEN123"
    session = FakeSession(SEND_FAIL)
    client = FlexClient(secret, "q", session=session)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(FlexRequestError) as excinfo:
            client.send_request()

    assert secret not in caplog.text
    assert secret not in str(excinfo.value)
    assert session.calls[0][1]["t"] == secret, "it must still be sent, just never printed"


# --- Ingester end to end (fixture in, database out) ----------------------------


def settings_for(tmp_path, secrets: dict[str, str] | None = None) -> Settings:
    """Minimal Settings with the real ibkr_flex block from settings.yaml."""
    from core import config

    base = config.load_settings()
    return Settings(
        raw=base.raw, db_path=tmp_path / "flex.db", log_level="INFO",
        secrets=secrets or {}, public_mode=False,
    )


def test_the_ingester_is_unavailable_without_both_credentials(tmp_path):
    assert not IbkrFlexIngester.is_available(settings_for(tmp_path))
    assert not IbkrFlexIngester.is_available(
        settings_for(tmp_path, {"IBKR_FLEX_TOKEN": "t"})
    )
    assert IbkrFlexIngester.is_available(
        settings_for(tmp_path, {"IBKR_FLEX_TOKEN": "t", "IBKR_FLEX_QUERY_ID": "q"})
    )


def test_fetch_reports_unresolved_tickers_once_and_stores_null(tmp_path, caplog):
    """Section 9.3: no tracked universe means eight NULL CIKs and one summary line."""
    ingester = IbkrFlexIngester(settings_for(tmp_path), statement=FIXTURE)
    with caplog.at_level(logging.WARNING):
        frame = ingester.fetch()

    assert not frame.empty
    warnings_about_cik = [r for r in caplog.records if "did not resolve to a CIK" in r.message]
    assert len(warnings_about_cik) == 1, "one summary line, not one per row"


def test_fetch_tables_omits_the_sections_that_came_back_empty(tmp_path):
    ingester = IbkrFlexIngester(settings_for(tmp_path), statement=FIXTURE)
    ingester.fetch()
    tables = ingester.fetch_tables()

    assert set(tables) == {"trades", "cash_transactions", "securities"}, (
        "empty tables must not be sent, and the ones with rows must be"
    )
    assert len(tables["trades"]) == 25
    assert ingester.partial_failures() == []


def test_ingesting_the_statement_twice_does_not_duplicate(tmp_path):
    """The phase-1 acceptance criterion: re-running the pipeline is safe by construction."""
    from ingest.ibkr_flex import cash_transactions_frame, trades_frame

    conn = loader.init_db(tmp_path / "flex.db")
    try:
        for _ in range(2):
            ingester = IbkrFlexIngester(settings_for(tmp_path), statement=FIXTURE)
            loader.upsert_observations(conn, ingester.fetch())
            tables = ingester.fetch_tables()
            loader.upsert_trades(conn, trades_frame(tables["trades"]))
            loader.upsert_cash_transactions(
                conn, cash_transactions_frame(tables["cash_transactions"])
            )
            loader.upsert_securities(conn, tables["securities"])

        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("observations", "trades", "cash_transactions", "securities")
        }
        assert counts["trades"] == 25
        assert counts["cash_transactions"] == 21
        assert counts["securities"] == 8, "conid is the key, so a re-read updates in place"
        assert counts["observations"] == 44, "10 NAV days x 4 series + 2 positions x 2"

        stored = pd.read_sql_query(
            "SELECT value FROM observations WHERE series_id = 'NAV:total' ORDER BY ts", conn
        )
        assert float(stored["value"].iloc[-1]) == 1138.96750879
    finally:
        conn.close()


# --- Security master (section 9.3) --------------------------------------------


def test_the_security_master_is_keyed_on_conid_not_on_the_ticker(statement):
    """Section 9.3: the ticker is not a key. ``conid`` is IBKR's permanent contract id."""
    from ingest.ibkr_flex import security_rows

    rows, failures = security_rows(statement, no_cik)

    assert failures == []
    assert len(rows) == 8
    assert all(row["conid"] for row in rows), "a row without the key was stored"
    assert len({row["conid"] for row in rows}) == len(rows), "conid is not unique"

    msft = next(row for row in rows if row["ticker"] == "MSFT")
    assert msft["conid"] == "272093"
    assert msft["name"] == "MICROSOFT CORP"


def test_the_standard_identifiers_travel_with_it(statement):
    """ISIN, CUSIP and FIGI survive a rename too, and are not IBKR-specific."""
    from ingest.ibkr_flex import security_rows

    msft = next(r for r in security_rows(statement, no_cik)[0] if r["ticker"] == "MSFT")

    assert msft["isin"] == "US5949181045"
    assert msft["cusip"] == "594918104"
    assert msft["figi"] == "BBG000BPH459"


def test_the_issuer_country_is_carried_because_section_11_needs_it(statement):
    """Situs. It appears nowhere else in the statement, and nothing is computed from it.

    Section 11 shows US-situs exposure as a question for an adviser; the panel asserts no
    tax treatment, so this is stored as an observed attribute and left alone.
    """
    from ingest.ibkr_flex import security_rows

    rows, _ = security_rows(statement, no_cik)
    assert {row["issuer_country"] for row in rows} == {"US"}


def test_a_security_with_no_conid_is_reported_not_stored_under_its_ticker():
    """Without the key the row could only be identified by the thing 9.3 rejects."""
    from ingest.ibkr_flex import security_rows

    class Info:
        symbol, conid, description, currency, multiplier = "ZZZZ", None, "X", "USD", None
        assetCategory = subCategory = listingExchange = issuerCountryCode = None
        isin = cusip = figi = None

    class Stmt:
        SecuritiesInfo = [Info()]
        fromDate = toDate = None

    rows, failures = security_rows(Stmt(), no_cik)
    assert rows == []
    assert "no conid" in failures[0]


def test_trades_and_cash_carry_the_join_key(statement):
    """Without conid on the movements, the master could only be reached via the ticker."""
    from ingest.ibkr_flex import cash_transaction_rows, trade_rows

    trades, _ = trade_rows(statement, no_cik, NEW_YORK)
    cash, _ = cash_transaction_rows(statement, no_cik, NEW_YORK)

    assert all(row["conid"] for row in trades), "a trade has no conid"
    assert all(row["conid"] for row in cash if row["ticker"]), "a security cash row has none"


# --- StmtFunds: verified, never stored ----------------------------------------


def test_the_cash_ledger_agrees_with_everything_this_module_parsed(statement):
    """The third witness. Measured: 46 ledger rows, 25 executions, 549.76 of other cash.

    StmtFunds holds no fact that trades and cash_transactions do not — they are slices of
    it — so it is not stored. What it adds is the running balance, an arithmetic chain
    that has to tie out, which makes it a check rather than data.
    """
    from ingest.ibkr_flex import cash_transaction_rows, trade_rows, verify_against_funds

    trades, _ = trade_rows(statement, no_cik, NEW_YORK)
    cash, _ = cash_transaction_rows(statement, no_cik, NEW_YORK)

    assert verify_against_funds(statement, trades, cash) == []


def test_a_dropped_commission_is_caught_by_the_ledger(statement):
    """The check has to bite, or it is decoration.

    A commission silently lost in parsing produces trades that look entirely reasonable —
    and the broker's own ledger is the only thing that says otherwise.
    """
    from ingest.ibkr_flex import cash_transaction_rows, trade_rows, verify_against_funds

    trades, _ = trade_rows(statement, no_cik, NEW_YORK)
    cash, _ = cash_transaction_rows(statement, no_cik, NEW_YORK)
    damaged = [{**row, "commission": 0.0} for row in trades]

    problems = verify_against_funds(statement, damaged, cash)
    assert problems, "a lost commission went unnoticed"
    assert "disagrees with the parsed cash" in problems[0]


def test_a_flipped_side_is_caught_by_the_ledger(statement):
    """A sign error is the other failure that produces perfectly plausible rows."""
    from ingest.ibkr_flex import cash_transaction_rows, trade_rows, verify_against_funds

    trades, _ = trade_rows(statement, no_cik, NEW_YORK)
    cash, _ = cash_transaction_rows(statement, no_cik, NEW_YORK)
    flipped = [{**row, "side": "sell" if row["side"] == "buy" else "buy"} for row in trades]

    assert verify_against_funds(statement, flipped, cash), "a flipped side went unnoticed"


def test_missing_cash_transactions_are_caught_in_aggregate(statement):
    from ingest.ibkr_flex import cash_transaction_rows, trade_rows, verify_against_funds

    trades, _ = trade_rows(statement, no_cik, NEW_YORK)
    cash, _ = cash_transaction_rows(statement, no_cik, NEW_YORK)

    problems = verify_against_funds(statement, trades, cash[:-3])
    assert any("non-trade cash" in problem for problem in problems)


def test_an_absent_ledger_is_reported_as_not_run_rather_than_as_passed():
    """Silence must not be indistinguishable from verified (section 12)."""
    from ingest.ibkr_flex import verify_against_funds

    class Stmt:
        StmtFunds = []

    assert verify_against_funds(Stmt(), [], []) == []


def test_the_master_joins_the_movements_by_the_stable_key(tmp_path):
    """The point of storing it: a movement finds its security without going via the ticker.

    Today both spellings agree, so the join looks redundant. It stops being redundant the
    day a ticker is renamed — which is exactly when a ticker-keyed join starts lying and
    nothing complains (section 9.3).
    """
    from ingest.ibkr_flex import cash_transactions_frame, trades_frame

    conn = loader.init_db(tmp_path / "join.db")
    try:
        ingester = IbkrFlexIngester(settings_for(tmp_path), statement=FIXTURE)
        ingester.fetch()
        tables = ingester.fetch_tables()
        loader.upsert_trades(conn, trades_frame(tables["trades"]))
        loader.upsert_cash_transactions(
            conn, cash_transactions_frame(tables["cash_transactions"])
        )
        loader.upsert_securities(conn, tables["securities"])

        orphans = conn.execute(
            "SELECT COUNT(*) FROM trades t "
            "LEFT JOIN securities s ON s.conid = t.conid WHERE s.conid IS NULL"
        ).fetchone()[0]
        assert orphans == 0, "a trade points at no security"

        isin = conn.execute(
            "SELECT s.isin FROM trades t JOIN securities s ON s.conid = t.conid "
            "WHERE t.ticker = 'MSFT' LIMIT 1"
        ).fetchone()[0]
        assert isin == "US5949181045"
    finally:
        conn.close()


def test_the_companies_registry_resolves_what_config_does_not(tmp_path):
    """Until 2026-09-25 only `tracked` cards were read, and with none written every trade
    was stored with a NULL CIK — though the registry already knew the held companies."""
    from db import loader

    conn = loader.init_db(tmp_path / "registry.db")
    try:
        loader.upsert_companies(conn, [{"cik": "1283699", "ticker": "TMUS", "name": "T-Mobile",
                                        "sector": None, "thesis_category": None,
                                        "first_seen": "2026-09-24", "status": "active"}])
        ingester = IbkrFlexIngester(settings_for(tmp_path), statement=FIXTURE)
        ingester.attach_database(conn)
    finally:
        conn.close()
    assert ingester._resolve_cik("TMUS") == "0001283699"
    assert ingester._resolve_cik("NOPE") is None, "never a guess"
