"""Level 4 transformations: valuation, FIFO lots, income and reconciliation.

The headline tests are the two that make this module's own arithmetic collide with the
broker's: :func:`test_the_fifo_engine_reproduces_the_brokers_realized_pnl` and
:func:`test_average_cost_matches_the_brokers_cost_basis`. Both run against the frozen real
statement, so they compare against figures IBKR computed independently, not against numbers
this project invented and then asserted.

The cases the captured window cannot contain — a sale with no purchase in it, an unpriced
position, a NAV that disagrees — are synthetic and marked as such.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from transform import portfolio
from transform.portfolio import (
    average_cost,
    concentration,
    cost_drag,
    fifo_lots,
    income_summary,
    latest_positions,
    latest_prices,
    nav_series,
    realized_pnl,
    reconcile_nav,
    target_drift,
    thesis_metadata,
    unpriced,
    valuation,
)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "ibkr_flex_activity.xml"


@pytest.fixture(scope="module")
def account():
    """Trades, cash and observations as the Flex ingester would have written them."""
    pytest.importorskip("ibflex", reason="the 'ibkr' extra is not installed")
    from zoneinfo import ZoneInfo

    from ingest.ibkr_flex import (
        cash_transaction_rows,
        observation_records,
        parse_statement,
        trade_rows,
    )

    statement, _ = parse_statement(FIXTURE)
    zone = ZoneInfo("America/New_York")
    trades, _ = trade_rows(statement, lambda _t: None, zone)
    cash, _ = cash_transaction_rows(statement, lambda _t: None, zone)
    return {
        "statement": statement,
        "trades": pd.DataFrame(trades),
        "cash": pd.DataFrame(cash),
        "observations": pd.DataFrame(observation_records(statement)),
    }


def observations_frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Build a long-format frame from ``(series_id, ts, value)`` triples."""
    return pd.DataFrame(
        [
            {"source": "test", "series_id": s, "ts": t, "ts_release": t, "value": v}
            for s, t, v in rows
        ]
    )


# --- The two collisions with the broker ---------------------------------------


def test_the_fifo_engine_reproduces_the_brokers_realized_pnl(account):
    """Per symbol, against IBKR's own FIFO summary, to less than a millionth of a dollar.

    This is what justifies having an engine at all instead of copying IBKR's number: an
    independent derivation that lands on the same figure is evidence the lot matching,
    the commission allocation on both legs and the ordering are all right. The fiscal
    layer inherits these lots, so an error here would be silent and expensive.
    """
    mine = realized_pnl(account["trades"]).set_index("ticker")["realized_pnl"]
    theirs = {
        row.symbol: float(row.totalRealizedPnl)
        for row in account["statement"].FIFOPerformanceSummaryInBase
        if row.symbol
    }
    assert theirs, "the fixture lost its FIFO summary"

    for symbol, reported in theirs.items():
        computed = float(mine.get(symbol, 0.0))
        assert computed == pytest.approx(reported, abs=1e-6), f"{symbol} disagrees"


def test_average_cost_matches_the_brokers_cost_basis(account):
    """Commission-inclusive cost of the open lots, against IBKR's costBasisMoney."""
    lots, _disposals, unmatched = fifo_lots(account["trades"])
    assert not unmatched
    mine = average_cost(lots).set_index("ticker")["cost"]

    for position in account["statement"].OpenPositions:
        assert float(mine[position.symbol]) == pytest.approx(
            float(position.costBasisMoney), abs=1e-6
        ), f"{position.symbol}: cost basis drifted from the broker's"


def test_the_fifo_result_does_not_depend_on_row_order(account):
    """Rows arrive from a database with no guaranteed order; the matching must not care."""
    shuffled = account["trades"].sample(frac=1.0, random_state=7)
    assert realized_pnl(shuffled).equals(realized_pnl(account["trades"]))


def test_fractional_lots_are_matched_without_rounding(account):
    """Fractional shares are enabled on this account: 0.64 must not become 1 or 0."""
    _lots, disposals, _unmatched = fifo_lots(account["trades"])
    fractional = disposals[disposals["quantity"] % 1 != 0]
    assert not fractional.empty, "the fractional disposals vanished"
    assert (fractional["quantity"] > 0).all()


def test_holding_days_are_available_for_the_fiscal_layer(account):
    """Section 11 needs an acquisition date per lot, not just a PnL total."""
    _lots, disposals, _unmatched = fifo_lots(account["trades"])
    assert (disposals["holding_days"] >= 0).all()
    assert disposals["acquired_ts"].notna().all()


def test_a_sale_with_no_purchase_is_reported_not_priced_at_zero():
    """Synthetic: the statement window starts after the position was opened.

    A zero-cost lot would turn missing history into a spectacular fake profit, which is
    exactly the plausible-and-wrong output section 12 forbids.
    """
    trades = pd.DataFrame([
        {"trade_id": "1", "ticker": "AAPL", "ts": "2026-03-02T14:00:00+00:00",
         "side": "sell", "quantity": 5.0, "price": 200.0, "commission": -0.35},
    ])
    _lots, disposals, unmatched = fifo_lots(trades)

    assert disposals.empty, "an unmatched sale must not become a disposal"
    assert len(unmatched) == 1 and "AAPL" in unmatched[0]
    assert realized_pnl(trades).empty, "no realized PnL can be stated for it"


# --- Valuation and weights ----------------------------------------------------


def test_valuation_keeps_an_unpriced_position_visible(account):
    positions = latest_positions(account["observations"])
    prices = pd.DataFrame([{"ticker": "TMUS", "price": 180.47, "price_ts": "2026-09-15"}])

    valued = valuation(positions, prices)
    assert set(valued["ticker"]) == {"TMUS", "UBER"}, "the unpriced row disappeared"
    assert pd.isna(valued.loc[valued["ticker"] == "UBER", "market_value"].iloc[0])
    assert unpriced(valued) == ["UBER"]


def test_weights_and_unrealized_pnl_are_computed_over_what_is_priced(account):
    positions = latest_positions(account["observations"])
    prices = pd.DataFrame([
        {"ticker": "TMUS", "price": 180.47, "price_ts": "2026-09-15"},
        {"ticker": "UBER", "price": 71.43, "price_ts": "2026-09-15"},
    ])
    valued = valuation(positions, prices)

    assert unpriced(valued) == []
    assert valued["weight"].sum() == pytest.approx(1.0)
    tmus = valued[valued["ticker"] == "TMUS"].iloc[0]
    # 1 share at 180.47 against a 216.949179 cost basis: a loss, and it must read as one.
    assert tmus["unrealized_pnl"] == pytest.approx(180.47 - 216.949179, abs=1e-6)
    assert tmus["unrealized_return"] < 0


def test_drift_counts_a_target_that_was_never_bought(account):
    """An unfilled target is a drift, not an absence: it is the thing left to do."""
    positions = latest_positions(account["observations"])
    prices = pd.DataFrame([
        {"ticker": "TMUS", "price": 180.47, "price_ts": "2026-09-15"},
        {"ticker": "UBER", "price": 71.43, "price_ts": "2026-09-15"},
    ])
    valued = valuation(positions, prices)

    drift = target_drift(valued, {"TMUS": 0.3, "UBER": 0.3, "MSFT": 0.4}).set_index("ticker")
    assert drift.loc["MSFT", "weight"] == 0.0
    assert drift.loc["MSFT", "drift"] == pytest.approx(-0.4)
    assert drift.loc["TMUS", "drift"] > 0, "an overweight position must show positive drift"


def test_no_targets_means_no_drift_table_rather_than_zeros(account):
    """Section 12: with nothing written down, the honest answer is an empty table."""
    assert target_drift(pd.DataFrame(), None).empty
    assert target_drift(pd.DataFrame(), {}).empty


# --- Income -------------------------------------------------------------------


def test_income_nets_withholding_and_reports_the_observed_rate(account):
    income = income_summary(account["cash"], account["trades"])

    assert income.dividends_gross == pytest.approx(10.94)
    assert income.withholding == pytest.approx(-3.30)
    assert income.dividends_net == pytest.approx(7.64)
    assert income.commissions == pytest.approx(-9.37078721)
    assert income.deposits == pytest.approx(542.12)
    # Observed on this account, not a tax rate this project asserts (section 11).
    assert income.effective_withholding_rate == pytest.approx(0.3016, abs=5e-4)


def test_a_payment_in_lieu_is_never_added_to_dividends():
    cash = pd.DataFrame([
        {"kind": "dividend", "amount": 10.0},
        {"kind": "payment_in_lieu", "amount": 4.0},
    ])
    income = income_summary(cash)
    assert income.dividends_gross == 10.0
    assert income.payments_in_lieu == 4.0


def test_income_without_dividends_has_no_withholding_rate():
    income = income_summary(pd.DataFrame(columns=["kind", "amount"]))
    assert income.effective_withholding_rate is None, "a rate over zero dividends is not 0"


def test_cost_drag_is_a_ratio_so_it_can_be_published(account):
    """9,37 USD means nothing without a denominator; 0,82 % of NAV is the real reading."""
    income = income_summary(account["cash"], account["trades"])
    drag = cost_drag(income, 1138.96750879)
    assert drag == pytest.approx(-0.00823, abs=1e-5)
    assert cost_drag(income, None) is None


# --- Reconciliation against the broker ----------------------------------------


def test_reconciliation_compares_both_sides_on_the_statement_date():
    """Synthetic: a later price must not be used against an earlier NAV.

    Comparing today's close with last week's NAV manufactures a discrepancy and hides a
    real one, so the price is taken as of the statement date.
    """
    observations = observations_frame([
        ("TMUS:position_qty", "2026-09-15T00:00:00+00:00", 2.0),
        ("TMUS:position_cost_basis", "2026-09-15T00:00:00+00:00", 300.0),
        ("NAV:stock", "2026-09-15T00:00:00+00:00", 360.0),
        ("TMUS:close_raw", "2026-09-15T00:00:00+00:00", 180.0),
        ("TMUS:close_raw", "2026-09-16T00:00:00+00:00", 900.0),   # must be ignored
    ])
    result = reconcile_nav(observations)

    assert result.as_of.startswith("2026-09-15")
    assert result.computed == pytest.approx(360.0)
    assert result.agrees is True


def test_a_gap_beyond_tolerance_is_flagged_and_never_adjusted():
    observations = observations_frame([
        ("TMUS:position_qty", "2026-09-15T00:00:00+00:00", 1.0),
        ("TMUS:position_cost_basis", "2026-09-15T00:00:00+00:00", 200.0),
        ("NAV:stock", "2026-09-15T00:00:00+00:00", 180.0),
        ("TMUS:close_raw", "2026-09-15T00:00:00+00:00", 200.0),
    ])
    result = reconcile_nav(observations, tolerance=0.005)

    assert result.agrees is False
    assert result.computed == pytest.approx(200.0), "the computed side must stay untouched"
    assert result.reported == pytest.approx(180.0), "IBKR's side must stay untouched"
    assert result.relative == pytest.approx(20 / 180)


def test_missing_data_reconciles_to_unknown_not_to_agreement():
    """``None`` is not ``True``: an unknown must never render as a tick in the panel."""
    assert reconcile_nav(pd.DataFrame()).agrees is None

    no_price = observations_frame([
        ("TMUS:position_qty", "2026-09-15T00:00:00+00:00", 1.0),
        ("NAV:stock", "2026-09-15T00:00:00+00:00", 180.0),
    ])
    result = reconcile_nav(no_price)
    assert result.agrees is None
    assert "TMUS" in result.detail


def test_the_nav_curve_comes_out_in_chronological_order(account):
    curve = nav_series(account["observations"])
    assert len(curve) == 10, "the trimmed fixture carries ten NAV days"
    assert curve["ts"].is_monotonic_increasing
    assert curve["value"].iloc[-1] == pytest.approx(1138.96750879)


def test_latest_prices_takes_the_most_recent_row_per_ticker():
    observations = observations_frame([
        ("SPY:close_raw", "2026-09-14T00:00:00+00:00", 500.0),
        ("SPY:close_raw", "2026-09-15T00:00:00+00:00", 510.0),
    ])
    prices = latest_prices(observations)
    assert prices.loc[0, "price"] == 510.0


# --- Concentration (section 5.3) ----------------------------------------------


def test_concentration_groups_by_thesis_and_exposes_the_unclassified(account):
    """Eight tickers from five sectors can be one bet; the thesis axis is the real one."""
    positions = latest_positions(account["observations"])
    prices = pd.DataFrame([
        {"ticker": "TMUS", "price": 180.47, "price_ts": "2026-09-15"},
        {"ticker": "UBER", "price": 71.43, "price_ts": "2026-09-15"},
    ])
    valued = valuation(positions, prices)
    metadata = thesis_metadata([
        {"ticker": "TMUS", "sector": "Communication Services",
         "thesis_category": "infraestructura de red"},
    ])

    grouped = concentration(valued, metadata).set_index("group")
    assert grouped.loc["infraestructura de red", "positions"] == 1
    assert "sin clasificar" in grouped.index, "a position with no thesis card must be visible"
    assert grouped["weight"].sum() == pytest.approx(1.0)


def test_concentration_without_any_thesis_card_says_so(account):
    positions = latest_positions(account["observations"])
    prices = pd.DataFrame([
        {"ticker": "TMUS", "price": 180.47, "price_ts": "2026-09-15"},
        {"ticker": "UBER", "price": 71.43, "price_ts": "2026-09-15"},
    ])
    grouped = concentration(valuation(positions, prices), None)
    assert list(grouped["group"]) == ["sin clasificar"]


# --- A position sold in full (found 2026-09-24, before a sale triggered it) --------------


def _account(rows):
    return pd.DataFrame([
        {"source": "ibkr", "series_id": sid, "ts": ts, "ts_release": ts, "value": value}
        for sid, ts, value in rows
    ])


def test_a_position_absent_from_the_latest_statement_is_closed():
    """Flex lists what is open and says nothing about what was sold, so the sold one keeps
    its old row as its last word."""
    account = _account([
        ("NAV:total", "2026-09-16", 300.0), ("NAV:total", "2026-10-01", 310.0),
        ("TMUS:position_qty", "2026-09-16", 1.0), ("UBER:position_qty", "2026-09-16", 2.0),
        ("UBER:position_qty", "2026-10-01", 2.0),
    ])
    assert list(portfolio.latest_positions(account)["ticker"]) == ["UBER"]


def test_everything_sold_is_an_empty_portfolio_not_the_last_one():
    """Dated by the NAV, which every statement carries even with nothing held."""
    account = _account([
        ("NAV:total", "2026-09-16", 300.0), ("NAV:total", "2026-10-01", 290.0),
        ("TMUS:position_qty", "2026-09-16", 1.0),
    ])
    assert portfolio.latest_positions(account).empty


def test_held_tickers_for_the_ingesters_follow_the_same_rule(tmp_path):
    from db import loader
    from ingest.base import held_tickers

    conn = loader.init_db(tmp_path / "held.db")
    try:
        loader.upsert_observations(conn, _account([
            ("NAV:total", "2026-10-01", 310.0),
            ("TMUS:position_qty", "2026-09-16", 1.0), ("UBER:position_qty", "2026-10-01", 2.0),
        ]))
        assert held_tickers(conn) == ["UBER"]
    finally:
        conn.close()


def test_the_commission_shares_add_up_to_the_realized_result():
    """Split apart for the local-currency layer, and still IBKR's number when summed."""
    trades = pd.DataFrame([
        {"trade_id": "1", "ticker": "X", "ts": "2025-01-02", "side": "buy", "quantity": 2.0,
         "price": 100.0, "commission": -1.0},
        {"trade_id": "2", "ticker": "X", "ts": "2025-06-02", "side": "sell", "quantity": 1.0,
         "price": 120.0, "commission": -0.5},
    ])
    _, disposals, _ = portfolio.fifo_lots(trades)
    row = disposals.iloc[0]
    assert row["buy_commission"] == pytest.approx(-0.5)
    assert row["sell_commission"] == pytest.approx(-0.5)
    assert row["proceeds"] - row["cost"] + row["buy_commission"] + row["sell_commission"] \
        == pytest.approx(row["realized_pnl"])
