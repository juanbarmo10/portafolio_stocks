"""IBKR against yfinance, and the ``needs_review`` flag (CLAUDE.md section 9.2).

What these tests defend is a refusal. The two providers disagree, the disagreement is
recorded rather than resolved, and the position is flagged for a human. Nothing here is
allowed to pick a winner and carry on, because the failure mode is silent and permanent: a
spin-off recorded as a split leaves the cost base whole on the original entity and
falsifies the PnL for as long as the position exists.

The XLF case is real and measured (2016-09-19, ratio 1.231). The IBKR side is synthetic —
the frozen statement's ``CorporateActions`` section is empty, which is itself a fact worth
stating: this fixture cannot cover section 9.2, and that limitation is recorded in
CLAUDE.md section 13.
"""

from __future__ import annotations

import pandas as pd
import pytest

from transform.corporate_actions import (
    data_quality_notes,
    reconcile_ticker,
    review_index,
    review_positions,
)


def action(ticker, kind, ex_date, source, ratio=None, amount=None) -> dict:
    return {"action_id": f"{source}:{ticker}:{kind}:{ex_date}", "cik": None,
            "ticker": ticker, "kind": kind, "ex_date": ex_date, "ratio": ratio,
            "amount": amount, "currency": "USD", "source": source}


def positions(*tickers) -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": t, "quantity": 1.0, "cost_basis": 100.0, "ts": "2026-09-15"}
        for t in tickers
    ])


# --- The case the module exists for -------------------------------------------


def test_a_spinoff_that_yfinance_calls_a_split_is_raised_as_a_mismatch():
    """The section 9.2 case, with both witnesses kept.

    The arithmetic is right for the price series either way, so ``close_raw`` survives it
    and nothing downstream complains. What does not survive is the cost base.
    """
    actions = [
        action("XLF", "spinoff", "2016-09-19", "ibkr"),
        action("XLF", "split", "2016-09-19", "yfinance", ratio=1.231),
    ]

    findings = reconcile_ticker(actions, "XLF")
    mismatch = next(f for f in findings if f.reason == "kind_mismatch")

    assert mismatch.ibkr["kind"] == "spinoff"
    assert mismatch.yfinance["kind"] == "split"
    assert "Manda IBKR" in mismatch.detail
    assert "no se corrige solo" in mismatch.detail


def test_an_ex_date_a_day_apart_is_still_the_same_event():
    """Providers stamp the ex-date from different systems; a day's gap is not two events."""
    actions = [
        action("XLF", "spinoff", "2016-09-19", "ibkr"),
        action("XLF", "split", "2016-09-20", "yfinance", ratio=1.231),
    ]
    assert [f.reason for f in reconcile_ticker(actions, "XLF")][0] == "kind_mismatch"


def test_far_apart_dates_are_two_events_not_one():
    """Too wide and the "match" would pair a 2016 spin-off with a 2020 split."""
    actions = [
        action("AAPL", "split", "2016-09-19", "ibkr"),
        action("AAPL", "split", "2020-08-31", "yfinance", ratio=4.0),
    ]
    assert all(f.reason == "unconfirmed" for f in reconcile_ticker(actions, "AAPL"))
    assert len(reconcile_ticker(actions, "AAPL")) == 2


def test_the_two_sources_agreeing_produces_nothing():
    """A clean reconciliation is silence. Otherwise the flags train the reader to ignore."""
    actions = [
        action("AAPL", "split", "2020-08-31", "ibkr"),
        action("AAPL", "split", "2020-08-31", "yfinance", ratio=4.0),
    ]
    assert reconcile_ticker(actions, "AAPL") == []


def test_a_split_only_yfinance_knows_about_is_unconfirmed():
    actions = [action("TMUS", "split", "2026-03-02", "yfinance", ratio=2.0)]
    finding = reconcile_ticker(actions, "TMUS")[0]

    assert finding.reason == "unconfirmed"
    assert finding.ibkr is None
    assert "fuente autorizada" in finding.detail


def test_a_reorg_only_ibkr_knows_about_is_unconfirmed_too():
    """The other direction matters: the price series may simply not reflect it (section 9.8)."""
    actions = [action("TMUS", "merger", "2026-03-02", "ibkr")]
    finding = reconcile_ticker(actions, "TMUS")[0]

    assert finding.reason == "unconfirmed"
    assert finding.yfinance is None
    assert "valoración de esa posición queda en duda" in finding.detail


def test_a_cash_dividend_is_not_a_structural_action():
    """It does not change what a share is; it reconciles against cash_transactions."""
    actions = [action("AAPL", "dividend", "2020-08-07", "yfinance", amount=0.82)]
    assert reconcile_ticker(actions, "AAPL") == []


# --- The needs_review flag (section 9.2) --------------------------------------


def test_only_held_positions_are_flagged():
    """A discrepancy on something nobody owns is a data note, not a blocked position."""
    actions = [
        action("XLF", "spinoff", "2016-09-19", "ibkr"),
        action("XLF", "split", "2016-09-19", "yfinance", ratio=1.231),
    ]

    assert review_positions(actions, positions("TMUS", "MSFT")) == []

    flagged = review_positions(actions, positions("XLF", "MSFT"))
    assert [r.ticker for r in flagged] == ["XLF"]
    assert flagged[0].needs_review is True


def test_a_clean_portfolio_produces_an_empty_list_not_a_roll_call():
    actions = [
        action("AAPL", "split", "2020-08-31", "ibkr"),
        action("AAPL", "split", "2020-08-31", "yfinance", ratio=4.0),
    ]
    assert review_positions(actions, positions("AAPL")) == []


def test_an_unmapped_reorg_from_the_ingester_is_carried_through():
    """The IBKR ingester already flags what it cannot classify; it is not re-derived.

    Two paths deciding the same thing is two paths that can disagree, so the ingester's
    own report is carried rather than reconstructed.
    """
    unmapped = ["corporate action 123 on TMUS: unmapped type SUBSCRIBERIGHTS — needs review"]
    flagged = review_positions([], positions("TMUS"), unmapped=unmapped)

    assert flagged[0].ticker == "TMUS"
    assert flagged[0].reviews[0].reason == "unmapped"
    assert "No se infiere el tipo" in flagged[0].reviews[0].detail


def test_an_unmapped_label_for_a_ticker_not_held_is_not_attached_to_another():
    """The label is matched to a held ticker, never to whichever one happens to be first."""
    unmapped = ["corporate action 123 on ZZZZ: unmapped type X — needs review"]
    assert review_positions([], positions("TMUS"), unmapped=unmapped) == []


def test_positions_accept_a_plain_ticker_list_too():
    actions = [action("XLF", "split", "2016-09-19", "yfinance", ratio=1.231)]
    assert [r.ticker for r in review_positions(actions, ["XLF"])] == ["XLF"]


def test_no_positions_and_no_actions_are_not_errors():
    assert review_positions(None, None) == []
    assert review_positions([], positions()) == []
    assert reconcile_ticker(None, "AAPL") == []


def test_the_index_lets_a_page_look_up_one_position():
    actions = [action("XLF", "split", "2016-09-19", "yfinance", ratio=1.231)]
    index = review_index(review_positions(actions, positions("XLF")))

    assert index["XLF"].needs_review is True
    assert "MSFT" not in index


# --- Ordering and separation --------------------------------------------------


def test_a_kind_mismatch_outranks_an_unclean_ratio():
    """Both describe the XLF event; the one naming both witnesses is the useful one."""
    actions = [
        action("XLF", "spinoff", "2016-09-19", "ibkr"),
        action("XLF", "split", "2016-09-19", "yfinance", ratio=1.231),
    ]
    assert [f.reason for f in reconcile_ticker(actions, "XLF")] == [
        "kind_mismatch", "unclean_ratio",
    ]


def test_notes_about_unheld_securities_are_kept_separate():
    """XLF is a market reference. Its 2016 spin-off shows forever and blocks nothing.

    Letting a permanent, harmless finding into the position flags is how a real flag gets
    ignored.
    """
    actions = [action("XLF", "split", "2016-09-19", "yfinance", ratio=1.231)]
    notes = data_quality_notes(actions)

    assert [n.ticker for n in notes] == ["XLF"]
    assert notes[0].reason == "unclean_ratio"
    assert data_quality_notes([]) == []


def test_a_ratio_of_exactly_four_is_never_a_note():
    actions = [action("AAPL", "split", "2020-08-31", "yfinance", ratio=4.0)]
    assert data_quality_notes(actions) == []


@pytest.mark.parametrize("ratio", [2.0, 3.0, 1.5, 0.1])
def test_ordinary_ratios_stay_quiet(ratio):
    actions = [action("X", "split", "2020-01-02", "yfinance", ratio=ratio)]
    assert data_quality_notes(actions) == []


def test_a_stored_timestamp_ex_date_is_shown_as_a_plain_day():
    """The table stores a full ISO timestamp; the panel shows days, not instants.

    Measured on the real database: XLF's spin-off is stored as
    ``2016-09-19T00:00:00+00:00`` and rendered raw it reads as an instant, which these
    events are not.
    """
    actions = [action("XLF", "split", "2016-09-19T00:00:00+00:00", "yfinance", ratio=1.231)]
    note = data_quality_notes(actions)[0]

    assert note.ex_date == "2016-09-19"


# --- Only events that happened while the position was held (found 2026-09-25) -------------


def test_an_event_before_the_position_existed_is_not_a_flag():
    """A split twelve years before the purchase was a permanent red flag: IBKR cannot
    corroborate what predates its statement, and it never touched this portfolio."""
    actions = [{"ticker": "ZZA", "kind": "split", "ex_date": "2013-05-01T00:00:00+00:00",
                "ratio": 0.5, "amount": None, "source": "yfinance"}]
    assert review_positions(actions, positions("ZZA"), since={"ZZA": "2025-06-02"}) == []
    flagged = review_positions(actions, positions("ZZA"), since={"ZZA": None})
    assert [r.ticker for r in flagged] == ["ZZA"], "unknown start: every event is checked"


def test_the_start_is_only_known_when_the_lots_explain_the_whole_position():
    from transform.corporate_actions import held_since

    held = pd.DataFrame([{"ticker": "ZZA", "quantity": 1.0}, {"ticker": "ZZB", "quantity": 3.0}])
    lots = pd.DataFrame([
        {"ticker": "ZZA", "ts": "2025-06-02T15:00:00+00:00", "quantity": 1.0},
        {"ticker": "ZZB", "ts": "2025-07-01T15:00:00+00:00", "quantity": 2.0},
    ])
    assert held_since(lots, held) == {"ZZA": "2025-06-02", "ZZB": None}
