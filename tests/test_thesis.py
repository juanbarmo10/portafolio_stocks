"""Thesis invalidation board (CLAUDE.md sections 5.2, 8 phase 2 point 7).

What is defended here is the line between what a program may decide and what it may not.
The written ``invalidation`` criterion is prose and stays prose: the board shows it and
shuts up. The mechanical checks — review date passed, earnings inside the blackout window,
amended financial statements, missing fundamentals — it does decide, and says why.

The optional structured rule is the bridge between the two, and its most important
behaviour is the one tested hardest: a rule that cannot be evaluated reports **unknown**,
never "fine".
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from core.config import load_settings
from ingest.sec_xbrl import extract_metric
from transform import thesis

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "sec_companyfacts_msft.json"
CIK = "0000789019"
TODAY = "2026-09-17"


@pytest.fixture(scope="module")
def observations() -> pd.DataFrame:
    cfg = load_settings().source("sec")
    facts = json.loads(FIXTURE.read_text(encoding="utf-8"))["facts"]
    records = []
    for metric, spec in cfg["concepts"].items():
        rows, _ = extract_metric(facts, metric, spec, CIK, cfg["duration_windows"])
        records.extend(rows)
    return pd.DataFrame(records)


def card(**overrides) -> dict:
    base = {
        "ticker": "MSFT", "cik": CIK,
        "thesis": "Cloud con costes de cambio altos.",
        "value_accrual": "Margen creciente con recompras que superan al SBC.",
        "key_metric": "cash_conversion",
        "invalidation": "Conversión a caja por debajo de 0,6 dos años seguidos.",
        "review_date": "2027-06-30",
    }
    base.update(overrides)
    return base


# --- What the program may not decide ------------------------------------------


def test_the_written_criterion_is_shown_verbatim_and_not_interpreted(observations):
    """Prose stays prose. Nothing paraphrases it, scores it or decides it was met."""
    row = thesis.status(card(), observations, None, None, TODAY)

    assert row.invalidation == "Conversión a caja por debajo de 0,6 dos años seguidos."
    assert row.rule_breached is None
    assert "lo juzgas tú" in row.rule_detail


def test_a_card_without_a_structured_rule_is_not_a_problem(observations):
    """The rule is optional; the prose is not (section 5.2)."""
    row = thesis.status(card(), observations, None, None, TODAY)
    assert row.rule is None
    assert not any("CRUZADO" in flag for flag in row.flags)


# --- What it does decide ------------------------------------------------------


def test_an_overdue_review_is_flagged_with_its_age(observations):
    row = thesis.status(card(review_date="2026-01-31"), observations, None, None, TODAY)

    assert row.review_overdue is True
    assert any("Revisión vencida" in flag and "229" in flag for flag in row.flags)


def test_a_future_review_is_not_flagged(observations):
    row = thesis.status(card(review_date="2027-06-30"), observations, None, None, TODAY)
    assert row.review_overdue is False


def test_earnings_inside_the_blackout_window_are_flagged(observations):
    """Section 2 makes this a hard rule, so the board states it in those words."""
    events = [{"cik": CIK, "category": "earnings", "ts": "2026-09-21", "is_estimated": 1}]
    row = thesis.status(card(), observations, None, events, TODAY)

    assert row.days_to_earnings == 4
    assert row.earnings_window is True
    assert row.earnings_is_estimated is True
    flag = next(f for f in row.flags if "Resultados" in f)
    assert "estimada" in flag, "an estimated date must never read as confirmed"
    assert "no se abre posición" in flag


def test_earnings_outside_the_window_are_reported_but_not_flagged(observations):
    events = [{"cik": CIK, "category": "earnings", "ts": "2026-10-28", "is_estimated": 1}]
    row = thesis.status(card(), observations, None, events, TODAY)

    assert row.earnings_date == "2026-10-28"
    assert row.earnings_window is False
    assert not any("Resultados" in flag for flag in row.flags)


def test_a_past_earnings_date_is_not_offered_as_the_next_one(observations):
    events = [
        {"cik": CIK, "category": "earnings", "ts": "2026-07-29", "is_estimated": 0},
        {"cik": CIK, "category": "earnings", "ts": "2026-10-28", "is_estimated": 1},
    ]
    row = thesis.status(card(), observations, None, events, TODAY)
    assert row.earnings_date == "2026-10-28"


def test_an_amended_financial_filing_is_a_governance_flag(observations):
    """The real case: TMUS filed a 10-Q/A in 2020, and it is a position (section 9.6)."""
    filings = [
        {"cik": CIK, "form": "10-Q/A", "filed_date": "2020-08-10", "is_amended": 1},
        {"cik": CIK, "form": "4/A", "filed_date": "2026-01-02", "is_amended": 1},
    ]
    row = thesis.status(card(), observations, filings, None, TODAY)

    assert [f["form"] for f in row.amended_filings] == ["10-Q/A"], "a 4/A is not a restatement"
    assert any("Posible reexpresión" in flag for flag in row.flags)


def test_missing_fundamentals_are_flagged_rather_than_assumed_fine(observations):
    row = thesis.status(card(ticker="NEW", cik="0000000001"), observations, None, None, TODAY)
    assert row.metrics["revenue_ttm"] is None
    assert any("Sin fundamentales calculables" in flag for flag in row.flags)


# --- The optional structured rule ---------------------------------------------


def test_a_structured_rule_is_evaluated_against_point_in_time_figures(observations):
    """Microsoft's cash conversion is ~0.50, so a "below 0.6" rule is breached."""
    row = thesis.status(
        card(invalidation_rule={"metric": "cash_conversion", "operator": "<",
                                "threshold": 0.6}),
        observations, None, None, TODAY,
    )

    assert row.rule_breached is True
    assert "CRUZADO" in row.rule_detail
    assert any("CRUZADO" in flag for flag in row.flags)


def test_a_rule_that_is_not_breached_says_so_plainly(observations):
    row = thesis.status(
        card(invalidation_rule={"metric": "net_margin", "operator": "<", "threshold": 0.10}),
        observations, None, None, TODAY,
    )
    assert row.rule_breached is False
    assert "no cruzado" in row.rule_detail


def test_a_rule_naming_an_unknown_metric_reports_unknown_not_fine(observations):
    """The most important behaviour in the module: silence would look like health."""
    row = thesis.status(
        card(invalidation_rule={"metric": "ebitda_adjusted", "operator": "<",
                                "threshold": 1.0}),
        observations, None, None, TODAY,
    )

    assert row.rule_breached is None
    assert "no calcula" in row.rule_detail
    assert "cash_conversion" in row.rule_detail, "the available metrics must be listed"


def test_a_rule_whose_metric_has_no_value_yet_is_unknown(observations):
    row = thesis.status(
        card(cik="0000000002",
             invalidation_rule={"metric": "roic", "operator": "<", "threshold": 0.1}),
        observations, None, None, TODAY,
    )
    assert row.rule_breached is None
    assert "no tiene valor publicado" in row.rule_detail


def test_an_unknown_operator_is_refused(observations):
    row = thesis.status(
        card(invalidation_rule={"metric": "roic", "operator": "~=", "threshold": 0.1}),
        observations, None, None, TODAY,
    )
    assert row.rule_breached is None
    assert "no reconocido" in row.rule_detail


def test_a_rule_without_a_threshold_is_refused(observations):
    row = thesis.status(
        card(invalidation_rule={"metric": "roic", "operator": "<"}),
        observations, None, None, TODAY,
    )
    assert row.rule_breached is None
    assert "umbral" in row.rule_detail


# --- The board ----------------------------------------------------------------


def test_the_board_puts_what_needs_attention_first(observations):
    cards = [
        card(ticker="QUIET", cik=CIK, review_date="2027-06-30"),
        card(ticker="NOISY", cik=CIK, review_date="2026-01-31",
             invalidation_rule={"metric": "cash_conversion", "operator": "<",
                                "threshold": 0.6}),
    ]
    rows = thesis.board(cards, observations, None, None, TODAY)

    assert [row.ticker for row in rows] == ["NOISY", "QUIET"]
    assert len(rows[0].flags) > len(rows[1].flags)


def test_an_empty_universe_is_an_empty_board_not_an_error(observations):
    assert thesis.board([], observations, None, None, TODAY) == []
    assert thesis.board(None, observations, None, None, TODAY) == []


def test_the_board_never_scores_or_ranks_a_thesis(observations):
    """Ordering is by attention needed, not by quality. A verdict is the user's."""
    rows = thesis.board([card()], observations, None, None, TODAY)
    assert not hasattr(rows[0], "score")
    assert not hasattr(rows[0], "rating")
