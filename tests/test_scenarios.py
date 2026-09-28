"""Scenarios and size by tolerable loss (CLAUDE.md §15.5, point 3), with known answers.

What is defended: the expected return is the annualized expected **wealth**, not the average
of annual returns (section 9.10); the sizes are ceilings that follow only from what the card
says; a scenario the engine cannot price is unknown, never guessed; and a card that would
size on the wrong tail (swapped scenarios, probabilities that do not add up) is refused.
"""

from __future__ import annotations

import math

import pytest

from core.config import scenario_problems
from transform import scenarios as scn
from transform.valuation import Valuation

SPEC = {"years": 5,
        "bear": {"prob": 0.25, "multiple": 0.4},
        "base": {"prob": 0.5, "multiple": 1.5},
        "bull": {"prob": 0.25, "multiple": 4.0}}


def test_the_expected_return_is_the_annualized_expected_wealth():
    a = scn.assess(SPEC, None, loss_budget=0.02, max_position=0.20)
    expected = 0.25 * 0.4 + 0.5 * 1.5 + 0.25 * 4.0
    assert a.expected_multiple == pytest.approx(expected)
    assert a.expected_return == pytest.approx(expected ** (1 / 5) - 1)
    mean_of_rates = sum(p * (m ** 0.2 - 1) for p, m in ((0.25, 0.4), (0.5, 1.5), (0.25, 4.0)))
    assert a.expected_return > mean_of_rates, "averaging yearly rates misstates it (§9.10)"


def test_loss_probability_worst_loss_and_the_size_the_budget_allows():
    a = scn.assess(SPEC, None, loss_budget=0.02, max_position=0.20)
    assert a.loss_probability == pytest.approx(0.25)
    assert a.worst_loss == pytest.approx(0.6)
    assert a.size_by_budget == pytest.approx(0.02 / 0.6), "the bear case costs exactly 2 %"
    assert a.size_if_zero == pytest.approx(0.02)
    assert a.ceiling == pytest.approx(min(0.02 / 0.6, a.kelly_ceiling, 0.20))


def test_kelly_maximizes_the_expected_log_of_wealth():
    probs, multiples = [0.25, 0.5, 0.25], [0.4, 1.5, 4.0]
    f = scn.kelly_fraction(probs, multiples)

    def growth(x):
        return sum(p * math.log(1 + x * (m - 1)) for p, m in zip(probs, multiples))

    assert 0 < f <= 1
    assert growth(f) >= growth(max(0.0, f - 0.05)) and growth(f) >= growth(min(1.0, f + 0.05))


def test_a_scenario_that_goes_to_zero_keeps_kelly_below_everything():
    f = scn.kelly_fraction([0.5, 0.5], [0.0, 3.0])
    assert f == pytest.approx(0.25, abs=0.002), "p·b − q over b, with b = 2: (1 − 0.5)/2"


def test_without_a_loss_budget_there_is_no_budget_size_and_it_says_so():
    a = scn.assess(SPEC, None, max_position=None)
    assert a.size_by_budget is None and a.size_if_zero is None
    assert a.ceiling == pytest.approx(a.kelly_ceiling)
    assert any("loss_budget" in n for n in a.notes)


def valuation(market_cap, fcf):
    fields = dict.fromkeys(Valuation.__dataclass_fields__)
    fields.update(as_of="2026-09-28", market_cap=market_cap, fcf_ttm=fcf,
                  debt_unidentified=False)
    return Valuation(**fields)


def test_a_growth_scenario_uses_the_implied_return_engine():
    spec = {**SPEC, "base": {"prob": 0.5, "growth": 0.10}}
    a = scn.assess(spec, valuation(1000.0, 50.0))
    base = a.scenarios[1]
    assert base.annual_return is not None and base.multiple == pytest.approx(
        (1 + base.annual_return) ** 5)


def test_a_growth_scenario_on_a_cash_burner_is_unknown_and_says_how_to_write_it():
    spec = {**SPEC, "base": {"prob": 0.5, "growth": 0.10}}
    a = scn.assess(spec, valuation(1000.0, -50.0))
    assert a.scenarios[1].multiple is None and "multiple" in a.scenarios[1].note
    assert a.expected_return is None and a.ceiling is None


def test_no_scenarios_no_assessment():
    assert scn.assess(None, None) is None


# --- Validation at load -----------------------------------------------------------------------


def card(**scenarios):
    return {"ticker": "AAA", "scenarios": {**SPEC, **scenarios}}


def test_a_well_written_card_passes():
    assert scenario_problems(card()) == []
    assert scenario_problems({"ticker": "AAA"}) == []


@pytest.mark.parametrize("change, fragment", [
    ({"bear": {"prob": 0.5, "multiple": 0.4}}, "add up to"),
    ({"bear": {"prob": 0.25, "multiple": 2.0}}, "better than"),
    ({"bear": {"prob": 0.25, "multiple": 0.4, "growth": 0.1}}, "exactly one"),
    ({"base": {"prob": 0.5, "growth": 12}}, "0.12 for 12"),
    ({"bull": {"prob": 25, "multiple": 4.0}}, "fraction"),
    ({"years": 0}, "years"),
])
def test_a_card_that_would_size_on_the_wrong_tail_is_refused(change, fragment):
    problems = scenario_problems(card(**change))
    assert any(fragment in p for p in problems), problems
