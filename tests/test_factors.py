"""The factor study (CLAUDE.md §15.4.9), on synthetic data with known answers.

What is tested is the machinery — point-in-time, orientation, the disjoint periods, the
decision rule — never a result: the study runs once, on real data, from the runner.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from ingest import member_facts
from validation import factors as fx
from validation.metrics import sign_flip_pvalue

WINDOWS = {"quarter": [80, 100], "half": [170, 195], "three_quarters": [260, 285],
           "year": [350, 380]}
CONCEPTS = {
    "gross_profit": {"kind": "duration", "unit": "USD", "tags": ["GrossProfit"]},
    "net_income": {"kind": "duration", "unit": "USD", "tags": ["NetIncomeLoss"]},
    "operating_cash_flow": {"kind": "duration", "unit": "USD",
                            "tags": ["NetCashProvidedByUsedInOperatingActivities"]},
    "assets": {"kind": "instant", "unit": "USD", "tags": ["Assets"]},
}


def fact(val, end, filed, start=None):
    out = {"end": end, "val": val, "filed": filed, "form": "10-K", "fp": "FY"}
    if start:
        out["start"] = start
    return out


def company(years, *, gp=None, ni=None, ocf=None, assets=None, lag_days=45):
    """companyfacts subset: one fiscal year per calendar year, filed ``lag_days`` after."""
    tags = {"GrossProfit": [], "NetIncomeLoss": [], "NetCashProvidedByUsedInOperatingActivities": [],
            "Assets": []}
    for year in years:
        end = f"{year}-12-31"
        filed = (pd.Timestamp(end) + pd.Timedelta(days=lag_days)).date().isoformat()
        start = f"{year}-01-01"
        for tag, source in (("GrossProfit", gp), ("NetIncomeLoss", ni),
                            ("NetCashProvidedByUsedInOperatingActivities", ocf)):
            if source is not None:
                tags[tag].append(fact(source(year), end, filed, start))
        if assets is not None:
            tags["Assets"].append(fact(assets(year), end, filed))
            # The balance sheet a year earlier, as every 10-K carries it.
            tags["Assets"].append(fact(assets(year - 1), f"{year - 1}-12-31", filed))
    return {"us-gaap": {t: {"units": {"USD": v}} for t, v in tags.items() if v}}


# --- Point-in-time -------------------------------------------------------------------------


def test_a_10k_is_invisible_the_day_before_it_is_filed():
    facts = {"C1": company([2019], gp=lambda y: 40.0, assets=lambda y: 100.0)}
    records = fx.fact_records(facts, CONCEPTS, WINDOWS)
    filed = pd.Timestamp("2019-12-31") + pd.Timedelta(days=45)
    assert fx.gross_profitability(records, filed - pd.Timedelta(days=1), 480).empty
    assert fx.gross_profitability(records, filed, 480)["C1"] == pytest.approx(0.40)


def test_a_restatement_replaces_the_original_only_from_its_filing_date():
    records = pd.DataFrame([
        ("C1", "gross_profit", "fy", "2019-12-31", "2020-02-14", 40.0),
        ("C1", "gross_profit", "fy", "2019-12-31", "2020-08-01", 30.0),   # the 10-K/A
        ("C1", "assets", "", "2019-12-31", "2020-02-14", 100.0),
    ], columns=fx.RECORD_COLUMNS)
    records["end"] = pd.to_datetime(records["end"])
    records["filed"] = pd.to_datetime(records["filed"])
    assert fx.gross_profitability(records, "2020-07-31", 480)["C1"] == pytest.approx(0.40)
    assert fx.gross_profitability(records, "2020-08-01", 480)["C1"] == pytest.approx(0.30)


def test_a_fiscal_year_older_than_the_limit_is_no_signal():
    facts = {"C1": company([2017], gp=lambda y: 40.0, assets=lambda y: 100.0)}
    records = fx.fact_records(facts, CONCEPTS, WINDOWS)
    assert not fx.gross_profitability(records, "2019-01-15", 480).empty    # 380 days old
    assert fx.gross_profitability(records, "2019-06-01", 480).empty        # 517 days old


def test_accruals_divide_by_the_average_of_both_balance_sheets_and_are_not_negated():
    facts = {"C1": company([2019], ni=lambda y: 30.0, ocf=lambda y: 10.0,
                           assets=lambda y: {2018: 100.0, 2019: 300.0}[y])}
    records = fx.fact_records(facts, CONCEPTS, WINDOWS)
    value = fx.accruals(records, "2020-06-01", 480)["C1"]
    assert value == pytest.approx((30 - 10) / 200), "average assets, positive = profit > cash"


def test_accruals_without_last_years_balance_sheet_are_no_signal():
    records = pd.DataFrame([
        ("C1", "net_income", "fy", "2019-12-31", "2020-02-14", 30.0),
        ("C1", "operating_cash_flow", "fy", "2019-12-31", "2020-02-14", 10.0),
        ("C1", "assets", "", "2019-12-31", "2020-02-14", 300.0),
    ], columns=fx.RECORD_COLUMNS)
    records["end"] = pd.to_datetime(records["end"])
    records["filed"] = pd.to_datetime(records["filed"])
    assert fx.accruals(records, "2020-06-01", 480).empty


# --- Prices ------------------------------------------------------------------------------


def frame_of(values: dict[str, list[float]], start="2020-01-01") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=len(next(iter(values.values()))))
    return pd.DataFrame(values, index=index)


def test_momentum_skips_the_last_month():
    days = 400
    steady = [100.0 * (1.001 ** i) for i in range(days)]
    late_jump = steady[:-10] + [v * 2 for v in steady[-10:]]   # doubles in the last 10 days
    past = frame_of({"A": steady, "B": late_jump})
    score = fx.momentum(past, past.index[-1], 365, 30)
    assert score["A"] == pytest.approx(score["B"]), "the last month does not count"


def test_forward_returns_enter_the_session_after_the_date():
    prices = frame_of({"A": [100.0, 200.0] + [200.0] * 200})     # the jump is ON day 1
    returns = fx.forward_returns(prices, prices.index[1], 30)
    assert returns["A"] == pytest.approx(0.0), "a move on the signal's own day is not earned"


def test_forward_returns_are_empty_when_the_window_runs_past_the_data():
    prices = frame_of({"A": [100.0] * 50})
    assert fx.forward_returns(prices, prices.index[30], 90).empty


def test_total_returns_add_the_dividend_on_its_date_and_undo_the_split():
    closes = pd.DataFrame({
        "series_id": "A:close_raw",
        "ts": ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
        "value": [100.0, 100.0, 50.0, 50.0],          # 2:1 split on the 6th
    })
    actions = pd.DataFrame([
        {"ticker": "A", "kind": "split", "ex_date": "2020-01-06", "ratio": 2.0,
         "amount": None, "source": "yfinance"},
        {"ticker": "A", "kind": "dividend", "ex_date": "2020-01-07", "ratio": None,
         "amount": 1.0, "source": "yfinance"},
    ])
    calendar = pd.DatetimeIndex(pd.to_datetime(closes["ts"]))
    tr = fx.total_return_wide(closes, actions, "2020-12-31", calendar)
    assert tr["A"].iloc[2] == pytest.approx(tr["A"].iloc[0]), "a split is not a loss"
    assert tr["A"].iloc[3] / tr["A"].iloc[2] == pytest.approx(51 / 50), "dividend added"


# --- One period --------------------------------------------------------------------------


def test_perfectly_sorted_returns_give_a_positive_long_and_spread():
    names = [f"T{i}" for i in range(100)]
    scores = pd.Series(np.arange(100, dtype=float), index=names)
    returns = pd.Series(np.arange(100, dtype=float) / 1000, index=names)
    long, spread, n = fx.period_values(scores, returns, 5, 50)
    assert n == 100
    assert long == pytest.approx(np.arange(80, 100).mean() / 1000 - np.arange(100).mean() / 1000)
    assert spread == pytest.approx(0.08)


def test_a_period_with_too_few_names_does_not_count():
    names = [f"T{i}" for i in range(40)]
    s = pd.Series(np.arange(40, dtype=float), index=names)
    assert fx.period_values(s, s, 5, 50) is None


def test_share_classes_of_one_company_vote_once():
    values = pd.Series({"GOOG": 1.0, "GOOGL": 1.1, "MSFT": 2.0, "NOCIK": 3.0})
    kept = fx.one_per_company(values, {"GOOG": "C1", "GOOGL": "C1", "MSFT": "C2"})
    assert sorted(kept.index) == ["GOOG", "MSFT", "NOCIK"]


def test_accruals_are_oriented_and_financials_excluded():
    records = fx.fact_records({
        "C1": company([2019], ni=lambda y: 30.0, ocf=lambda y: 10.0, assets=lambda y: 100.0),
        "C2": company([2019], ni=lambda y: 10.0, ocf=lambda y: 30.0, assets=lambda y: 100.0),
        "C3": company([2019], ni=lambda y: 10.0, ocf=lambda y: 30.0, assets=lambda y: 100.0),
    }, CONCEPTS, WINDOWS)
    cfg = {"max_fiscal_age_days": 480,
           "signals": {"accruals": {"expected": "negative", "exclude_financials": True}}}
    scores = fx.signal_scores("accruals", pd.Timestamp("2020-06-01"), records=records,
                              past=pd.DataFrame(), members=["A", "B", "BANK"],
                              cik_of={"A": "C1", "B": "C2", "BANK": "C3"},
                              financial={"C1": False, "C2": False, "C3": True}, cfg=cfg)
    assert sorted(scores.index) == ["A", "B"], "the bank is out"
    assert scores["B"] > scores["A"], "less accrual = higher score"


# --- The test and the rule -------------------------------------------------------------


def test_sign_flip_finds_a_consistent_effect_and_not_a_symmetric_one():
    assert sign_flip_pvalue([0.01] * 20, n=2000) < 0.001
    assert sign_flip_pvalue([0.01, -0.01] * 10, n=2000) > 0.9
    assert sign_flip_pvalue([], n=10) is None


def result(signal, horizon, test, mean, halves, significant=True):
    return fx.Result(signal=signal, horizon=horizon, test=test, n=30, mean=mean, median=mean,
                     hit_rate=0.6, names=300, p=0.001, q=0.01, significant=significant,
                     halves=halves)


CFG_RULE = {"signals": {"momentum": {}, "accruals": {}}, "decision_test": "long",
            "decision_horizons": [90, 180]}


def test_the_rule_needs_the_long_test_at_a_decision_horizon_in_both_halves():
    results = [
        result("momentum", 30, "long", 0.02, (0.01, 0.02)),          # 30 d does not decide
        result("momentum", 90, "spread", 0.02, (0.01, 0.02)),        # spread does not decide
        result("accruals", 180, "long", 0.02, (0.01, 0.03)),
    ]
    assert fx.decision(results, CFG_RULE) == {"momentum": False, "accruals": True}


def test_the_rule_rejects_a_sign_that_flips_between_halves():
    results = [result("accruals", 90, "long", 0.01, (-0.01, 0.03))]
    assert fx.decision(results, CFG_RULE)["accruals"] is False


# --- End to end, with a planted effect --------------------------------------------------


def test_the_study_finds_a_planted_quality_effect():
    """Sixty companies whose drift grows with their gross profitability: the machinery has
    to see it, oriented the right way."""
    rng = np.random.default_rng(1)
    calendar = pd.bdate_range("2017-01-02", "2020-12-31")
    rows, facts, cik_of, intervals = [], {}, {}, []
    for i in range(60):
        ticker, cik = f"T{i:02d}", f"{i:010d}"
        quality = (i % 20) / 20                                   # GP/A 0 … 0.95
        drift = 0.0004 * quality
        path = 50 * np.exp(np.cumsum(drift + rng.normal(0, 0.004, len(calendar))))
        rows += [{"series_id": f"{ticker}:close_raw", "ts": d.date().isoformat(), "value": v}
                 for d, v in zip(calendar, path)]
        facts[cik] = company(range(2016, 2021), gp=lambda y, q=quality: 100 * q,
                             assets=lambda y: 100.0)
        cik_of[ticker] = cik
        intervals.append({"universe": "sp500", "ticker": ticker, "start_date": "2010-01-01",
                          "end_date": None})
    rows += [{"series_id": "SPY:close_raw", "ts": d.date().isoformat(), "value": 100.0}
             for d in calendar]
    cfg = {"signals": {"gross_profitability": {"expected": "positive",
                                               "exclude_financials": True}},
           "max_fiscal_age_days": 480, "first_date": "2018-01-02", "horizons_days": [90],
           "groups": 5, "min_names": 50, "permutations": 500, "seed": 0, "fdr_alpha": 0.10,
           "halves_split": "2019-06-30", "decision_horizons": [90], "decision_test": "long",
           "calendar_ticker": "SPY"}
    outcome = fx.study(facts, cik_of, {}, pd.DataFrame(rows), pd.DataFrame(),
                       pd.DataFrame(intervals), CONCEPTS, WINDOWS, cfg, "2020-12-31")
    long = next(r for r in outcome["results"] if r.test == "long")
    assert long.n >= 8 and long.mean > 0 and long.p < 0.05
    assert outcome["counts"]["with_gross_profit"] == 60


# --- The download cache ----------------------------------------------------------------


def test_the_subset_keeps_only_the_concepts_asked_for():
    document = {"facts": {"us-gaap": {"Assets": {"units": {}}, "Goodwill": {"units": {}}}}}
    assert member_facts.subset(document, ["Assets", "GrossProfit"]) == \
        {"us-gaap": {"Assets": {"units": {}}}}


def test_a_cached_company_is_read_without_the_network(tmp_path):
    stored = {"tags": ["Assets", "GrossProfit"], "facts": {"us-gaap": {"Assets": {}}}}
    (tmp_path / "0000000001.json.gz").write_bytes(gzip.compress(json.dumps(stored).encode()))
    got = member_facts.load("0000000001", ["Assets"], tmp_path, base_url="http://x",
                            user_agent="t", delay_s=0)
    assert got == {"us-gaap": {"Assets": {}}}
