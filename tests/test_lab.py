"""The rule laboratory (``lab/``), on synthetic data with known answers.

What is defended: macro is read as published, a revision never reaching back; the entry is
the session after the signal; comparing within a date cancels the market's move; an event's
baseline stays away from its events; the holdout is out of reach unless asked for; a
strategy's arithmetic; the registry counts every distinct test and corrects over all of
them; and every catalogue rule runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lab import evaluate, registry, signals
from lab.data import Lab, known_as_of

DAYS = pd.bdate_range("2018-01-01", "2021-12-31")


class FakeLab(Lab):
    """A Lab over frames given in memory: no database, no cache."""

    def __init__(self, tr: pd.DataFrame, members: list[str], fred=None, holdout="2021-06-01"):
        from core.config import load_settings
        self.settings = load_settings()
        self.settings.raw["lab"] = {"first_date": "2018-01-01", "first_date_timing":
                                    "2018-01-01", "holdout_from": holdout}
        self._tr, self._members, self._fred = tr, members, fred or {}
        self._memo = {}

    def total_return(self, tickers=None):
        return self._tr if tickers is None else self._tr.reindex(columns=tickers)

    def prices(self, tickers=None):
        return self.total_return(tickers)

    def calendar(self):
        return self._tr.index

    def members(self):
        return pd.DataFrame(True, index=self._tr.index, columns=self._members)

    def members_on(self, day):
        return list(self._members)

    def fred(self, series_id):
        return self._fred[series_id]


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "PATH", tmp_path / "registro.jsonl")


def test_macro_is_read_as_published_and_a_revision_never_reaches_back():
    rows = pd.DataFrame([
        {"ts": "2020-01-01", "ts_release": "2020-02-14", "value": 1.0},
        {"ts": "2020-02-01", "ts_release": "2020-03-13", "value": 2.0},
        {"ts": "2020-01-01", "ts_release": "2020-03-20", "value": 1.5},   # revision of Jan
        {"ts": "2020-03-01", "ts_release": "", "value": 9.0},             # unknown release
    ])
    cal = pd.bdate_range("2020-02-10", "2020-03-31")
    s = known_as_of(rows, cal)
    assert pd.isna(s["2020-02-13"]), "not yet published"
    assert s["2020-02-14"] == 1.0 and s["2020-03-12"] == 1.0
    assert s["2020-03-13"] == 2.0, "the newest reference date, as soon as it is out"
    assert s["2020-03-31"] == 2.0, "January's revision does not replace February's print"
    assert 9.0 not in s.to_numpy(), "a row with no release date cannot be placed in time"


def test_timing_enters_the_session_after_and_compares_on_against_off():
    # Flat asset, except a 10 % dip in the weeks after every third grid date ("on"), starting
    # after the entry session: "on" must come out worse than "off".
    spy = pd.Series(100.0, index=DAYS)
    on = pd.Series(0.0, index=DAYS)
    grid = evaluate.grid_dates(DAYS[DAYS < "2021-06-01"], 30)
    flagged = set(grid[::3])
    for g in flagged:
        entry = DAYS[DAYS.get_loc(g) + 1]
        spy[(DAYS > entry) & (DAYS <= g + pd.Timedelta(days=45))] = 90.0
        on[g] = 1.0
    lab = FakeLab(pd.DataFrame({"SPY": spy}), [])
    results = evaluate.timing(lab, on, rule="t", horizons=(30,), expected="lower")
    assert results[0].mean < 0 and results[0].n == len(flagged)


def test_cross_section_cancels_the_market_and_sees_the_flagged_underperform():
    rng = np.random.default_rng(0)
    market = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, len(DAYS))), index=DAYS)
    tr = pd.DataFrame({"SPY": market, **{f"S{i}": market * (1 + 0.0 * i) for i in range(10)}})
    tr["BAD1"] = market * np.linspace(1, 0.5, len(DAYS))     # drifts down relative to all
    tr["BAD2"] = market * np.linspace(1, 0.5, len(DAYS))
    names = [c for c in tr.columns if c != "SPY"]
    lab = FakeLab(tr, names)

    def flag(day, members):
        return pd.Series({m: m.startswith("BAD") for m in members})

    results, per_date = evaluate.cross_section(lab, flag, rule="c", horizons=(90,),
                                               min_names=2, expected="lower")
    assert results[0].mean < 0 and (per_date["value"] < 0).all()
    assert per_date["date"].max() < pd.Timestamp("2021-06-01"), "holdout out of reach"


def test_events_against_an_unconditional_baseline():
    tr = pd.DataFrame({"SPY": pd.Series(100.0, index=DAYS),
                       "A": pd.Series(100.0, index=DAYS)})
    event = pd.Timestamp("2019-06-03")
    tr.loc[tr.index > DAYS[DAYS.get_loc(event) + 1], "A"] = 120.0   # jumps after entry
    lab = FakeLab(tr, ["A"])
    ev = pd.DataFrame({"ticker": ["A"], "date": [event]})
    results, frame = evaluate.events(lab, ev, rule="e", horizons=(30,))
    base = frame[frame["group"] == "base"]
    assert results[0].mean == pytest.approx(0.2 - base["value"].mean())
    away, frame2 = evaluate.events(lab, ev, rule="e", horizons=(30,), away_days=30)
    base2 = frame2[frame2["group"] == "base"]
    assert ((base2["date"] - event).abs() > pd.Timedelta(days=30)).all()


def test_excluding_near_event_dates_from_the_baseline_biases_it():
    """A stock that jumps +10 % on every event day and is flat otherwise: after the entry
    nothing happens. Random days hold the jumps, so the fair difference is negative. Excluding
    the near-event dates removes exactly the windows with a jump, and the same zero looks
    like a positive "drift" — the bias found in the lab on 2026-09-29."""
    price = pd.Series(100.0, index=DAYS)
    events_ = pd.bdate_range("2018-03-01", "2020-12-01", freq="95D")
    for d in events_:
        price[price.index >= d] *= 1.10
    tr = pd.DataFrame({"SPY": pd.Series(100.0, index=DAYS), "A": price})
    lab = FakeLab(tr, ["A"])
    ev = pd.DataFrame({"ticker": "A", "date": events_ + pd.Timedelta(days=1)})
    fair, _ = evaluate.events(lab, ev, rule="e", horizons=(60,), placebo_days=None)
    biased, _ = evaluate.events(lab, ev, rule="e", horizons=(60,), placebo_days=None,
                                away_days=60)
    assert fair[0].mean < 0, "after the entry nothing happens; random days hold the jumps"
    assert biased[0].mean > fair[0].mean + 0.02, "excluding pre-event windows inflates it"


def test_the_vectorized_excess_matches_the_scalar_one():
    rng = np.random.default_rng(1)
    tr = pd.DataFrame(np.cumprod(1 + rng.normal(0, 0.01, (len(DAYS), 3)), axis=0),
                      index=DAYS, columns=["SPY", "A", "B"])
    tr.loc[DAYS[100:110], "A"] = np.nan
    future = tr.bfill(limit=evaluate.FILL)
    days = [DAYS[50], DAYS[95], DAYS[400]]
    v = evaluate.excess_at(future, tr["SPY"], ["A", "A", "B"], days, 60)
    s = [evaluate.forward_excess(tr[[t]], tr["SPY"], d, 60, future=future[[t]]).iloc[0]
         for t, d in zip(["A", "A", "B"], days)]
    assert np.allclose(v, s)


def test_strategy_arithmetic_and_costs():
    tr = pd.DataFrame({"SPY": pd.Series(np.linspace(100, 110, len(DAYS)), index=DAYS),
                       "A": pd.Series(np.linspace(100, 150, len(DAYS)), index=DAYS)})
    lab = FakeLab(tr, ["A"])
    bt = evaluate.strategy(lab, lambda d, m: ["A"], rebalance_days=91, cost_bps=0)
    a = tr["A"].loc[bt.curve.index]
    assert bt.curve["strategy"].iloc[-1] == pytest.approx(a.iloc[-1] / a.iloc[0], rel=1e-3)
    costly = evaluate.strategy(lab, lambda d, m: ["A"], rebalance_days=91, cost_bps=100)
    assert costly.curve["strategy"].iloc[-1] < bt.curve["strategy"].iloc[-1]
    assert bt.turnover < 0.1, "held the same stock: one entry, no turnover afterwards"


def test_the_registry_counts_every_distinct_test_and_corrects_over_all():
    r = evaluate.Result("x", "timing", 30, 20, -0.01, 0.6, 0.01, (0.0, 0.0), ("a", "b"),
                        {"k": 1})
    registry.record([r])
    registry.record([r])                                    # same test again
    others = [evaluate.Result(f"y{i}", "timing", 30, 20, 0.0, 0.5, 0.9, (0, 0), ("a", "b"))
              for i in range(19)]
    registry.record(others)
    registry.record([r], holdout=True)
    frame = registry.summary()
    ins = frame[~frame["holdout"].astype(bool)]
    assert len(ins) == 20, "the repeated test counts once"
    assert int(ins.loc[ins["rule"] == "x", "runs"].iloc[0]) == 2
    assert float(ins.loc[ins["rule"] == "x", "q"].iloc[0]) == pytest.approx(0.2), \
        "p 0.01 among 20 tests is q 0.20: not significant once everything is counted"
    assert frame["holdout"].astype(bool).sum() == 1, "the holdout is its own family"


def test_every_catalogue_rule_declares_what_it_needs():
    for name, spec in signals.RULES.items():
        assert spec.kind in {"timing", "cross", "events"}, name
        assert spec.expected in {"lower", "higher"}, name
        assert spec.description, name
    cat = signals.catalogue()
    assert set(cat["regla"]) == set(signals.RULES)
