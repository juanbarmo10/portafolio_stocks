"""Hand-written catalysts (CLAUDE.md §15.5, point 9).

What is defended: a catalyst without the source of its date is rejected at load (section
9.8); a window counts down to its start, not its end; a past catalyst is not "upcoming";
the alert fires once per catalyst and date, and again when the date is moved; and nothing
invented — the config is read as written, never completed.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from alerts import rules as ar
from core import config
from transform import catalysts as cat

TODAY = dt.date(2026, 10, 1)


def entry(**kw):
    return {"ticker": "ibrx", "kind": "pdufa", "date": "2026-10-20",
            "label": "Decisión sobre la indicación nueva", "source": "8-K del 2026-08-05", **kw}


def test_a_catalyst_needs_its_source_and_a_known_kind():
    assert cat.problems([entry()]) == []
    assert any("source is required" in p for p in cat.problems([entry(source="")]))
    assert any("kind" in p for p in cat.problems([entry(kind="rumour")]))
    assert any("date" in p for p in cat.problems([entry(date="Q4 2026")]))
    assert any("from is after date" in p for p in cat.problems([entry(**{"from": "2026-11-01"})]))
    assert cat.problems(None) == [] and cat.problems("x") == ["must be a list of catalysts"]


def test_yaml_dates_and_strings_both_parse_and_the_ticker_is_normalised():
    one = cat.parse([entry(date=dt.date(2026, 10, 20))])[0]
    assert one.ticker == "IBRX" and one.start == one.end == dt.date(2026, 10, 20)
    assert not one.is_window and one.when == "2026-10-20"


def test_a_window_counts_down_to_its_start():
    window = cat.parse([entry(**{"from": "2026-10-10"}, date="2026-12-31")])[0]
    assert window.is_window and window.days_to(TODAY) == 9
    assert cat.upcoming([window], TODAY, within_days=14) == [window], "starts in 9 days"
    assert cat.countdown(window, dt.date(2026, 11, 1)) == "ventana abierta"
    assert cat.upcoming([window], dt.date(2026, 11, 1)) == [window], "still open"


def test_past_and_far_catalysts_are_not_upcoming():
    past, far = cat.parse([entry(date="2026-09-20"), entry(date="2027-03-01")])
    assert cat.upcoming([past, far], TODAY) == [far]
    assert cat.upcoming([past, far], TODAY, within_days=14) == []
    assert cat.upcoming([far], TODAY, tickers=["CCC"]) == []


def test_the_alert_fires_once_per_date_and_again_when_it_moves():
    now = pd.Timestamp("2026-10-10T12:00:00Z")   # 10 days before 2026-10-20
    first = ar.catalyst_soon(ar.Snapshot(now=now, catalysts=cat.parse([entry()])), {})
    assert len(first) == 1 and "IBRX" in first[0].text and "8-K del 2026-08-05" in first[0].text
    moved = ar.catalyst_soon(ar.Snapshot(now=now, catalysts=cat.parse(
        [entry(date="2026-10-22")])), {})
    assert moved[0].key != first[0].key, "a moved date is news again"
    far = ar.catalyst_soon(ar.Snapshot(now=now, catalysts=cat.parse(
        [entry(date="2026-12-01")])), {})
    assert far == [], "outside the 14-day window"
    wider = ar.catalyst_soon(ar.Snapshot(now=now, catalysts=cat.parse(
        [entry(date="2026-12-01")]), catalyst_window_days=90), {})
    assert len(wider) == 1


def test_loading_settings_rejects_a_catalyst_without_source(tmp_path, monkeypatch):
    local = tmp_path / "settings.local.yaml"
    local.write_text("catalysts:\n  - {ticker: IBRX, kind: pdufa, date: 2026-10-20, "
                     "label: x}\n", encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", local)
    config.load_settings.cache_clear()
    with pytest.raises(ValueError, match="source is required"):
        config.load_settings()
    config.load_settings.cache_clear()


def test_the_configured_rule_is_known():
    config.load_settings.cache_clear()
    kinds = [r["kind"] for r in config.load_settings().raw["alerts"]["rules"]]
    assert "catalyst_soon" in kinds and set(kinds) <= set(ar.RULES)
