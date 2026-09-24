"""S&P 500 membership over time, and the member prices breadth needs (section 9.5).

Against four real consecutive snapshots of the index around 2022-06-09, the day the
``FB`` ticker became ``META`` — which is exactly what a rename looks like in a membership
list: one exit and one entry on the same date. Licence for the excerpt in
``fixtures/sp500_membership_LICENSE.txt``.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from core.config import Settings, load_settings
from ingest.universe import (
    MembershipFormatError,
    incomplete_sessions,
    UniverseIngester,
    members_since,
    membership_intervals,
    parse_snapshots,
    yf_symbol,
)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "sp500_membership_snapshots.csv"


@pytest.fixture(scope="module")
def snapshots():
    return parse_snapshots(FIXTURE.read_text(encoding="utf-8"))


# --- Parsing ---------------------------------------------------------------------


def test_the_real_snapshots_parse_into_full_memberships(snapshots):
    assert [day for day, _ in snapshots] == [
        "2022-05-10", "2022-06-08", "2022-06-09", "2022-06-21",
    ]
    assert all(500 <= len(members) <= 505 for _, members in snapshots)


def test_a_structural_change_fails_loudly_instead_of_shrinking_the_universe():
    with pytest.raises(MembershipFormatError, match="columns"):
        parse_snapshots("fecha,simbolos\n2022-01-01,A\n")


def test_a_truncated_row_is_refused_rather_than_treated_as_the_index():
    """A 30-name "S&P 500" would quietly describe a different universe."""
    with pytest.raises(MembershipFormatError, match="truncated"):
        parse_snapshots("date,tickers\n2022-01-03," + ",".join(f"T{i}" for i in range(30)))


# --- Intervals -------------------------------------------------------------------


def test_a_rename_is_an_exit_and_an_entry_on_the_same_day(snapshots):
    """FB -> META as the list records it. The two cannot be told apart from a replacement
    by the list alone, which is why rename recovery is not automatic (RESEARCH.md 2.25)."""
    rows = {r["ticker"]: r for r in membership_intervals(snapshots, first_seen="2026-09-23")}

    assert rows["FB"]["end_date"] == "2022-06-09"
    assert rows["META"]["start_date"] == "2022-06-09"
    assert rows["META"]["end_date"] is None


def test_the_end_date_is_exclusive():
    """The ticker is not a member ON its end date — off-by-one here moves every reading."""
    snaps = [("2020-01-02", frozenset({"A", "B"})), ("2020-02-03", frozenset({"A"}))]
    rows = {r["ticker"]: r for r in membership_intervals(snaps, first_seen="x")}
    assert rows["B"] == {**rows["B"], "start_date": "2020-01-02", "end_date": "2020-02-03"}


def test_a_ticker_that_leaves_and_returns_gets_two_intervals():
    """Collapsing to first/last dates would count it as a member while it was out."""
    snaps = [
        ("2020-01-02", frozenset({"A", "B"})),
        ("2020-06-01", frozenset({"A"})),
        ("2021-03-01", frozenset({"A", "B"})),
    ]
    b = [r for r in membership_intervals(snaps, first_seen="x") if r["ticker"] == "B"]

    assert [(r["start_date"], r["end_date"]) for r in b] == [
        ("2020-01-02", "2020-06-01"), ("2021-03-01", None),
    ]


def test_every_interval_carries_its_provenance(snapshots):
    rows = membership_intervals(snapshots, first_seen="2026-09-23")
    assert {r["source"] for r in rows} == {"fja05680_sp500"}
    assert {r["first_seen"] for r in rows} == {"2026-09-23"}


def test_members_since_includes_whoever_left_after_the_date(snapshots):
    rows = membership_intervals(snapshots, first_seen="x")
    tickers = members_since(rows, "2022-06-01")
    assert "FB" in tickers and "META" in tickers


def test_class_shares_are_spelled_the_way_yahoo_spells_them():
    assert yf_symbol("BRK.B") == "BRK-B"
    assert yf_symbol("AAPL") == "AAPL"


# --- The ingester ------------------------------------------------------------------


def settings_for(tmp_path) -> Settings:
    base = load_settings()
    raw = json.loads(json.dumps(base.raw, default=str))
    raw["sources"]["sp500_membership"]["cache_dir"] = str(tmp_path / "cache")
    raw["sources"]["sp500_membership"]["price_history_start"] = "2022-06-01"
    raw["sources"]["sp500_membership"]["batch_pause_s"] = 0
    return Settings(raw=raw, db_path=tmp_path / "t.db", log_level="INFO",
                    secrets={}, public_mode=False)


def fake_batch(priced: set[str]):
    """A stand-in for yfinance.download(group_by='ticker'): prices for some, none for others."""
    days = pd.date_range("2022-06-01", periods=3, freq="B")

    def download(self, tickers):
        frames = {}
        for ticker in tickers:
            symbol = yf_symbol(ticker)
            if ticker in priced:
                frames[symbol] = pd.DataFrame(
                    {"Open": 10.0, "High": 11.0, "Low": 9.0, "Close": 10.5,
                     "Adj Close": 10.5, "Volume": 1000, "Dividends": 0.0,
                     "Stock Splits": 0.0}, index=days,
                )
            else:
                frames[symbol] = pd.DataFrame(
                    {"Close": [float("nan")] * 3}, index=days
                )
        return pd.concat(frames, axis=1)

    return download


def test_unpriceable_members_are_the_coverage_gap_not_failures(tmp_path, monkeypatch):
    """Members yfinance cannot price are expected and counted — they are the survivorship
    gap the coverage figure reports. Making them failures would turn every run red and
    train the operator to ignore the exit code."""
    ingester = UniverseIngester(settings_for(tmp_path))
    cache = tmp_path / "cache" / "sp500_historical_components.csv"
    cache.parent.mkdir(parents=True)
    cache.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(UniverseIngester, "_download_batch", fake_batch({"AAPL", "MSFT"}))

    frame = ingester.fetch()

    assert set(frame["series_id"].str.split(":").str[0]) == {"AAPL", "MSFT"}
    assert (frame["series_id"].str.endswith(":close_raw")).all(), "breadth needs closes only"
    assert "FB" in ingester.unpriced
    assert ingester.partial_failures() == []

    tables = ingester.fetch_tables()
    assert tables["universe_membership"]


def test_a_failed_batch_is_a_real_failure(tmp_path, monkeypatch):
    ingester = UniverseIngester(settings_for(tmp_path))
    cache = tmp_path / "cache" / "sp500_historical_components.csv"
    cache.parent.mkdir(parents=True)
    cache.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    def boom(self, tickers):
        raise ConnectionError("network down")

    monkeypatch.setattr(UniverseIngester, "_download_batch", boom)
    ingester._retry_kwargs = {"attempts": 1}
    ingester.fetch()

    assert ingester.partial_failures(), "a network failure must reach the exit code"


def test_membership_upserts_idempotently_and_keeps_first_seen(tmp_path):
    """``end_date`` updates in place; ``first_seen`` never moves once written."""
    from db import loader

    conn = loader.init_db(tmp_path / "m.db")
    try:
        row = {"universe": "sp500", "ticker": "FB", "start_date": "2013-12-23",
               "end_date": None, "source": "fja05680_sp500", "first_seen": "2026-09-23"}
        loader.upsert_universe_membership(conn, [row])
        loader.upsert_universe_membership(
            conn, [{**row, "end_date": "2022-06-09", "first_seen": "2027-01-01"}]
        )
        stored = conn.execute(
            "SELECT end_date, first_seen FROM universe_membership WHERE ticker = 'FB'"
        ).fetchall()
        assert stored == [("2022-06-09", "2026-09-23")]
    finally:
        conn.close()


# --- Sessions that come back empty ------------------------------------------------


def test_a_session_empty_for_most_tickers_is_detected():
    """The real case: 2026-09-22 came back with 73 closes where its neighbours had 614.

    The download succeeded and the frame had rows — the hole made no noise. Stored as-is
    it cut that day's breadth coverage to 12 % and split the usable series in two.
    """
    days = pd.bdate_range("2026-08-03", periods=40)
    closes = pd.DataFrame(10.0, index=days, columns=[f"T{i}" for i in range(50)])
    closes.iloc[30, 5:] = float("nan")          # one session, most tickers empty

    assert incomplete_sessions(closes) == [days[30].date().isoformat()]


def test_a_slow_rise_in_listings_is_not_a_hole():
    """Tickers listing over the years change the count gradually; that is normal."""
    days = pd.bdate_range("2018-01-01", periods=300)
    closes = pd.DataFrame(10.0, index=days, columns=[f"T{i}" for i in range(60)])
    for i in range(60):
        closes.iloc[: i * 4, i] = float("nan")  # each ticker lists a little later

    assert incomplete_sessions(closes) == []


def holed_download(hole_at: int, days):
    """Every ticker missing the same session, on every request — a persistent source hole."""
    def download(self, tickers):
        frames = {}
        for ticker in tickers:
            close = pd.Series(10.0, index=days)
            close.iloc[hole_at] = float("nan")
            frames[yf_symbol(ticker)] = pd.DataFrame(
                {"Close": close, "Dividends": 0.0, "Stock Splits": 0.0,
                 "Open": close, "High": close, "Low": close, "Volume": 1}
            )
        return pd.concat(frames, axis=1)
    return download


def run_with_hole(tmp_path, monkeypatch, hole_at: int):
    ingester = UniverseIngester(settings_for(tmp_path))
    cache = tmp_path / "cache" / "sp500_historical_components.csv"
    cache.parent.mkdir(parents=True)
    cache.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    days = pd.bdate_range("2022-06-01", periods=30)
    monkeypatch.setattr(UniverseIngester, "_download_batch", holed_download(hole_at, days))
    monkeypatch.setattr("ingest.universe.time.sleep", lambda s: None)
    return ingester, ingester.fetch(), days


def test_a_recent_hole_reaches_the_exit_code(tmp_path, monkeypatch):
    """Fresh, it may still be transient — tomorrow's full-window re-fetch may fill it."""
    ingester, _frame, _days = run_with_hole(tmp_path, monkeypatch, hole_at=27)
    assert any("recent incomplete session" in f for f in ingester.partial_failures())


def test_an_old_hole_is_logged_but_does_not_fail_every_run(tmp_path, monkeypatch):
    """Still there after days, it is a known defect of the source. Failing every run over
    it would be an alarm that cries wolf; the breadth transform flags it on screen."""
    ingester, _frame, _days = run_with_hole(tmp_path, monkeypatch, hole_at=15)
    assert ingester.partial_failures() == []


def test_a_hole_is_never_filled(tmp_path, monkeypatch):
    """A forward-filled close is an estimate presented as data (section 12)."""
    _ingester, frame, days = run_with_hole(tmp_path, monkeypatch, hole_at=15)
    assert days[15].date().isoformat() not in set(frame["ts"].str[:10])
