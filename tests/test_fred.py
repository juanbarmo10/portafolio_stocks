"""FRED parser tests against a frozen fixture (CLAUDE.md sections 4.2, 9.4, 10).

No network and no API key: the parser is pure, so the point-in-time guarantee is
verifiable in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from db.loader import OBSERVATION_COLUMNS
from ingest.fred import FredIngester, derive_pre_vintage, parse_initial_releases

FIXTURE = Path(__file__).parent / "fixtures" / "fred_cpiaucsl_output4.json"


@pytest.fixture()
def observations() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["observations"]


def test_output_honors_the_loader_contract(observations):
    df = parse_initial_releases(observations, "CPIAUCSL")
    assert list(df.columns) == OBSERVATION_COLUMNS
    assert (df["source"] == "fred").all()
    assert (df["series_id"] == "CPIAUCSL").all()


def test_release_date_is_after_reference_date(observations):
    """The whole point of output_type=4: January CPI is published in February.

    If this ever inverts, every downstream backtest is using data before it existed.
    """
    df = parse_initial_releases(observations, "CPIAUCSL")
    assert not df.empty
    assert (df["ts_release"] > df["ts"]).all()


def test_publication_lag_is_weeks_not_days(observations):
    """The real lag is what makes the whole point-in-time apparatus necessary.

    Measured on this captured response: CPI lands 37-44 days after the month it
    describes. Assigning it to its reference month would use, for six weeks, a figure
    nobody could see. A near-zero lag here would mean the release dates were lost
    somewhere in the pipeline.
    """
    df = parse_initial_releases(observations, "CPIAUCSL")
    lag = (
        pd.to_datetime(df["ts_release"], utc=True) - pd.to_datetime(df["ts"], utc=True)
    ).dt.days
    assert lag.min() >= 30
    assert lag.max() <= 60


def test_reference_date_is_not_used_as_release_date(observations):
    """Regression guard: stamping ts_release = ts would silently reintroduce look-ahead."""
    df = parse_initial_releases(observations, "CPIAUCSL")
    row = df[df["ts"].str.startswith("2021-01-01")].iloc[0]
    assert row["ts_release"].startswith("2021-02-10")


# The two cases below are synthetic on purpose: a captured output_type=4 window contains
# neither a repeated reference date (those only arise when bisected windows are
# concatenated) nor, for CPI, a missing value. Both are real FRED behaviours the parser
# must handle, so they are constructed inline rather than faked inside the frozen fixture.

def test_earliest_release_wins_for_a_repeated_reference_date():
    """A revision must not overwrite the initial print.

    Only the value published in May was knowable in May, so that is what is stored.
    """
    synthetic = [
        {"date": "2021-04-01", "realtime_start": "2021-05-12", "value": "267.054"},
        {"date": "2021-04-01", "realtime_start": "2021-07-13", "value": "267.999"},
    ]
    df = parse_initial_releases(synthetic, "CPIAUCSL")
    assert len(df) == 1
    assert df.iloc[0]["value"] == pytest.approx(267.054)
    assert df.iloc[0]["ts_release"].startswith("2021-05-12")


def test_missing_sentinel_is_dropped_not_zeroed():
    """FRED's '.' means "not reported". Storing it as 0.0 would invent a data point."""
    synthetic = [
        {"date": "2021-04-01", "realtime_start": "2021-05-12", "value": "267.054"},
        {"date": "2021-05-01", "realtime_start": "2021-06-10", "value": "."},
    ]
    df = parse_initial_releases(synthetic, "CPIAUCSL")
    assert len(df) == 1
    assert not df["ts"].str.startswith("2021-05-01").any()


def test_timestamps_are_iso8601_utc(observations):
    """Section 10: every timestamp is ISO8601 UTC."""
    df = parse_initial_releases(observations, "CPIAUCSL")
    for column in ("ts", "ts_release"):
        assert df[column].str.endswith("+00:00").all()
        assert pd.to_datetime(df[column], utc=True).notna().all()


def test_no_observations_returns_the_empty_contract():
    df = parse_initial_releases([], "CPIAUCSL")
    assert df.empty
    assert list(df.columns) == OBSERVATION_COLUMNS


def test_all_values_missing_returns_the_empty_contract():
    df = parse_initial_releases(
        [{"date": "2021-01-01", "realtime_start": "2021-02-10", "value": "."}], "CPIAUCSL"
    )
    assert df.empty
    assert list(df.columns) == OBSERVATION_COLUMNS


def test_availability_requires_a_key():
    """Without FRED_API_KEY the runner skips rather than crashing the pipeline.

    The isolate_config fixture guarantees no key is present, on any machine.
    """
    from core import config

    settings = config.load_settings()
    assert FredIngester.is_available(settings) is False
    with pytest.raises(RuntimeError, match="FRED_API_KEY not set"):
        FredIngester(settings)


def test_availability_with_a_key(monkeypatch):
    from core import config

    monkeypatch.setenv("FRED_API_KEY", "test-key")
    config.load_settings.cache_clear()
    settings = config.load_settings()
    assert FredIngester.is_available(settings) is True
    # Every configured series is a plain FRED code, which is also its series_id.
    ingester = FredIngester(settings)
    assert "BAMLH0A0HYM2" in ingester._series


class _Resp:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = b"x"

    def json(self) -> dict:
        return self._payload


def _ingester(monkeypatch) -> FredIngester:
    from core import config

    monkeypatch.setenv("FRED_API_KEY", "test-key")
    config.load_settings.cache_clear()
    return FredIngester(config.load_settings())


def test_vintage_limit_bisects_the_window(monkeypatch):
    """Daily series exceed FRED's 2000-vintage cap; the window is split, not abandoned."""
    ingester = _ingester(monkeypatch)
    calls: list[tuple[str, str]] = []

    def fake_get(code, start, end):
        calls.append((start, end))
        span_days = (
            __import__("datetime").date.fromisoformat(end)
            - __import__("datetime").date.fromisoformat(start)
        ).days
        if span_days > 2:
            return _Resp(400, {"error_message": "exceeds the maximum of 2000 vintage dates"})
        return _Resp(200, {"observations": [
            {"date": start, "realtime_start": start, "value": "1.0"}
        ]})

    monkeypatch.setattr(ingester, "_get", fake_get)
    observations = ingester.fetch_observations("DFF", "2021-01-01", "2021-01-09")
    assert len(observations) > 1, "bisection produced no additional observations"
    assert len(calls) > 1
    # Every returned date falls inside the requested window: nothing was lost or invented.
    assert all("2021-01-01" <= o["date"] <= "2021-01-09" for o in observations)


def test_clock_skew_clamps_to_freds_today(monkeypatch):
    """A UTC-derived 'today' can run a day ahead of FRED's US-Central calendar."""
    ingester = _ingester(monkeypatch)
    seen: list[str] = []

    def fake_get(code, start, end):
        seen.append(end)
        if end > "2026-08-07":
            return _Resp(400, {
                "error_message": "realtime_end is after today's date (2026-08-07)."
            })
        return _Resp(200, {"observations": []})

    monkeypatch.setattr(ingester, "_get", fake_get)
    assert ingester.fetch_observations("DFF", "2026-08-01", "2026-08-08") == []
    assert seen[-1] == "2026-08-07"


def test_unrecoverable_http_error_fails_loudly(monkeypatch):
    """A changed or broken upstream raises; it never returns an empty series."""
    ingester = _ingester(monkeypatch)
    monkeypatch.setattr(
        ingester, "_get", lambda *a: _Resp(500, {"error_message": "internal error"})
    )
    with pytest.raises(RuntimeError, match="HTTP 500"):
        ingester.fetch_observations("DFF", "2021-01-01", "2021-01-02")


# --- Real-time window clamping (the T10Y2Y failure of 2026-08-28) ------------------
#
# A series' observation history and its vintage history start on different dates.
# Requesting initial releases from before ALFRED archives the series returns a 400
# whose message names the *series*, so it reads like "this series does not exist".
# Verified against the live API: T10Y2Y has observations from 1976 and vintages from
# 2014-01-27; asking from 2000-01-01 killed the whole FRED ingester.


def test_window_is_clamped_to_the_first_archived_vintage(monkeypatch):
    """fetch() asks from the first vintage, not from the configured observation_start."""
    ingester = _ingester(monkeypatch)
    monkeypatch.setattr(ingester, "_series", ["T10Y2Y"])
    monkeypatch.setattr(ingester, "_observation_start", "2000-01-01")
    monkeypatch.setattr(ingester, "_first_vintage", lambda code: "2014-01-27")

    asked: list[str] = []

    def fake_get(code, start, end):
        asked.append(start)
        return _Resp(200, {"observations": [
            {"date": "2014-01-27", "realtime_start": "2014-01-28", "value": "2.4"}
        ]})

    monkeypatch.setattr(ingester, "_get", fake_get)
    df = ingester.fetch()
    assert asked == ["2014-01-27"], "the request must start at the first vintage"
    assert len(df) == 1


def test_observation_start_wins_when_it_is_later_than_the_archive(monkeypatch):
    """Clamping only ever moves the window forward; it never widens what was asked for."""
    ingester = _ingester(monkeypatch)
    monkeypatch.setattr(ingester, "_series", ["CPIAUCSL"])
    monkeypatch.setattr(ingester, "_observation_start", "2020-01-01")
    monkeypatch.setattr(ingester, "_first_vintage", lambda code: "1972-07-21")

    asked: list[str] = []
    monkeypatch.setattr(
        ingester, "_get",
        lambda code, start, end: (asked.append(start), _Resp(200, {"observations": []}))[1],
    )
    ingester.fetch()
    assert asked == ["2020-01-01"]


def test_window_before_the_archive_is_empty_not_an_error(monkeypatch):
    """Inside a bisection, a half that predates the archive is empty, not a failure."""
    ingester = _ingester(monkeypatch)
    monkeypatch.setattr(ingester, "_get", lambda *a: _Resp(400, {
        "error_message": "Bad Request. The series does not exist in ALFRED but may exist "
                         "in FRED. Try setting realtime_start and realtime_end to today's "
                         "date or removing the realtime_start and realtime_end parameters."
    }))
    assert ingester.fetch_observations("T10Y2Y", "2000-01-01", "2001-01-01") == []


def test_unknown_series_code_fails_loudly(monkeypatch):
    """A typo in settings.yaml must raise, never quietly produce an empty series."""
    ingester = _ingester(monkeypatch)

    class _Session:
        RequestException = requests.RequestException

        @staticmethod
        def get(url, params=None, timeout=None):
            return _Resp(400, {"error_message": "Bad Request. The series does not exist."})

    monkeypatch.setattr("ingest.fred.requests", _Session)
    with pytest.raises(RuntimeError, match="settings.yaml"):
        ingester._first_vintage("CPIAUSCL")  # transposed letters


def test_first_vintage_is_the_earliest_of_every_vintage_span(monkeypatch):
    """FRED returns one metadata record per vintage span; the archive starts at the min."""
    ingester = _ingester(monkeypatch)

    class _Session:
        RequestException = requests.RequestException

        @staticmethod
        def get(url, params=None, timeout=None):
            return _Resp(200, {"seriess": [
                {"realtime_start": "2019-05-02"},
                {"realtime_start": "2014-01-27"},
                {"realtime_start": "2021-11-30"},
            ]})

    monkeypatch.setattr("ingest.fred.requests", _Session)
    assert ingester._first_vintage("T10Y2Y") == "2014-01-27"


def test_one_broken_series_does_not_cost_the_others(monkeypatch):
    """Eleven healthy series must survive a twelfth that fails (RESEARCH.md 1.6)."""
    ingester = _ingester(monkeypatch)
    monkeypatch.setattr(ingester, "_series", ["CPIAUCSL", "BROKEN", "DFF"])
    monkeypatch.setattr(ingester, "_first_vintage", lambda code: "2000-01-01")
    # DFF is configured for the pre-vintage backfill, which has its own request; left on,
    # this test reached the real FRED API. The backfill has its own test below.
    monkeypatch.setattr(ingester, "_pre_vintage", {})

    def fake_get(code, start, end):
        if code == "BROKEN":
            return _Resp(500, {"error_message": "internal error"})
        return _Resp(200, {"observations": [
            {"date": "2026-01-01", "realtime_start": "2026-02-11", "value": "1.0"}
        ]})

    monkeypatch.setattr(ingester, "_get", fake_get)
    df = ingester.fetch()

    assert set(df["series_id"]) == {"CPIAUCSL", "DFF"}, "healthy series were lost"
    assert ingester.partial_failures() == ["BROKEN"], "the failure must be reported"


# --- Before the vintage archive: derived publication dates (RESEARCH.md §2.30) -----------


def test_pre_vintage_rows_get_the_maximum_lag_in_business_days():
    """Friday + 3 business days is Wednesday: the weekend does not count, and the date
    errs late — a late date only makes the panel cautious, an early one is look-ahead."""
    obs = [{"date": "2008-10-10", "value": "69.95"}, {"date": "2008-10-13", "value": "54.99"}]
    out = derive_pre_vintage(obs, "VIXCLS", before="2010-11-22", lag_business_days=3)
    assert out["ts_release"].tolist() == ["2008-10-15T00:00:00+00:00",
                                          "2008-10-16T00:00:00+00:00"]
    assert out["ts"].iloc[0] == "2008-10-10T00:00:00+00:00"


def test_pre_vintage_rows_stop_where_real_vintages_start():
    obs = [{"date": "2010-11-19", "value": "18.0"}, {"date": "2010-11-22", "value": "19.0"}]
    out = derive_pre_vintage(obs, "VIXCLS", before="2010-11-22", lag_business_days=3)
    assert out["ts"].tolist() == ["2010-11-19T00:00:00+00:00"], "no overlap with real vintages"


def test_pre_vintage_drops_the_missing_sentinel():
    obs = [{"date": "2008-12-25", "value": "."}]
    assert derive_pre_vintage(obs, "VIXCLS", "2010-11-22", 3).empty


def test_only_unrevised_series_are_configured_for_backfill():
    """A revised series (NFCI, the dollar) back-filled with today's values would put
    later knowledge into 2008. The list is short on purpose and checked here; DFF is in
    it with its measured, immaterial revisions written next to it in the config."""
    from core.config import load_settings

    configured = load_settings().source("fred").get("pre_vintage_backfill") or {}
    assert set(configured) <= {"VIXCLS", "VXVCLS", "DFF"}


def test_a_failed_backfill_keeps_the_series_and_is_reported(monkeypatch):
    """The archived history is good data; losing the older, derived stretch must not cost
    it — and must reach the exit code."""
    ingester = _ingester(monkeypatch)
    monkeypatch.setattr(ingester, "_series", ["VIXCLS"])
    monkeypatch.setattr(ingester, "_pre_vintage", {"VIXCLS": 3})
    monkeypatch.setattr(ingester, "_first_vintage", lambda code: "2010-11-22")
    monkeypatch.setattr(ingester, "_get", lambda code, start, end: _Resp(200, {"observations": [
        {"date": "2010-11-22", "realtime_start": "2010-11-23", "value": "20.0"}]}))

    def broken(code, before):
        raise RuntimeError("backfill down")

    monkeypatch.setattr(ingester, "_backfill", broken)
    df = ingester.fetch()
    assert list(df["series_id"]) == ["VIXCLS"]
    assert ingester.partial_failures() == ["VIXCLS (pre-vintage)"]
