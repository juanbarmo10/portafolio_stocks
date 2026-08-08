"""FRED parser tests against a frozen fixture (CLAUDE.md sections 4.2, 9.4, 10).

No network and no API key: the parser is pure, so the point-in-time guarantee is
verifiable in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from db.loader import OBSERVATION_COLUMNS
from ingest.fred import FredIngester, parse_initial_releases

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
