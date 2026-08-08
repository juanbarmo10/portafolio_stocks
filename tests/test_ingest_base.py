"""Ingest base-class tests: retry policy and the fetch() contract (CLAUDE.md sections 7, 10)."""

from __future__ import annotations

import pandas as pd
import pytest

from db.loader import OBSERVATION_COLUMNS, TS_RELEASE_UNKNOWN
from ingest.base import Ingester, empty_observations, retry


def test_retry_returns_on_first_success():
    assert retry(lambda: 42) == 42


def test_retry_recovers_after_transient_failures(monkeypatch):
    """The IBKR Flex handshake is asynchronous: the first GetStatement legitimately
    comes back not-ready (section 4.1), so a transient failure must not be fatal."""
    monkeypatch.setattr("ingest.base.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("statement not ready")
        return "xml"

    assert retry(flaky, base_delay_s=0) == "xml"
    assert calls["n"] == 3


def test_retry_reraises_after_exhausting_attempts(monkeypatch):
    """A dead source fails loudly; it never returns an empty result (section 10)."""
    monkeypatch.setattr("ingest.base.time.sleep", lambda _s: None)

    def always_fails():
        raise ConnectionError("upstream down")

    with pytest.raises(ConnectionError):
        retry(always_fails, attempts=3, base_delay_s=0)


def test_retry_does_not_swallow_unlisted_exceptions(monkeypatch):
    monkeypatch.setattr("ingest.base.time.sleep", lambda _s: None)

    def bad_parse():
        raise ValueError("upstream structure changed")

    with pytest.raises(ValueError):
        retry(bad_parse, attempts=3, base_delay_s=0, exceptions=(ConnectionError,))


class _Fake(Ingester):
    source = "fake"

    def fetch(self) -> pd.DataFrame:  # pragma: no cover - not exercised here
        return empty_observations()


def test_validate_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        _Fake.validate(pd.DataFrame([{"source": "fake", "ts": "2026-01-02"}]))


def test_validate_rejects_null_key_columns():
    """A null in a key column means the parser dropped a row silently (section 10)."""
    df = pd.DataFrame([{
        "source": "fake", "series_id": None, "ts": "2026-01-02",
        "ts_release": "", "value": 1.0,
    }])
    with pytest.raises(ValueError, match="null 'series_id'"):
        _Fake.validate(df)


def test_validate_allows_null_value():
    """A missing figure is legitimate and stays visible as None (section 12)."""
    df = pd.DataFrame([{
        "source": "fake", "series_id": "X", "ts": "2026-01-02",
        "ts_release": "", "value": None,
    }])
    assert _Fake.validate(df) is df


def test_observation_frame_defaults_unknown_release_date():
    df = Ingester.observation_frame(
        [{"source": "fake", "series_id": "X", "ts": "2026-01-02", "value": 1.0}]
    )
    assert list(df.columns) == OBSERVATION_COLUMNS
    assert df.loc[0, "ts_release"] == TS_RELEASE_UNKNOWN


def test_empty_observations_honors_the_contract():
    df = empty_observations()
    assert df.empty
    assert list(df.columns) == OBSERVATION_COLUMNS
    assert Ingester.validate(df) is df
