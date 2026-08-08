"""FRED ingester: macro series with point-in-time release dates (CLAUDE.md sections 4.2, 9.4).

Level 1 of the checklist — "is there risk appetite?" — runs on this data.

This is where the look-ahead guard is enforced at the source. FRED data is revised, and a
datum's *reference* date (June CPI) precedes its *publication* date (mid-July). Both are
stored:

    ts          = reference date (the period the datum describes)
    ts_release  = the date the value was FIRST published (its initial release)
    value       = the value as first published (the initial print, not the latest revision)

Storing the initial print makes every point knowable at ``ts_release``, so a backtest
filtering ``ts_release <= sim_date`` never peeks into the future. Using the latest revision
would quietly rewrite history: today's CPI series for 2021 is not the series anyone traded
on in 2021.

Why the raw REST endpoint instead of ``fredapi``:
    Point-in-time data comes from ``series/observations`` with ``output_type=4`` ("initial
    release only"), which returns one row per observation stamped with its first-release
    ``realtime_start``. ``fredapi`` cannot request that mode, and its
    ``get_series_all_releases`` downloads *every* vintage, tripping FRED's 2000-vintage-date
    cap on daily series (DFF, T10Y2Y). Calling REST directly also lets us bisect the
    real-time window when that cap is hit, which works for any series frequency.

Requires FRED_API_KEY (free). Without it the runner skips this ingester rather than failing
the whole pipeline.

fetch() -> DataFrame[source, series_id, ts, ts_release, value]
    series_id = the FRED code, e.g. "CPIAUCSL".
"""

from __future__ import annotations

import datetime as dt
import re

import pandas as pd
import requests

from core.config import Settings
from core.logging_setup import get_logger
from ingest.base import Ingester, empty_observations, retry

log = get_logger(__name__)

# FRED's sentinel for a missing value in the observations endpoint.
_MISSING = "."

# Substring identifying the recoverable "too many vintage dates" 400 error.
_VINTAGE_LIMIT_MARKER = "vintage dates"

# FRED runs on US Central time, so a UTC-derived "today" can be a day ahead of FRED's
# calendar. When that happens FRED reports its own today in the 400 message; parse it
# and clamp realtime_end rather than failing on a one-day clock skew.
_FRED_TODAY_RE = re.compile(r"today's date \((\d{4}-\d{2}-\d{2})\)")


def parse_initial_releases(observations: list[dict], code: str) -> pd.DataFrame:
    """Convert FRED ``output_type=4`` observations to the long loader contract.

    Pure function, no network, so it is unit-testable against a frozen fixture without
    an API key (section 10).

    Args:
        observations: ``{"date", "realtime_start", "value"}`` dicts from the FRED
            observations endpoint with ``output_type=4``.
        code: FRED series code stamped on the output (e.g. "CPIAUCSL").

    Returns:
        Long DataFrame ``[source, series_id, ts, ts_release, value]``, one row per
        reference date. When a reference date appears more than once, the **earliest**
        release wins — that is the only vintage that was knowable at the time, and it
        also makes bisected windows safe to concatenate. Rows whose value is FRED's
        missing sentinel are dropped rather than stored as 0 (section 12).
    """
    earliest: dict[str, dict] = {}
    for obs in observations:
        value = obs.get("value")
        if value is None or value == _MISSING:
            continue
        ref = obs["date"]
        if ref not in earliest or obs["realtime_start"] < earliest[ref]["realtime_start"]:
            earliest[ref] = obs

    if not earliest:
        return empty_observations()

    refs = sorted(earliest)
    return pd.DataFrame({
        "source": "fred",
        "series_id": code,
        "ts": [f"{r}T00:00:00+00:00" for r in refs],
        "ts_release": [f"{earliest[r]['realtime_start']}T00:00:00+00:00" for r in refs],
        "value": [float(earliest[r]["value"]) for r in refs],
    })


class FredIngester(Ingester):
    """Fetch the configured FRED macro series with point-in-time release dates."""

    source = "fred"

    def __init__(self, settings: Settings) -> None:
        """Store config and the API key.

        Raises:
            RuntimeError: If FRED_API_KEY is absent. The runner checks availability
                first and skips, so reaching here without a key is a programming error.
        """
        cfg = settings.source("fred")
        self._base_url = cfg.get("base_url", "https://api.stlouisfed.org/fred").rstrip("/")
        self._output_type = cfg.get("output_type", 4)
        self._timeout = cfg.get("request_timeout_s", 20)
        self._observation_start = str(cfg.get("observation_start", "2000-01-01"))
        self._series: list[str] = list(cfg.get("series", []))

        retry_cfg = settings.raw.get("ingest", {}).get("retry", {})
        self._retry_kwargs = {
            "attempts": retry_cfg.get("attempts", 4),
            "base_delay_s": retry_cfg.get("base_delay_s", 1.0),
            "max_delay_s": retry_cfg.get("max_delay_s", 30.0),
        }

        api_key = settings.secret("FRED_API_KEY")
        if not api_key:
            raise RuntimeError("FRED_API_KEY not set; cannot construct FredIngester.")
        self._api_key = api_key

    @staticmethod
    def is_available(settings: Settings) -> bool:
        """Whether the prerequisites (API key, at least one series) are present."""
        return bool(settings.secret("FRED_API_KEY") and settings.source("fred").get("series"))

    def _get(self, code: str, realtime_start: str, realtime_end: str) -> requests.Response:
        """One observations request, retried on transient network errors."""
        params = {
            "series_id": code,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": self._observation_start,
            "realtime_start": realtime_start,
            "realtime_end": realtime_end,
            "output_type": self._output_type,  # 4 = initial release only
        }
        return retry(
            lambda: requests.get(
                f"{self._base_url}/series/observations", params=params, timeout=self._timeout
            ),
            exceptions=(requests.RequestException,),
            **self._retry_kwargs,
        )

    def fetch_observations(self, code: str, realtime_start: str, realtime_end: str) -> list[dict]:
        """Fetch initial-release observations, bisecting the real-time window if needed.

        FRED caps a request at 2000 vintage dates; daily series (DFF, T10Y2Y) exceed that
        over a multi-year window. On that specific 400 the real-time window is split in
        half and recursed: each initial release falls in exactly one half, so the union
        is complete and no observation is lost.

        Raises:
            RuntimeError: On any non-recoverable HTTP error — a changed upstream must
                surface, never degrade into an empty series (section 10).
        """
        resp = self._get(code, realtime_start, realtime_end)
        if resp.status_code == 200:
            return resp.json().get("observations", [])

        message = resp.json().get("error_message", "") if resp.content else ""
        if resp.status_code == 400:
            # Clock skew: realtime_end is past FRED's own today -> clamp and retry.
            match = _FRED_TODAY_RE.search(message)
            if match and realtime_end > match.group(1):
                return self.fetch_observations(code, realtime_start, match.group(1))

            if _VINTAGE_LIMIT_MARKER in message:
                start = dt.date.fromisoformat(realtime_start)
                end = dt.date.fromisoformat(realtime_end)
                if start >= end:
                    raise RuntimeError(f"FRED {code}: vintage limit on a single day {start}.")
                mid = start + (end - start) // 2
                left = self.fetch_observations(code, realtime_start, mid.isoformat())
                right = self.fetch_observations(
                    code, (mid + dt.timedelta(days=1)).isoformat(), realtime_end
                )
                return left + right

        raise RuntimeError(f"FRED {code}: HTTP {resp.status_code} {message[:160]}")

    def fetch(self) -> pd.DataFrame:
        """Return every configured macro series as one long observations DataFrame."""
        realtime_end = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        frames: list[pd.DataFrame] = []
        for code in self._series:
            observations = self.fetch_observations(code, self._observation_start, realtime_end)
            reduced = parse_initial_releases(observations, code)
            log.info(
                "FRED: %d initial-release points.", len(reduced),
                extra={"source": self.source, "series_id": code},
            )
            frames.append(reduced)

        df = pd.concat(frames, ignore_index=True) if frames else empty_observations()
        return self.validate(df)
