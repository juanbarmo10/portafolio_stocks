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

import numpy as np
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

# Substring of the 400 FRED returns when the requested real-time window predates the
# series' first ALFRED vintage. The message names ALFRED, not the window, so it reads
# like "this series does not exist" — see _first_vintage for why that reading is wrong.
_NOT_IN_ALFRED_MARKER = "does not exist in ALFRED"

# Sentinel real-time bounds: FRED accepts these and returns one record per vintage span,
# which is how the earliest archived vintage of a series is discovered.
_ALFRED_EPOCH = "1776-07-04"
_ALFRED_FOREVER = "9999-12-31"


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


def derive_pre_vintage(
    observations: list[dict], code: str, before: str, lag_business_days: int
) -> pd.DataFrame:
    """Rows for the dates **before** ALFRED's archive, with a DERIVED publication date.

    Only for series that are never revised — verified, not assumed (RESEARCH.md §2.30):
    for those, today's value of a past date *is* the value first published, and the only
    unknown is **when** it was published. That is derived as the reference date plus
    ``lag_business_days``, the **maximum** lag measured over the archived period, so the
    derived date errs late (a late date only makes the panel cautious; an early one is
    look-ahead, section 9.4).

    Args:
        observations: ``{"date", "value"}`` dicts of the series' current values.
        code: FRED series code.
        before: First reference date covered by real vintages; only earlier dates are kept.
        lag_business_days: Publication lag to add, in business days.

    Returns:
        Long DataFrame in the loader contract. Missing-value sentinels are dropped.
    """
    rows = [o for o in observations
            if o.get("value") not in (None, _MISSING) and str(o["date"]) < before]
    if not rows:
        return empty_observations()
    dates = np.array([str(o["date"]) for o in rows], dtype="datetime64[D]")
    released = np.busday_offset(dates, lag_business_days, roll="forward")
    return pd.DataFrame({
        "source": "fred",
        "series_id": code,
        "ts": [f"{d}T00:00:00+00:00" for d in dates.astype(str)],
        "ts_release": [f"{d}T00:00:00+00:00" for d in released.astype(str)],
        "value": [float(o["value"]) for o in rows],
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
        # Series allowed to extend before ALFRED's archive, with their measured maximum
        # publication lag. Only non-revised series belong here (see derive_pre_vintage).
        self._pre_vintage: dict[str, int] = {
            str(code): int(lag) for code, lag in (cfg.get("pre_vintage_backfill") or {}).items()
        }

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

        # Series that failed while others succeeded, reported via partial_failures().
        self._failures: list[str] = []
        # code -> earliest ALFRED vintage date, memoized across a run.
        self._first_vintage_cache: dict[str, str] = {}

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

    def _first_vintage(self, code: str) -> str:
        """Earliest date at which ALFRED archives a vintage of ``code``.

        A series' *observation* history and its *vintage* history are different things and
        start on different dates. T10Y2Y has observations back to 1976 but ALFRED only
        archives it from 2014-01-27; asking for initial releases before that returns a 400
        whose message ("the series does not exist in ALFRED") describes the window, not the
        series. Requesting from this date instead is what makes output_type=4 usable.

        Args:
            code: FRED series code.

        Returns:
            ISO date of the earliest archived vintage.

        Raises:
            RuntimeError: If FRED does not know the series at all — a typo in
                settings.yaml must fail loudly, never degrade into an empty series
                (section 10).
        """
        if code in self._first_vintage_cache:
            return self._first_vintage_cache[code]

        params = {
            "series_id": code,
            "api_key": self._api_key,
            "file_type": "json",
            "realtime_start": _ALFRED_EPOCH,
            "realtime_end": _ALFRED_FOREVER,
        }
        resp = retry(
            lambda: requests.get(
                f"{self._base_url}/series", params=params, timeout=self._timeout
            ),
            exceptions=(requests.RequestException,),
            **self._retry_kwargs,
        )
        if resp.status_code != 200:
            message = resp.json().get("error_message", "") if resp.content else ""
            raise RuntimeError(
                f"FRED {code}: cannot resolve series metadata (HTTP {resp.status_code} "
                f"{message[:120]}). Check the code in config/settings.yaml."
            )
        records = resp.json().get("seriess", [])
        if not records:
            raise RuntimeError(f"FRED {code}: series metadata came back empty.")

        first = min(str(r["realtime_start"]) for r in records)
        self._first_vintage_cache[code] = first
        return first

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

            # The window predates the series' first archived vintage. Inside a bisection
            # that is a legitimately empty half, not an error: fetch() has already clamped
            # the outer window, so reaching here means the left half fell entirely before
            # the archive begins.
            if _NOT_IN_ALFRED_MARKER in message:
                log.debug(
                    "No ALFRED vintages in %s..%s; treating as empty.",
                    realtime_start, realtime_end,
                    extra={"source": self.source, "series_id": code},
                )
                return []

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
        """Return every configured macro series as one long observations DataFrame.

        Each series is fetched independently: one broken code does not cost the other ten
        (see :meth:`Ingester.partial_failures`). The real-time window is clamped to the
        series' first archived vintage, and the clamp is logged — a series whose
        point-in-time history starts in 2014 must not look like one going back to 2000.
        """
        realtime_end = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        frames: list[pd.DataFrame] = []
        self._failures = []

        for code in self._series:
            try:
                first_vintage = self._first_vintage(code)
                realtime_start = max(self._observation_start, first_vintage)
                if realtime_start != self._observation_start:
                    log.info(
                        "Point-in-time history starts %s (requested %s): ALFRED archives no "
                        "earlier vintage.", realtime_start, self._observation_start,
                        extra={"source": self.source, "series_id": code},
                    )
                observations = self.fetch_observations(code, realtime_start, realtime_end)
                reduced = parse_initial_releases(observations, code)
            except Exception:
                log.exception(
                    "FRED series failed; continuing with the rest.",
                    extra={"source": self.source, "series_id": code},
                )
                self._failures.append(code)
                continue

            log.info(
                "FRED: %d initial-release points.", len(reduced),
                extra={"source": self.source, "series_id": code},
            )
            frames.append(reduced)

            if code in self._pre_vintage and not reduced.empty:
                try:
                    frames.append(self._backfill(code, reduced["ts"].min()[:10]))
                except Exception:
                    log.exception("Pre-vintage backfill failed.",
                                  extra={"source": self.source, "series_id": code})
                    self._failures.append(f"{code} (pre-vintage)")

        df = pd.concat(frames, ignore_index=True) if frames else empty_observations()
        return self.validate(df)

    def _backfill(self, code: str, before: str) -> pd.DataFrame:
        """Current values before the first archived date, with derived release dates."""
        params = {
            "series_id": code, "api_key": self._api_key, "file_type": "json",
            "observation_start": self._observation_start,
            "observation_end": (dt.date.fromisoformat(before) - dt.timedelta(days=1)).isoformat(),
        }
        resp = retry(
            lambda: requests.get(f"{self._base_url}/series/observations", params=params,
                                 timeout=self._timeout),
            exceptions=(requests.RequestException,), **self._retry_kwargs,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"FRED {code}: HTTP {resp.status_code} on the backfill")
        derived = derive_pre_vintage(resp.json().get("observations", []), code, before,
                                     self._pre_vintage[code])
        log.warning(
            "FRED: %d points before %s with a DERIVED publication date (+%d business days, "
            "the measured maximum); the value is taken as published (see the config note on "
            "revisions), the date is derived, not observed.",
            len(derived), before, self._pre_vintage[code],
            extra={"source": self.source, "series_id": code},
        )
        return derived

    def partial_failures(self) -> list[str]:
        """FRED codes that failed during the last :meth:`fetch`."""
        return list(self._failures)
