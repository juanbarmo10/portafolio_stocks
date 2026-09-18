"""SEC EDGAR XBRL: audited fundamentals, point-in-time by construction (sections 4.4, 9.4).

This is the source that defines the project. Every XBRL fact carries ``end`` — the period
it describes — **and** ``filed`` — the day it became public. That is exactly the
``ts`` / ``ts_release`` pair the schema was built around, so the point-in-time discipline
that cost a bisection of FRED vintages to obtain comes free here. Free *if it is used*:
assigning a 10-K to "2025" when it was filed in February 2026 hands a backtest two months
of the future, systematically in its favour (section 9.4).

Three decisions carry most of the weight:

**Taxonomy is normalized, and the normalization is recorded.** Companies use different
concepts for the same thing, and the *same* company changes concept over time — Microsoft
reports ``Revenues`` until ~2010 and ``RevenueFromContractWithCustomerExcludingAssessedTax``
after ASC 606. Each metric is therefore an ordered preference list in config, and for every
stored value the ingester also stores **which preference produced it**, as a companion
``...:src`` series. Without that, a step in a revenue series is indistinguishable from a
company that changed tags, and section 9.8 asks for exactly this kind of provenance. When
no concept resolves, nothing is written: a visible hole beats an invented number (section 12).

**Durations are classified, never mixed.** ``companyfacts`` returns 3-, 6-, 9- and 12-month
figures for the same concept, all sharing an ``end`` date. The 6- and 9-month ones are
year-to-date cumulatives. Filing a YTD figure into the quarterly series would produce a
series that looks perfectly plausible and is wrong by a factor of two or three — the exact
failure mode section 12 exists to prevent. Facts outside the configured windows are
discarded **and counted**.

**Restatements are kept, not overwritten.** The same period gets reported again by a later
filing, sometimes with a different number. Both rows survive because ``ts_release`` is part
of the primary key (section 9.6); the panel shows the latest, a backtest uses the one in
force on its simulated date.

fetch() -> DataFrame[source, series_id, ts, ts_release, value]
    series_id = "{cik}:{metric}"         balance-sheet instants
              | "{cik}:{metric}:q"       ~quarterly flows
              | "{cik}:{metric}:fy"      ~annual flows
              | "<any of the above>:src" which preference produced that value
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import requests

from core.config import Settings
from core.logging_setup import get_logger
from ingest.base import Ingester, empty_observations, retry

log = get_logger(__name__)

SOURCE = "sec"
COMPANYFACTS_URL = "{base}/api/xbrl/companyfacts/CIK{cik}.json"
TAXONOMY = "us-gaap"
PROVENANCE_SUFFIX = "src"


class SecRequestError(RuntimeError):
    """EDGAR refused the request or answered something unusable."""


def _period_label(
    fact: Mapping[str, Any], windows: Mapping[str, Iterable[int]]
) -> str | None:
    """Classify a fact as instant, quarterly or annual.

    Args:
        fact: One entry of ``units[<unit>]`` in companyfacts.
        windows: ``{'quarter': [lo, hi], 'year': [lo, hi]}`` in days.

    Returns:
        ``''`` for an instant (no ``start``), ``'q'``, ``'fy'``, or ``None`` when the
        duration matches no window — a year-to-date cumulative, typically, which must not
        be filed as if it were a quarter.
    """
    start = fact.get("start")
    if not start:
        return ""
    try:
        days = (dt.date.fromisoformat(fact["end"]) - dt.date.fromisoformat(start)).days
    except (ValueError, KeyError, TypeError):
        return None
    quarter_lo, quarter_hi = windows["quarter"]
    year_lo, year_hi = windows["year"]
    if quarter_lo <= days <= quarter_hi:
        return "q"
    if year_lo <= days <= year_hi:
        return "fy"
    return None


def extract_metric(
    facts: Mapping[str, Any],
    metric: str,
    spec: Mapping[str, Any],
    cik: str,
    windows: Mapping[str, Iterable[int]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pull one normalized metric out of a companyfacts document.

    Concepts are tried in the configured order, and the **first** one that has a value for
    a given ``(period, end, filed)`` wins. Later preferences fill only the gaps, so a
    company that switched tags keeps one continuous series instead of two half ones — and
    the ``:src`` row says which tag was behind each point.

    Returns:
        ``(records, stats)`` — observation dicts, plus counters for what was skipped
        (wrong unit, unclassifiable duration, missing value).
    """
    concepts = facts.get(TAXONOMY, {})
    wanted_unit = spec.get("unit", "USD")
    stats = {"wrong_unit": 0, "unclassified_period": 0, "no_value": 0}
    chosen: dict[tuple[str, str, str], tuple[int, float]] = {}

    for preference, tag in enumerate(spec.get("tags", [])):
        concept = concepts.get(tag)
        if not concept:
            continue
        for unit, entries in concept.get("units", {}).items():
            if unit != wanted_unit:
                stats["wrong_unit"] += len(entries)
                continue
            for fact in entries:
                period = _period_label(fact, windows)
                if period is None:
                    stats["unclassified_period"] += 1
                    continue
                value, end, filed = fact.get("val"), fact.get("end"), fact.get("filed")
                if value is None or not end or not filed:
                    stats["no_value"] += 1
                    continue
                # First preference wins; a later one only fills a gap it left.
                chosen.setdefault((period, end, filed), (preference, float(value)))

    records: list[dict[str, Any]] = []
    for (period, end, filed), (preference, value) in chosen.items():
        series_id = f"{cik}:{metric}" + (f":{period}" if period else "")
        records.append({
            "source": SOURCE, "series_id": series_id,
            "ts": end, "ts_release": filed, "value": value,
        })
        # Provenance as a number, because `observations.value` is REAL: the index into the
        # preference list in config, which decodes back to the concept name. It is what
        # makes "revenue jumped 40%" answerable with "the company changed tags" instead of
        # a shrug (section 9.8).
        records.append({
            "source": SOURCE, "series_id": f"{series_id}:{PROVENANCE_SUFFIX}",
            "ts": end, "ts_release": filed, "value": float(preference),
        })
    return records, stats


def concept_for(spec: Mapping[str, Any], preference: float | int | None) -> str | None:
    """Decode a ``:src`` value back into the XBRL concept that produced the number."""
    if preference is None or pd.isna(preference):
        return None
    tags = list(spec.get("tags", []))
    index = int(preference)
    return tags[index] if 0 <= index < len(tags) else None


class SecXbrlIngester(Ingester):
    """Audited fundamentals for the written universe, one companyfacts call per company."""

    source = SOURCE

    def __init__(self, settings: Settings) -> None:
        cfg = settings.source("sec")
        self._base_url = str(cfg.get("base_url", "https://data.sec.gov")).rstrip("/")
        self._user_agent = settings.secret(str(cfg.get("user_agent_env", "SEC_USER_AGENT")))
        self._concepts: dict[str, Any] = dict(cfg.get("concepts", {}))
        self._windows = dict(cfg.get("duration_windows", {"quarter": [80, 100],
                                                          "year": [350, 380]}))
        self._cache_dir = Path(str(cfg.get("cache_dir", ".cache/sec")))
        self._cache_ttl = dt.timedelta(hours=float(cfg.get("cache_ttl_hours", 24)))
        self._delay_s = 1.0 / float(cfg.get("rate_limit_rps", 8) or 8)
        self._companies = self.companies_for(settings)
        retry_cfg = settings.raw.get("ingest", {}).get("retry", {})
        self._retry_kwargs = {
            "attempts": retry_cfg.get("attempts", 4),
            "base_delay_s": retry_cfg.get("base_delay_s", 1.0),
            "max_delay_s": retry_cfg.get("max_delay_s", 30.0),
        }
        self._failures: list[str] = []

    @staticmethod
    def companies_for(settings: Settings) -> list[tuple[str, str]]:
        """``(cik, ticker)`` for every tracked company that has a CIK.

        The CIK is the key, not the ticker (section 9.3). A thesis card without one is
        skipped rather than guessed at.
        """
        out: list[tuple[str, str]] = []
        for card in settings.tracked_companies:
            cik, ticker = card.get("cik"), card.get("ticker", "?")
            if cik:
                out.append((str(cik).zfill(10), str(ticker)))
        return out

    @staticmethod
    def is_available(settings: Settings) -> bool:
        """Needs an identifiable User-Agent and at least one company with a CIK."""
        cfg = settings.source("sec")
        agent = settings.secret(str(cfg.get("user_agent_env", "SEC_USER_AGENT")))
        return bool(agent) and bool(SecXbrlIngester.companies_for(settings))

    def _cache_path(self, cik: str) -> Path:
        return self._cache_dir / f"companyfacts_CIK{cik}.json"

    def _download(self, cik: str) -> dict[str, Any]:
        """One companyfacts document, from cache when fresh, from EDGAR otherwise.

        On a network failure an expired cache copy is used and reported, rather than
        losing the company for the run: stale audited data beats no data, as long as
        nobody is told it is fresh.
        """
        path = self._cache_path(cik)
        if path.exists():
            age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
            if age < self._cache_ttl:
                return json.loads(path.read_text(encoding="utf-8"))

        url = COMPANYFACTS_URL.format(base=self._base_url, cik=cik)

        def call() -> dict[str, Any]:
            # The SEC blocks requests without a User-Agent naming a real person and email
            # (section 4.4), and rate-limits around 10 req/s.
            response = requests.get(
                url,
                headers={"User-Agent": self._user_agent, "Accept-Encoding": "gzip, deflate"},
                timeout=60,
            )
            if response.status_code == 403:
                raise SecRequestError(
                    "EDGAR returned 403. The User-Agent must identify you with a real "
                    "name and email (SEC_USER_AGENT in config/.env, section 4.4)."
                )
            response.raise_for_status()
            return response.json()

        try:
            payload = retry(call, **self._retry_kwargs)
        except Exception as exc:
            if path.exists():
                self._failures.append(f"CIK {cik}: using expired cache ({exc})")
                log.warning(
                    "CIK %s: download failed (%s); falling back to the expired cache copy.",
                    cik, exc, extra={"source": SOURCE},
                )
                return json.loads(path.read_text(encoding="utf-8"))
            raise
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        time.sleep(self._delay_s)
        return payload

    def fetch(self) -> pd.DataFrame:
        """Normalized fundamentals for every tracked company with a CIK."""
        if not self._companies:
            log.warning(
                "No tracked company has a CIK. Write the thesis cards in "
                "settings.local.yaml (sections 5.2, 9.3).", extra={"source": SOURCE},
            )
            return empty_observations()

        records: list[dict[str, Any]] = []
        for cik, ticker in self._companies:
            try:
                payload = self._download(cik)
            except Exception as exc:
                # One company must not take the other twenty down (RESEARCH.md 1.6).
                log.exception("CIK %s (%s) failed.", cik, ticker, extra={"source": SOURCE})
                self._failures.append(f"CIK {cik} ({ticker}): {exc}")
                continue

            facts = payload.get("facts", {})
            company_records: list[dict[str, Any]] = []
            unresolved: list[str] = []
            for metric, spec in self._concepts.items():
                metric_records, stats = extract_metric(
                    facts, metric, spec, cik, self._windows
                )
                if not metric_records:
                    unresolved.append(metric)
                elif stats["unclassified_period"]:
                    log.debug(
                        "CIK %s %s: skipped %d fact(s) whose duration is neither a "
                        "quarter nor a year (year-to-date cumulatives).",
                        cik, metric, stats["unclassified_period"], extra={"source": SOURCE},
                    )
                company_records.extend(metric_records)

            if unresolved:
                # Visible, not estimated: no concept in the preference list resolved, so
                # the metric simply has no series for this company (section 12).
                log.warning(
                    "CIK %s (%s): no concept resolved for %s. Those metrics stay empty; "
                    "add a synonym to sources.sec.concepts if the company uses another tag.",
                    cik, ticker, ", ".join(unresolved), extra={"source": SOURCE},
                )
            log.info(
                "CIK %s (%s): %d observations over %d metrics.",
                cik, ticker, len(company_records),
                len(self._concepts) - len(unresolved), extra={"source": SOURCE},
            )
            records.extend(company_records)

        if not records:
            return empty_observations()
        return self.validate(self.observation_frame(records))

    def partial_failures(self) -> list[str]:
        """Companies that failed, or that were served from an expired cache."""
        return list(self._failures)
