"""Fundamentals for the quarterly screen, from the SEC ``frames`` API (CLAUDE.md §15.1).

Decided by the user on 2026-09-25: a **quarterly** screen of a fixed universe — the current
S&P 500 plus the researched companies — to find candidates by quality, shareholder value
accrual and valuation. Not an earnings-play screener (section 12): one run a quarter, on
audited annual figures.

``frames`` (section 4.4) returns one concept for **every** filer in one calendar period, so
~40 requests cover ~500 companies where ``companyfacts`` would need 500. Two limits, both
said on the page:

- **Calendar-aligned periods.** A fiscal year is filed under the calendar year it most
  overlaps: a June-to-May year lands in "CY2025". Good enough to rank, not to reconcile.
- **No filing date.** ``frames`` gives the accession, not ``filed``. These rows are stored
  with an unknown ``ts_release`` and serve today's screen only — never a backtest (§9.4).

**Industry (SIC) for the peer comparison.** ``frames`` does not carry it; the company's
``submissions`` file does (~150 KB each). Only a ``{cik: [sic, description, fetched]}`` map
is kept, in ``.cache/sec/sic_map.json``, and a CIK is fetched again only after
``SIC_REFRESH_DAYS`` — after the first run, a quarter costs a handful of requests. The code
goes to its own column of ``companies``: it is not GICS and never goes into ``sector``.

Taxonomy normalization is the same as ``sec_xbrl``: each metric's ordered tag list in
``sources.sec.concepts``, first tag with a value wins per company.

fetch() -> observations, source ``sec_frames``, series ``{cik}:{metric}:{period}``
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
from db.loader import TS_RELEASE_UNKNOWN
from ingest.base import Ingester, empty_observations, held_tickers, retry
from ingest.sec_filings import normalize_ticker, ticker_map

log = get_logger(__name__)

SOURCE = "sec_frames"
FRAMES_URL = "{base}/api/xbrl/frames/us-gaap/{tag}/{unit}/{period}.json"
SUBMISSIONS_URL = "{base}/submissions/CIK{cik}.json"
SIC_REFRESH_DAYS = 180


def screen_year(today: dt.date) -> int:
    """The latest calendar year most annual reports have been filed for.

    10-Ks land from February to April; before April the previous year is still thin.
    """
    return today.year - 1 if today.month >= 4 else today.year - 2


def periods_for(metric: str, spec: Mapping[str, Any], year: int) -> list[str]:
    """``CY{year}`` for flows and averages, ``CY{year}Q4I`` for balance-sheet instants.

    Revenue and the share count also get the year before, for growth and dilution.
    """
    if spec.get("kind") == "instant":
        return [f"CY{year}Q4I"]
    current = [f"CY{year}"]
    return current + [f"CY{year - 1}"] if metric in ("revenue", "diluted_shares") else current


def frame_rows(payload: Mapping[str, Any], metric: str, period: str,
               ciks: set[str] | None, preference: int, *,
               source: str = SOURCE) -> dict[str, dict[str, Any]]:
    """``{cik: row}`` for the wanted companies in one frames payload (``ciks=None``: every
    filer in it — the growth screen's universe).

    Raises:
        ValueError: If the payload has no ``data`` list — a changed endpoint fails loudly.
    """
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError(f"frames payload for {metric} {period} has no data list")
    out = {}
    for row in data:
        cik = str(row["cik"]).zfill(10)
        if (ciks is None or cik in ciks) and row.get("val") is not None:
            out[cik] = {"source": source, "series_id": f"{cik}:{metric}:{period}",
                        "ts": str(row["end"]), "ts_release": TS_RELEASE_UNKNOWN,
                        "value": float(row["val"]), "_preference": preference}
    return out


# Metrics whose tags can be a total or one of its parts: NVE Corp files its whole revenue as
# ``Revenues`` and a single line of it as ``RevenueFromContractWithCustomer…`` (11,0 M against
# 0,3 M in 2026-Q2); Valaris uses one tag in some periods and the other in others (539 M
# total against 7,5 M part). Found 2026-09-28 on the growth screen's first run.
LARGEST_IS_TOTAL = frozenset({"revenue"})


def resolve_tags(by_tag: Mapping[int, Mapping[str, dict[str, Any]]], *,
                 largest: bool) -> dict[str, dict[str, Any]]:
    """``{period: row}`` for one company and one metric, from its candidate tags
    (``{preference: {period: row}}``).

    **One tag per company**, not the first tag with a value in each period: mixing tags
    across periods turns a change of tag into a change of the business. The chosen tag is
    the first by preference — or, with ``largest`` (a total and its parts share the
    metric), the one that is larger in the periods the tags share, because a total is
    never smaller than one of its parts. Another tag fills a period the chosen one lacks
    (a company that changed tags) only if it was never smaller than the chosen one where
    both cover the same period (same end date): a part must not stand in for the total.
    """
    tags = sorted(by_tag)
    if not tags:
        return {}

    def shared(a: int, b: int) -> list[str]:
        # Comparable only when both facts end on the same day: Amcor's two tags in "CY2024"
        # are fiscal years ending in June and in September, not a total and its part.
        return [p for p in set(by_tag[a]) & set(by_tag[b])
                if str(by_tag[a][p].get("ts")) == str(by_tag[b][p].get("ts"))]

    best = tags[0]
    if largest and len(tags) > 1:
        wins = {t: 0 for t in tags}
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                for period in shared(a, b):
                    va, vb = by_tag[a][period]["value"], by_tag[b][period]["value"]
                    if va > vb:
                        wins[a] += 1
                    elif vb > va:
                        wins[b] += 1
        best = max(tags, key=lambda t: (wins[t], len(by_tag[t]), -t))
    out = dict(by_tag[best])
    for tag in tags:
        if tag == best:
            continue
        if largest and any(by_tag[tag][p]["value"] < by_tag[best][p]["value"]
                           for p in shared(tag, best)):
            continue
        for period, row in by_tag[tag].items():
            out.setdefault(period, row)
    return out


def resolved_records(candidates: Mapping[tuple[str, str], Mapping[int, Mapping[str, dict]]]
                     ) -> list[dict[str, Any]]:
    """Observation rows from ``{(cik, metric): {preference: {period: row}}}``."""
    return [{k: v for k, v in row.items() if k != "_preference"}
            for (_, metric), by_tag in candidates.items()
            for row in resolve_tags(by_tag, largest=metric in LARGEST_IS_TOTAL).values()]


def cached_json(url: str, path: Path, *, user_agent: str, delay_s: float,
                max_age_days: int = 7) -> dict[str, Any]:
    """A SEC JSON document, from ``path`` when younger than ``max_age_days``. A 404 is an
    empty frame (nobody filed that tag for the period), not an error."""
    if path.exists():
        age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
        if age < dt.timedelta(days=max_age_days):
            return json.loads(path.read_text(encoding="utf-8"))

    def call() -> dict[str, Any]:
        response = requests.get(url, headers={"User-Agent": user_agent}, timeout=90)
        if response.status_code == 404:
            return {"data": []}
        response.raise_for_status()
        return response.json()

    payload = retry(call, exceptions=(requests.RequestException,))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(delay_s)
    return payload


def sic_map(ciks: Iterable[str], path: Path, *, base_url: str, user_agent: str,
            delay_s: float, failures: list[str]) -> dict[str, list[Any]]:
    """``{cik: [sic, description, fetched_iso]}`` from the SEC submissions, cached in
    ``path`` and fetched again only after ``SIC_REFRESH_DAYS``. A failed company is
    appended to ``failures`` and left out, never guessed. Shared with the factor study."""
    cached: dict[str, list[Any]] = {}
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
    cutoff = (dt.date.today() - dt.timedelta(days=SIC_REFRESH_DAYS)).isoformat()
    wanted = set(ciks)
    stale = sorted(c for c in wanted if c not in cached or cached[c][2] < cutoff)
    for i, cik in enumerate(stale):
        url = SUBMISSIONS_URL.format(base=base_url, cik=cik)
        try:
            response = retry(lambda u=url: requests.get(
                u, headers={"User-Agent": user_agent}, timeout=60),
                exceptions=(requests.RequestException,))
            response.raise_for_status()
            body = response.json()
            cached[cik] = [str(body.get("sic") or "") or None,
                           body.get("sicDescription") or None, dt.date.today().isoformat()]
        except Exception as exc:  # noqa: BLE001 — one company must not sink the run
            failures.append(f"SIC {cik}: {exc}")
        time.sleep(delay_s)
        if i % 100 == 99:   # progress survives an interruption
            path.write_text(json.dumps(cached), encoding="utf-8")
    if stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached), encoding="utf-8")
        log.info("SIC fetched for %d compan(ies); %d from cache.", len(stale),
                 len(wanted) - len(stale), extra={"source": SOURCE})
    return cached


class ScreenIngester(Ingester):
    """Annual fundamentals of the screen's universe, quarterly."""

    source = SOURCE

    def __init__(self, settings: Settings) -> None:
        cfg = settings.source("sec")
        screen = settings.raw.get("screen", {})
        self._base_url = str(cfg.get("base_url", "https://data.sec.gov")).rstrip("/")
        self._user_agent = settings.secret(str(cfg.get("user_agent_env", "SEC_USER_AGENT")))
        self._concepts = {m: spec for m, spec in dict(cfg.get("concepts", {})).items()
                          if m in set(screen.get("metrics", []))}
        self._cache_dir = Path(str(cfg.get("cache_dir", ".cache/sec"))) / "frames"
        self._ticker_map_path = Path(str(cfg.get("cache_dir", ".cache/sec"))) / "company_tickers.json"
        self._sic_path = Path(str(cfg.get("cache_dir", ".cache/sec"))) / "sic_map.json"
        self._ticker_map_url = str(cfg.get("ticker_map_url",
                                           "https://www.sec.gov/files/company_tickers.json"))
        self._delay_s = 1.0 / float(cfg.get("rate_limit_rps", 8) or 8)
        self._every_days = int(screen.get("run_every_days", 90))
        self._researched = [str(c["ticker"]) for c in settings.researched_companies
                            if c.get("ticker")]
        self.force = False
        self._skip_reason: str | None = None
        self._researched_ciks = {str(c["cik"]).zfill(10) for c in settings.researched_companies
                                 if c.get("cik")}
        self._tickers: list[str] = []
        self._companies: list[dict[str, Any]] = []
        self._failures: list[str] = []

    @staticmethod
    def is_available(settings: Settings) -> bool:
        cfg = settings.source("sec")
        agent = settings.secret(str(cfg.get("user_agent_env", "SEC_USER_AGENT")))
        return bool(agent) and bool(settings.raw.get("screen", {}).get("metrics"))

    def attach_database(self, conn: Any) -> None:
        """The universe (current S&P 500 members + researched + held) and whether it is due."""
        try:
            rows = conn.execute("SELECT ticker FROM universe_membership "
                                "WHERE universe = 'sp500' AND end_date IS NULL").fetchall()
        except Exception as exc:  # noqa: BLE001 — no membership yet: researched only
            log.debug("No membership table: %s", exc)
            rows = []
        self._tickers = sorted({str(r[0]) for r in rows} | set(self._researched)
                               | set(held_tickers(conn)))
        if self.force or self._every_days <= 0:
            return
        try:
            last = conn.execute("SELECT MAX(ingested_at) FROM observations "
                                "WHERE source = ?", (SOURCE,)).fetchone()
        except Exception:  # noqa: BLE001
            return
        if last and last[0]:
            age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(str(last[0]))
            if age < dt.timedelta(days=self._every_days):
                self._skip_reason = (f"last run {age.days} day(s) ago, due every "
                                     f"{self._every_days}; `--force` to run it now")

    def _get_json(self, url: str, path: Path) -> dict[str, Any]:
        return cached_json(url, path, user_agent=self._user_agent, delay_s=self._delay_s)

    def _sic_map(self, ciks: set[str]) -> dict[str, list[Any]]:
        """``{cik: [sic, description, fetched_iso]}``, fetching only what is missing or old."""
        return sic_map(ciks, self._sic_path, base_url=self._base_url,
                       user_agent=self._user_agent, delay_s=self._delay_s,
                       failures=self._failures)

    def fetch(self) -> pd.DataFrame:
        if self._skip_reason:
            log.info("Screen skipped: %s.", self._skip_reason, extra={"source": SOURCE})
            return empty_observations()
        self._failures = []
        raw_map = self._get_json(self._ticker_map_url, self._ticker_map_path)
        mapping = ticker_map(raw_map)
        ciks = {mapping[normalize_ticker(t)] for t in self._tickers
                if normalize_ticker(t) in mapping}
        # The registry gains the screen's universe, so the page can name each CIK. The
        # researched companies are left to sec_filings, which writes their card's sector.
        names = {str(r["cik_str"]).zfill(10): str(r["title"]) for r in raw_map.values()}
        today = dt.date.today().isoformat()
        sic = self._sic_map(ciks)
        self._companies = [
            {"cik": mapping[normalize_ticker(t)], "ticker": t,
             "name": names.get(mapping[normalize_ticker(t)]), "sector": None,
             "thesis_category": None, "first_seen": today, "status": "active",
             "sic": (sic.get(mapping[normalize_ticker(t)]) or [None])[0],
             "sic_description": (sic.get(mapping[normalize_ticker(t)]) or [None, None])[1]}
            for t in self._tickers if normalize_ticker(t) in mapping
            and mapping[normalize_ticker(t)] not in self._researched_ciks
        ]
        missing = [t for t in self._tickers if normalize_ticker(t) not in mapping]
        if missing:
            log.info("%d ticker(s) without a SEC CIK left out of the screen (ETFs, recent "
                     "renames): %s", len(missing), missing[:10], extra={"source": SOURCE})
        year = screen_year(dt.date.today())
        candidates: dict[tuple[str, str], dict[int, dict[str, dict[str, Any]]]] = {}
        for metric, spec in self._concepts.items():
            unit = spec.get("unit", "USD")
            for period in periods_for(metric, spec, year):
                for preference, tag in enumerate(spec.get("tags", [])):
                    url = FRAMES_URL.format(base=self._base_url, tag=tag, unit=unit,
                                            period=period)
                    try:
                        payload = self._get_json(url, self._cache_dir / f"{tag}_{unit}_{period}.json")
                        rows = frame_rows(payload, metric, period, ciks, preference)
                    except Exception as exc:  # noqa: BLE001 — one tag must not sink the rest
                        log.exception("frames %s %s failed.", tag, period,
                                      extra={"source": SOURCE})
                        self._failures.append(f"{tag} {period}: {exc}")
                        continue
                    for cik, row in rows.items():
                        candidates.setdefault((cik, metric), {}).setdefault(
                            preference, {})[period] = row
        records = resolved_records(candidates)
        log.info("Screen: %d values for %d companies, fiscal year CY%d.", len(records),
                 len(ciks), year, extra={"source": SOURCE})
        if not records:
            return empty_observations()
        return self.validate(pd.DataFrame(records))

    def fetch_tables(self) -> dict[str, list[dict[str, Any]]]:
        return {"companies": self._companies} if self._companies else {}

    def partial_failures(self) -> list[str]:
        return list(self._failures)
