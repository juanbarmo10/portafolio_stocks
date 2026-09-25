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

Taxonomy normalization is the same as ``sec_xbrl``: each metric's ordered tag list in
``sources.sec.concepts``, first tag with a value wins per company.

fetch() -> observations, source ``sec_frames``, series ``{cik}:{metric}:{period}``
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Mapping

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
               ciks: set[str], preference: int) -> dict[str, dict[str, Any]]:
    """``{cik: row}`` for the wanted companies in one frames payload.

    Raises:
        ValueError: If the payload has no ``data`` list — a changed endpoint fails loudly.
    """
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError(f"frames payload for {metric} {period} has no data list")
    out = {}
    for row in data:
        cik = str(row["cik"]).zfill(10)
        if cik in ciks and row.get("val") is not None:
            out[cik] = {"source": SOURCE, "series_id": f"{cik}:{metric}:{period}",
                        "ts": str(row["end"]), "ts_release": TS_RELEASE_UNKNOWN,
                        "value": float(row["val"]), "_preference": preference}
    return out


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
        if path.exists():
            age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
            if age < dt.timedelta(days=7):
                return json.loads(path.read_text(encoding="utf-8"))

        def call() -> dict[str, Any]:
            response = requests.get(url, headers={"User-Agent": self._user_agent}, timeout=90)
            if response.status_code == 404:
                return {"data": []}          # nobody filed that tag for the period
            response.raise_for_status()
            return response.json()

        payload = retry(call, exceptions=(requests.RequestException,))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        time.sleep(self._delay_s)
        return payload

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
        self._companies = [
            {"cik": mapping[normalize_ticker(t)], "ticker": t,
             "name": names.get(mapping[normalize_ticker(t)]), "sector": None,
             "thesis_category": None, "first_seen": today, "status": "active"}
            for t in self._tickers if normalize_ticker(t) in mapping
            and mapping[normalize_ticker(t)] not in self._researched_ciks
        ]
        missing = [t for t in self._tickers if normalize_ticker(t) not in mapping]
        if missing:
            log.info("%d ticker(s) without a SEC CIK left out of the screen (ETFs, recent "
                     "renames): %s", len(missing), missing[:10], extra={"source": SOURCE})
        year = screen_year(dt.date.today())
        chosen: dict[tuple[str, str], dict[str, Any]] = {}
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
                        chosen.setdefault((cik, f"{metric}:{period}"), row)
        records = [{k: v for k, v in row.items() if k != "_preference"}
                   for row in chosen.values()]
        log.info("Screen: %d values for %d companies, fiscal year CY%d.", len(records),
                 len(ciks), year, extra={"source": SOURCE})
        if not records:
            return empty_observations()
        return self.validate(pd.DataFrame(records))

    def fetch_tables(self) -> dict[str, list[dict[str, Any]]]:
        return {"companies": self._companies} if self._companies else {}

    def partial_failures(self) -> list[str]:
        return list(self._failures)
