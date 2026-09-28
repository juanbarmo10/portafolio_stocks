"""Annual figures for the peers the growth screen cannot read (CLAUDE.md §15.5, point 11).

The peer comparison of 🏢 Empresa reads each peer from the growth screen (``sec_growth``:
every SEC filer at its latest quarter, in USD). A few peers are not there: a foreign filer
that reports in US GAAP but **only annually and in its own currency** — DiDi files a 20-F
in yuan — never appears in a USD quarterly frame. For those, and only those, this ingester
reads the company's ``companyfacts`` and keeps the **fiscal years**, in the currency the
company reports revenue in. Every metric the comparison shows is a ratio (growth, margins,
dilution), so the currency cancels; nothing here is converted.

Unlike ``frames``, ``companyfacts`` carries the filing date: ``ts_release`` is ``filed``,
observed, and every version is kept (section 9.6).

fetch() -> observations, source ``sec_peers``, series ``{cik}:{metric}:fy``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import Settings
from core.logging_setup import get_logger
from ingest.base import Ingester, empty_observations
from ingest.growth_screen import SOURCE as GROWTH_SOURCE
from ingest.growth_screen import first_tickers
from ingest.screen import cached_json
from ingest.sec_xbrl import COMPANYFACTS_URL, TAXONOMY, extract_metric

log = get_logger(__name__)

SOURCE = "sec_peers"
METRICS = ("revenue", "gross_profit", "operating_income", "operating_cash_flow", "capex",
           "sbc", "basic_shares")


def reporting_currency(facts: dict[str, Any], revenue_tags: list[str]) -> str | None:
    """The unit most of the company's revenue facts are in (``USD``, ``CNY``…)."""
    counts: dict[str, int] = {}
    for tag in revenue_tags:
        for unit, entries in (facts.get(TAXONOMY, {}).get(tag, {}).get("units") or {}).items():
            counts[unit] = counts.get(unit, 0) + len(entries)
    return max(counts, key=lambda u: counts[u]) if counts else None


def annual_rows(facts: dict[str, Any], cik: str, concepts: dict[str, Any],
                windows: dict[str, Any]) -> list[dict[str, Any]]:
    """Fiscal-year observations of :data:`METRICS`, money in the reporting currency."""
    currency = reporting_currency(facts, list(concepts.get("revenue", {}).get("tags", [])))
    if currency is None:
        return []
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        spec = dict(concepts.get(metric) or {})
        if not spec:
            continue
        if spec.get("unit", "USD") == "USD":
            spec["unit"] = currency
        records, _stats = extract_metric(facts, metric, spec, cik, windows)
        rows += [{**r, "source": SOURCE} for r in records
                 if r["series_id"] == f"{cik}:{metric}:fy"]
    return rows


class PeersIngester(Ingester):
    """``companyfacts`` fiscal years for configured peers absent from the growth screen."""

    source = SOURCE

    def __init__(self, settings: Settings) -> None:
        cfg = settings.source("sec")
        self._base_url = str(cfg.get("base_url", "https://data.sec.gov")).rstrip("/")
        self._user_agent = settings.secret(str(cfg.get("user_agent_env", "SEC_USER_AGENT")))
        self._concepts = dict(cfg.get("concepts", {}))
        self._windows = dict(cfg.get("duration_windows", {}))
        cache = Path(str(cfg.get("cache_dir", ".cache/sec")))
        self._cache = cache
        self._ticker_map_url = str(cfg.get("ticker_map_url",
                                           "https://www.sec.gov/files/company_tickers.json"))
        self._delay_s = 1.0 / float(cfg.get("rate_limit_rps", 8) or 8)
        self._peers = sorted({p for group in settings.peers.values() for p in group})
        self._researched = {str(c["cik"]).zfill(10) for c in settings.researched_companies
                            if c.get("cik")}
        self._covered: set[str] = set()
        self._companies: list[dict[str, Any]] = []
        self._failures: list[str] = []

    @staticmethod
    def is_available(settings: Settings) -> bool:
        cfg = settings.source("sec")
        agent = settings.secret(str(cfg.get("user_agent_env", "SEC_USER_AGENT")))
        return bool(agent) and bool(settings.peers)

    def attach_database(self, conn: Any) -> None:
        """The CIKs the growth screen already reads: those need nothing from here."""
        try:
            rows = conn.execute("SELECT DISTINCT substr(series_id, 1, 10) FROM observations "
                                "WHERE source = ? AND series_id LIKE '%:revenue:%'",
                                (GROWTH_SOURCE,)).fetchall()
        except Exception as exc:  # noqa: BLE001 — no growth screen yet: fetch every peer
            log.debug("No growth screen rows: %s", exc)
            rows = []
        self._covered = {str(r[0]) for r in rows}

    def fetch(self) -> pd.DataFrame:
        self._failures, self._companies = [], []
        raw_map = cached_json(self._ticker_map_url, self._cache / "company_tickers.json",
                              user_agent=self._user_agent, delay_s=self._delay_s)
        registry = first_tickers(raw_map)
        by_ticker = {ticker: cik for cik, (ticker, _) in registry.items()}
        records: list[dict[str, Any]] = []
        for ticker in self._peers:
            cik = by_ticker.get(ticker)
            if cik is None:
                self._failures.append(f"peer {ticker}: no CIK in the SEC ticker map")
                continue
            if cik in self._covered:
                continue
            url = COMPANYFACTS_URL.format(base=self._base_url, cik=cik)
            try:
                payload = cached_json(url, self._cache / f"companyfacts_CIK{cik}.json",
                                      user_agent=self._user_agent, delay_s=self._delay_s)
            except Exception as exc:  # noqa: BLE001 — one peer must not sink the rest
                log.exception("companyfacts %s failed.", ticker, extra={"source": SOURCE})
                self._failures.append(f"peer {ticker}: {exc}")
                continue
            rows = annual_rows(payload.get("facts", {}), cik, self._concepts, self._windows)
            if not rows:
                self._failures.append(f"peer {ticker}: no US GAAP annual revenue (IFRS filer?)")
                continue
            records += rows
            if cik not in self._researched:
                self._companies.append({
                    "cik": cik, "ticker": ticker, "name": registry[cik][1], "sector": None,
                    "thesis_category": None, "first_seen": dt.date.today().isoformat(),
                    "status": "active", "sic": None, "sic_description": None})
        log.info("Peers: %d annual value(s) for %d peer(s) outside the growth screen.",
                 len(records), len(self._companies), extra={"source": SOURCE})
        return self.validate(pd.DataFrame(records)) if records else empty_observations()

    def fetch_tables(self) -> dict[str, list[dict[str, Any]]]:
        return {"companies": self._companies} if self._companies else {}

    def partial_failures(self) -> list[str]:
        return list(self._failures)
