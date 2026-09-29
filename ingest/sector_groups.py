"""Annual figures for a sector group the user names — silver miners first (CLAUDE.md §15.5,
point 14).

The growth screen reads every SEC filer from ``frames``, but ``frames`` exist only for US GAAP
(``/frames/ifrs-full/…`` answers 404, verified 2026-09-29), and in some sectors most of the
companies are foreign filers: of the listed silver miners, only Hecla, Coeur and SSR Mining
file 10-K/10-Q; Pan American, First Majestic, Endeavour, Fortuna, Silvercorp, Avino and
Americas Gold & Silver file a 40-F in IFRS. Their ``companyfacts`` do carry the IFRS facts of
the annual report, so a group can be compared **on fiscal years**, on the same definitions,
whatever the taxonomy.

Each company is read in the taxonomy whose revenue is most recent (``us-gaap`` or
``ifrs-full``), with the tags of ``sources.sec.concepts`` or ``sources.sec.ifrs_concepts``.
Kept: the annual flows, the balance-sheet instants and the cover page's shares outstanding
(``dei``, the only count with a reliable scale: Avino's weighted average is filed a thousand
times too large). Quarters are dropped — a 40-F company has none, and mixing would compare a
year with a quarter.

``ts_release`` is ``filed``, observed; every version is kept (section 9.6).

fetch() -> observations, source ``sec_sector``: ``{cik}:{metric}:fy`` (flows),
``{cik}:{metric}`` (instants), ``{cik}:shares_outstanding`` and ``{cik}:ifrs`` (1 = read in
IFRS, dated at the latest fiscal year).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from core.config import Settings
from core.logging_setup import get_logger
from ingest.base import Ingester, empty_observations
from ingest.growth_screen import first_tickers
from ingest.screen import cached_json
from ingest.sec_xbrl import COMPANYFACTS_URL, PROVENANCE_SUFFIX, TAXONOMY, extract_metric

log = get_logger(__name__)

SOURCE = "sec_sector"
IFRS = "ifrs-full"
METRICS = ("revenue", "gross_profit", "operating_income", "net_income", "operating_cash_flow",
           "capex", "depreciation", "cash", "long_term_debt", "debt_current", "debt_total",
           "equity", "basic_shares")
ANNUAL_FORMS = ("10-K", "40-F", "20-F")


def latest_annual_revenue(facts: Mapping[str, Any], taxonomy: str,
                          tags: list[str]) -> str | None:
    """The end date of the newest annual revenue fact in ``taxonomy``, or ``None``."""
    ends = []
    for tag in tags:
        for entries in (facts.get(taxonomy, {}).get(tag, {}).get("units") or {}).values():
            ends += [e["end"] for e in entries
                     if e.get("fp") == "FY" and str(e.get("form", "")).startswith(ANNUAL_FORMS)
                     and e.get("start") and e.get("end")]
    return max(ends) if ends else None


def choose_taxonomy(facts: Mapping[str, Any], concepts: Mapping[str, Any],
                    ifrs_concepts: Mapping[str, Any]) -> str | None:
    """The taxonomy with the most recent annual revenue; US GAAP on a tie."""
    us = latest_annual_revenue(facts, TAXONOMY, list(concepts["revenue"]["tags"]))
    ifrs = latest_annual_revenue(facts, IFRS, list(ifrs_concepts["revenue"]["tags"]))
    if us is None and ifrs is None:
        return None
    if ifrs is None or (us is not None and us >= ifrs):
        return TAXONOMY
    return IFRS


def shares_outstanding(facts: Mapping[str, Any], cik: str) -> list[dict[str, Any]]:
    """The cover page's shares outstanding (``dei``), one row per cover date and filing."""
    entries = (facts.get("dei", {}).get("EntityCommonStockSharesOutstanding", {})
               .get("units", {}).get("shares") or [])
    seen = {}
    for e in entries:
        if e.get("val") is None or not e.get("end") or not e.get("filed"):
            continue
        seen[(e["end"], e["filed"])] = float(e["val"])
    return [{"source": SOURCE, "series_id": f"{cik}:shares_outstanding", "ts": end,
             "ts_release": filed, "value": value} for (end, filed), value in seen.items()]


def company_rows(facts: Mapping[str, Any], cik: str, concepts: Mapping[str, Any],
                 ifrs_concepts: Mapping[str, Any],
                 windows: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every kept observation of one company; empty without annual revenue."""
    taxonomy = choose_taxonomy(facts, concepts, ifrs_concepts)
    if taxonomy is None:
        return []
    specs = ifrs_concepts if taxonomy == IFRS else concepts
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        spec = specs.get(metric)
        if not spec:
            continue
        records, _ = extract_metric(facts, metric, spec, cik, windows, taxonomy=taxonomy)
        for r in records:
            sid = r["series_id"]
            core = sid.removesuffix(f":{PROVENANCE_SUFFIX}")
            # Flows and averages: the fiscal year only. Instants: every balance date (the
            # transform reads the one at the fiscal year's end).
            if core.endswith(":fy") or core == f"{cik}:{metric}":
                rows.append({**r, "source": SOURCE})
    end = latest_annual_revenue(facts, taxonomy, list(specs["revenue"]["tags"]))
    filed = max((r["ts_release"] for r in rows if r["series_id"] == f"{cik}:revenue:fy"
                 and r["ts"] == end), default=None)
    if end and filed:
        rows.append({"source": SOURCE, "series_id": f"{cik}:ifrs", "ts": end,
                     "ts_release": filed, "value": 1.0 if taxonomy == IFRS else 0.0})
    return rows + shares_outstanding(facts, cik)


class SectorGroupsIngester(Ingester):
    """``companyfacts`` fiscal years for every ticker of ``universe.sector_groups``."""

    source = SOURCE

    def __init__(self, settings: Settings) -> None:
        cfg = settings.source("sec")
        self._base_url = str(cfg.get("base_url", "https://data.sec.gov")).rstrip("/")
        self._user_agent = settings.secret(str(cfg.get("user_agent_env", "SEC_USER_AGENT")))
        self._concepts = dict(cfg.get("concepts", {}))
        self._ifrs = dict(cfg.get("ifrs_concepts", {}))
        self._windows = dict(cfg.get("duration_windows", {}))
        self._cache = Path(str(cfg.get("cache_dir", ".cache/sec")))
        self._ticker_map_url = str(cfg.get("ticker_map_url",
                                           "https://www.sec.gov/files/company_tickers.json"))
        self._delay_s = 1.0 / float(cfg.get("rate_limit_rps", 8) or 8)
        self._tickers = sorted({t for group in settings.sector_groups.values() for t in group})
        self._researched = {str(c["cik"]).zfill(10) for c in settings.researched_companies
                            if c.get("cik")}
        self._known: set[str] = set()
        self._companies: list[dict[str, Any]] = []
        self._failures: list[str] = []

    def attach_database(self, conn: Any) -> None:
        """The CIKs already in ``companies``: registering them again would overwrite the SIC
        code the growth screen wrote with ``None``."""
        try:
            self._known = {str(r[0]) for r in conn.execute("SELECT cik FROM companies")}
        except Exception as exc:  # noqa: BLE001 — no table yet: register every company
            log.debug("No companies table: %s", exc)
            self._known = set()

    @staticmethod
    def is_available(settings: Settings) -> bool:
        cfg = settings.source("sec")
        agent = settings.secret(str(cfg.get("user_agent_env", "SEC_USER_AGENT")))
        return bool(agent) and bool(settings.sector_groups)

    def fetch(self) -> pd.DataFrame:
        self._failures, self._companies = [], []
        raw_map = cached_json(self._ticker_map_url, self._cache / "company_tickers.json",
                              user_agent=self._user_agent, delay_s=self._delay_s)
        registry = first_tickers(raw_map)
        by_ticker = {ticker: cik for cik, (ticker, _) in registry.items()}
        records: list[dict[str, Any]] = []
        for ticker in self._tickers:
            cik = by_ticker.get(ticker)
            if cik is None:
                self._failures.append(f"sector group {ticker}: no CIK in the SEC ticker map "
                                      "(delisted or acquired?)")
                continue
            url = COMPANYFACTS_URL.format(base=self._base_url, cik=cik)
            try:
                payload = cached_json(url, self._cache / f"companyfacts_CIK{cik}.json",
                                      user_agent=self._user_agent, delay_s=self._delay_s,
                                      max_age_days=1)
            except Exception as exc:  # noqa: BLE001 — one company must not sink the rest
                log.exception("companyfacts %s failed.", ticker, extra={"source": SOURCE})
                self._failures.append(f"sector group {ticker}: {exc}")
                continue
            rows = company_rows(payload.get("facts", {}), cik, self._concepts, self._ifrs,
                                self._windows)
            if not rows:
                self._failures.append(f"sector group {ticker}: no annual revenue in US GAAP "
                                      "or IFRS")
                continue
            records += rows
            if cik not in self._researched and cik not in self._known:
                self._companies.append({
                    "cik": cik, "ticker": ticker, "name": registry[cik][1], "sector": None,
                    "thesis_category": None, "first_seen": dt.date.today().isoformat(),
                    "status": "active", "sic": None, "sic_description": None})
        log.info("Sector groups: %d value(s) for %d compan(ies).", len(records),
                 len(self._tickers) - len(self._failures), extra={"source": SOURCE})
        return self.validate(pd.DataFrame(records)) if records else empty_observations()

    def fetch_tables(self) -> dict[str, list[dict[str, Any]]]:
        return {"companies": self._companies} if self._companies else {}

    def partial_failures(self) -> list[str]:
        return list(self._failures)
