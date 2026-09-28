"""Business segments, from the XBRL of each filing (CLAUDE.md §15.5, point 15).

``companyfacts`` and ``frames`` carry only facts **without dimensions**, and a segment is a
dimension (``StatementBusinessSegmentsAxis``). The only free structured source is the XBRL
instance of each 10-Q and 10-K, published next to the document on EDGAR (``*_htm.xml`` for
inline XBRL). Verified on 2026-09-28 on Uber (Mobility, Delivery, Freight) and MercadoLibre
(one segment per country).

What is kept, for the researched companies and the positions held:

- facts whose context has **one** segment member and no other axis but
  ``ConsolidationItemsAxis = OperatingSegmentsMember`` (the segment's own figure, not an
  elimination, a product line or a country inside it — those would double count);
- only the concepts listed in ``sources.segments``: revenue (the same tags as the
  fundamentals) and the **segment profit measures**, which are company-specific. Uber
  tagged its measure ``uber:AdjustedEarningsBeforeInterestTaxesDepreciationAndAmortization``
  in 2025 and ``us-gaap:OperatingIncomeLoss`` from 2026, when it changed the definition;
  MercadoLibre tags ``meli:DirectContribution``. The concept is part of the series, so a
  change of measure shows as a change of concept and is never spliced (section 9.8);
- durations classified as ``sec_xbrl`` does (``q``, ``ytd2``, ``ytd3``, ``fy``): the
  quarter hidden in a cumulative is derived in ``transform/segments`` (sections 9.12, 9.14).

``ts_release`` is the filing date, observed. A later filing that repeats a period with
another value — the comparative of a restatement or of a new measure — is kept as another
version (section 9.6), and ``transform/segments`` says so.

Each filing is parsed once and its segment facts cached as JSON (a filed document does not
change). fetch() -> observations, source ``sec_segments``,
series ``{cik}:{member}:{concept}:{period}``.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import requests

from core.config import Settings
from core.logging_setup import get_logger
from ingest.base import Ingester, empty_observations, held_tickers, retry
from ingest.screen import cached_json
from ingest.sec_filings import SUBMISSIONS_URL, normalize_ticker, ticker_map
from ingest.sec_xbrl import _period_label

log = get_logger(__name__)

SOURCE = "sec_segments"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/"
FORMS = frozenset({"10-Q", "10-K", "10-Q/A", "10-K/A"})
SEGMENT_AXIS = "StatementBusinessSegmentsAxis"
CONSOLIDATION_AXIS = "ConsolidationItemsAxis"
OWN_SEGMENT = "OperatingSegmentsMember"

_CONTEXT = re.compile(r"<(?:\w+:)?context\b[^>]*\bid=\"([^\"]+)\"[^>]*>(.*?)</(?:\w+:)?context>",
                      re.S)
_MEMBER = re.compile(r"dimension=\"([^\"]+)\"\s*>\s*([^<\s]+)\s*<")
_PERIOD = re.compile(r"<(?:\w+:)?(startDate|endDate|instant)>\s*([^<\s]+)\s*<")
_FACT = re.compile(r"<([\w-]+):(\w+)\b([^>]*?\bcontextRef=\"[^\"]+\"[^>]*)>([^<]*)</\1:\2>")


def local(name: str) -> str:
    """``uber:MobilityMember`` → ``MobilityMember``."""
    return name.split(":")[-1]


def member_label(member: str) -> str:
    """``uber:MobilityMember`` → ``Mobility``; ``meli:BrazilSegmentMember`` → ``Brazil``."""
    name = local(member)
    for suffix in ("SegmentMember", "Member"):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def segment_contexts(xml: str) -> dict[str, tuple[str, dict[str, str]]]:
    """``{context id: (segment member, {start, end})}`` for contexts that are one segment's
    own figure: a single ``StatementBusinessSegmentsAxis`` member and, at most, the
    consolidation axis set to ``OperatingSegmentsMember``. Anything else is left out."""
    out = {}
    for cid, body in _CONTEXT.findall(xml):
        dims = _MEMBER.findall(body)
        segment = [m for d, m in dims if local(d) == SEGMENT_AXIS]
        others = [(d, m) for d, m in dims if local(d) != SEGMENT_AXIS]
        if len(segment) != 1:
            continue
        if any(local(d) != CONSOLIDATION_AXIS or local(m) != OWN_SEGMENT for d, m in others):
            continue
        period = dict(_PERIOD.findall(body))
        if "startDate" in period and "endDate" in period:
            out[cid] = (segment[0], {"start": period["startDate"], "end": period["endDate"]})
    return out


def segment_facts(xml: str, concepts: set[str]) -> list[dict[str, Any]]:
    """The wanted concepts' facts in segment contexts: ``[{member, concept, start, end,
    value}]``. ``concepts`` are local names (``Revenues``, ``DirectContribution``)."""
    contexts = segment_contexts(xml)
    out, seen = [], set()
    for _prefix, name, attrs, text in _FACT.findall(xml):
        if name not in concepts:
            continue
        cid = re.search(r"contextRef=\"([^\"]+)\"", attrs).group(1)
        if cid not in contexts or not text.strip():
            continue
        try:
            value = float(text.strip())
        except ValueError:
            continue
        # Inline XBRL states the sign in the value; a `sign="-"` attribute would be HTML.
        member, period = contexts[cid]
        key = (member, name, period["start"], period["end"])
        if key in seen:               # the same fact shown twice in the document
            continue
        seen.add(key)
        out.append({"member": member, "concept": name, **period, "value": value})
    return out


def observation_rows(cik: str, facts: Iterable[Mapping[str, Any]], filed: str,
                     windows: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Observations for one filing's segment facts, classified by duration."""
    rows = []
    for f in facts:
        label = _period_label({"start": f["start"], "end": f["end"]}, windows)
        if not label:                  # an instant or an unclassifiable duration
            continue
        rows.append({"source": SOURCE,
                     "series_id": f"{cik}:{member_label(f['member'])}:{f['concept']}:{label}",
                     "ts": f["end"], "ts_release": filed, "value": float(f["value"])})
    return rows


class SegmentsIngester(Ingester):
    """Segment revenue and profit from the latest filings of the watched companies."""

    source = SOURCE

    def __init__(self, settings: Settings) -> None:
        cfg = settings.source("sec")
        seg = dict(settings.source("segments"))
        self._base_url = str(cfg.get("base_url", "https://data.sec.gov")).rstrip("/")
        self._user_agent = settings.secret(str(cfg.get("user_agent_env", "SEC_USER_AGENT")))
        self._windows = dict(cfg.get("duration_windows", {}))
        cache = Path(str(cfg.get("cache_dir", ".cache/sec")))
        self._cache = cache
        self._parsed = cache / "segments"
        self._ticker_map_url = str(cfg.get("ticker_map_url",
                                           "https://www.sec.gov/files/company_tickers.json"))
        self._delay_s = 1.0 / float(cfg.get("rate_limit_rps", 8) or 8)
        revenue = list(dict(cfg.get("concepts", {})).get("revenue", {}).get("tags", []))
        self._concepts = set(revenue) | set(seg.get("profit_concepts", []))
        self._filings = int(seg.get("filings", 9))
        self._companies = [(str(c["cik"]).zfill(10), str(c["ticker"]))
                           for c in settings.researched_companies if c.get("cik")]
        self._failures: list[str] = []

    @staticmethod
    def is_available(settings: Settings) -> bool:
        cfg = settings.source("sec")
        agent = settings.secret(str(cfg.get("user_agent_env", "SEC_USER_AGENT")))
        return bool(agent) and bool(settings.source("segments").get("profit_concepts"))

    def attach_database(self, conn: Any) -> None:
        """The positions held without a card, as ``sec_filings`` adds them."""
        known = {normalize_ticker(t) for _, t in self._companies}
        missing = [t for t in held_tickers(conn) if normalize_ticker(t) not in known]
        if not missing:
            return
        try:
            mapping = ticker_map(cached_json(self._ticker_map_url,
                                             self._cache / "company_tickers.json",
                                             user_agent=self._user_agent,
                                             delay_s=self._delay_s))
        except Exception as exc:  # noqa: BLE001
            self._failures.append(f"company_tickers.json: {exc}")
            return
        ciks = {c for c, _ in self._companies}
        for ticker in missing:
            cik = mapping.get(normalize_ticker(ticker))
            if cik and cik not in ciks:
                self._companies.append((cik, ticker))
                ciks.add(cik)

    def _get(self, url: str) -> requests.Response:
        def call() -> requests.Response:
            response = requests.get(url, headers={"User-Agent": self._user_agent}, timeout=120)
            response.raise_for_status()
            return response
        response = retry(call, exceptions=(requests.RequestException,))
        time.sleep(self._delay_s)
        return response

    def _filing_facts(self, cik: str, accession: str) -> list[dict[str, Any]]:
        """One filing's segment facts, parsed once and cached (a filing never changes)."""
        path = self._parsed / f"{accession}.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if set(cached.get("concepts", [])) >= self._concepts:
                return cached["facts"]
        folder = ARCHIVE.format(cik=int(cik), folder=accession.replace("-", ""))
        names = [item["name"] for item in
                 self._get(folder + "index.json").json()["directory"]["item"]]
        instance = next((n for n in names if n.endswith("_htm.xml")), None) or next(
            (n for n in names if re.fullmatch(r"[a-z0-9]+-\d{8}\.xml", n)), None)
        facts = [] if instance is None else segment_facts(
            self._get(folder + instance).text, self._concepts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"concepts": sorted(self._concepts), "facts": facts}),
                        encoding="utf-8")
        return facts

    def fetch(self) -> pd.DataFrame:
        self._failures = []
        records: list[dict[str, Any]] = []
        with_segments = []
        for cik, ticker in self._companies:
            try:
                recent = cached_json(SUBMISSIONS_URL.format(base=self._base_url, cik=cik),
                                     self._cache / f"submissions_CIK{cik}.json",
                                     user_agent=self._user_agent, delay_s=self._delay_s,
                                     max_age_days=1)["filings"]["recent"]
            except Exception as exc:  # noqa: BLE001 — one company must not sink the rest
                self._failures.append(f"{ticker} submissions: {exc}")
                continue
            filings = [(recent["accessionNumber"][i], recent["filingDate"][i])
                       for i, form in enumerate(recent["form"]) if form in FORMS][: self._filings]
            found = 0
            for accession, filed in filings:
                try:
                    facts = self._filing_facts(cik, accession)
                except Exception as exc:  # noqa: BLE001 — one filing must not sink the rest
                    log.exception("Segments %s %s failed.", ticker, accession,
                                  extra={"source": SOURCE})
                    self._failures.append(f"{ticker} {accession}: {exc}")
                    continue
                rows = observation_rows(cik, facts, filed, self._windows)
                found += len(rows)
                records += rows
            if found:
                with_segments.append(ticker)
        log.info("Segments: %d value(s); companies reporting segments: %s (of %d read).",
                 len(records), with_segments or "none", len(self._companies),
                 extra={"source": SOURCE})
        return self.validate(pd.DataFrame(records)) if records else empty_observations()

    def partial_failures(self) -> list[str]:
        return list(self._failures)
