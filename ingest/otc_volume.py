"""Weekly off-exchange volume from FINRA, a proxy of retail participation (user request).

There is no free source of retail **direction** (whether retail buys or sells): that exists
only in paid data. What FINRA publishes for free, per symbol and week, is the volume traded
off-exchange, split in two (``otcMarket/weeklySummary``, verified on 2026-09-29):

- ``OTC_W_SMBL``: over-the-counter, **outside** alternative trading systems — mostly orders
  internalized by wholesalers (Citadel Securities, Virtu, G1, Jane Street…), which is where
  US retail orders are sent. On HIMS, week of 2026-08-24: Citadel 6,4 M shares, Virtu 3,1 M.
- ``ATS_W_SMBL``: inside alternative trading systems ("dark pools"), mostly institutional.

Their share of total volume (``transform/retail_flow``) measures **how much** retail trades a
stock, never which way. FINRA publishes each week two (Tier 1) to four (Tier 2) weeks later;
``ts_release`` is the last update date it reports (observed, and late on purpose when a week
was revised).

fetch() -> observations, source ``finra_otc``, series ``{TICKER}:otc_nonats:w`` and
``{TICKER}:ats:w`` (shares, ``ts`` = week start).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import requests

from core.config import Settings
from core.logging_setup import get_logger
from ingest.base import Ingester, empty_observations, held_tickers, retry

log = get_logger(__name__)

SOURCE = "finra_otc"
URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
SERIES = {"OTC_W_SMBL": "otc_nonats", "ATS_W_SMBL": "ats"}


def rows_for(ticker: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Observation rows from FINRA's symbol-level weekly records."""
    out = []
    for r in records:
        kind = SERIES.get(str(r.get("summaryTypeCode")))
        shares = r.get("totalWeeklyShareQuantity")
        if kind is None or shares is None or not r.get("weekStartDate"):
            continue
        release = str(r.get("lastUpdateDate") or r.get("initialPublishedDate") or "")[:10]
        out.append({"source": SOURCE, "series_id": f"{ticker}:{kind}:w",
                    "ts": str(r["weekStartDate"])[:10], "ts_release": release,
                    "value": float(shares)})
    return out


class OtcVolumeIngester(Ingester):
    """FINRA weekly off-exchange volume of the researched and held companies."""

    source = SOURCE

    def __init__(self, settings: Settings) -> None:
        cfg = dict(settings.source("otc_volume"))
        self._years = int(cfg.get("years", 3))
        self._tickers = sorted({str(c["ticker"]) for c in settings.researched_companies
                                if c.get("ticker")})
        self._failures: list[str] = []

    @staticmethod
    def is_available(settings: Settings) -> bool:
        return bool(settings.source("otc_volume"))

    def attach_database(self, conn: Any) -> None:
        self._tickers = sorted(set(self._tickers) | set(held_tickers(conn)))

    def _query(self, ticker: str, code: str, since: str) -> list[dict[str, Any]]:
        out, offset = [], 0
        while True:
            body = {"limit": 1000, "offset": offset,
                    "compareFilters": [
                        {"compareType": "EQUAL", "fieldName": "issueSymbolIdentifier",
                         "fieldValue": ticker},
                        {"compareType": "EQUAL", "fieldName": "summaryTypeCode",
                         "fieldValue": code}],
                    "dateRangeFilters": [{"fieldName": "weekStartDate", "startDate": since,
                                          "endDate": dt.date.today().isoformat()}]}

            def call(payload: dict[str, Any] = body) -> list[dict[str, Any]]:
                response = requests.post(URL, json=payload, timeout=90,
                                         headers={"Accept": "application/json"})
                response.raise_for_status()
                return response.json() if response.text.strip() else []

            page = retry(call, exceptions=(requests.RequestException,))
            out += page
            if len(page) < 1000:
                return out
            offset += 1000

    def fetch(self) -> pd.DataFrame:
        self._failures = []
        since = (dt.date.today() - dt.timedelta(days=365 * self._years)).isoformat()
        records: list[dict[str, Any]] = []
        for ticker in self._tickers:
            try:
                raw = [r for code in SERIES for r in self._query(ticker, code, since)]
            except Exception as exc:  # noqa: BLE001 — one ticker must not sink the rest
                self._failures.append(f"FINRA weekly {ticker}: {exc}")
                continue
            records += rows_for(ticker, raw)
        log.info("FINRA weekly off-exchange volume: %d value(s) for %d ticker(s).",
                 len(records), len(self._tickers), extra={"source": SOURCE})
        return self.validate(pd.DataFrame(records)) if records else empty_observations()

    def partial_failures(self) -> list[str]:
        return list(self._failures)
