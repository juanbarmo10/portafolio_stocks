"""Insider purchases and sales of the watched companies, from each Form 4 (§15.5, user request).

Context, not a signal: the studies found that insider purchases do **not** precede better
returns, in the S&P 500 (RESEARCH.md §2.42) or outside it (§2.62). What the user wants is
the *trend* of insiders' buying and selling as information when reading a company, next to
everything else on its page.

Read from each Form 4 of the researched and held companies, listed in the issuer's
submissions (not from the quarterly data sets, which arrive a quarter late). Each filing is
parsed once and cached as JSON in ``.cache/sec/form4``: a filed document does not change.

What counts (verified on HIMS's filings, 2026-09-29):

- **Non-derivative** transactions with code **P** (open-market or private purchase) or **S**
  (open-market sale). Option exercises (M), tax withholding (F), grants (A) and gifts (G) are
  compensation mechanics, not decisions to buy or sell.
- By an owner who is a **director or an officer**. A "10 % owner" that is neither is nearly
  always a fund; its trading is portfolio management (same rule as the study).
- A sale is **plan** when the filing ticks ``aff10b5One`` (a Rule 10b5-1 plan, set up months
  earlier and far less informative), **discretionary** when it says no, and **unknown** before
  the box existed (filings before April 2023).
- Amendments (4/A) are left out: they repeat a transaction already filed.

Stored per day and filing date (``ts`` = transaction date, ``ts_release`` = filing date), as
``{cik}:insider:{buy|sell}:{plan|discretionary|unknown}`` (USD) and ``...:owners`` (how many
insiders). Names are not kept: the role is enough, and a person's name adds nothing a chart
needs. fetch() -> observations, source ``sec_form4``.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from core.config import Settings
from core.logging_setup import get_logger
from ingest.base import Ingester, empty_observations, held_tickers, retry
from ingest.screen import cached_json
from ingest.sec_filings import SUBMISSIONS_URL, normalize_ticker, ticker_map

log = get_logger(__name__)

SOURCE = "sec_form4"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{document}"
CODES = {"P": "buy", "S": "sell"}

_TRANSACTION = re.compile(r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>", re.S)


def _field(xml: str, tag: str) -> str | None:
    """The value of ``<tag>`` (or of its ``<value>`` child), or ``None``."""
    match = re.search(rf"<{tag}>\s*(?:<value>)?\s*([^<]*?)\s*(?:</value>\s*)?</{tag}>", xml,
                      re.S)
    return match.group(1).strip() if match else None


def parse_form4(xml: str) -> list[dict[str, Any]]:
    """The P and S transactions of one Form 4 by a director or an officer:
    ``[{kind, plan, date, shares, price, value, role}]``. Empty for anything else."""
    director = (_field(xml, "isDirector") or "0").strip() in ("1", "true")
    officer = (_field(xml, "isOfficer") or "0").strip() in ("1", "true")
    if not (director or officer):
        return []
    flag = _field(xml, "aff10b5One")
    plan = ("plan" if flag in ("1", "true") else "discretionary" if flag in ("0", "false")
            else "unknown")
    role = (_field(xml, "officerTitle") or "").strip() or ("Consejero" if director else "")
    out = []
    for block in _TRANSACTION.findall(xml):
        code = _field(block, "transactionCode")
        if code not in CODES:
            continue
        try:
            shares = float(_field(block, "transactionShares") or "nan")
            price = float(_field(block, "transactionPricePerShare") or "nan")
        except ValueError:
            continue
        if not shares > 0 or not price > 0:
            continue
        out.append({"kind": CODES[code], "plan": plan if code == "S" else "none",
                    "date": (_field(block, "transactionDate") or "")[:10],
                    "shares": shares, "price": price, "value": shares * price, "role": role})
    return out


def observation_rows(cik: str, filings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Daily sums per (kind, plan, transaction day, filing day): USD and distinct filings.

    ``filings``: ``[{filed, owner, transactions}]`` — ``owner`` only to count insiders, and
    never stored.
    """
    sums: dict[tuple[str, str, str], float] = {}
    owners: dict[tuple[str, str, str], set[str]] = {}
    for filing in filings:
        for t in filing["transactions"]:
            if not t["date"]:
                continue
            series = f"{cik}:insider:{t['kind']}" + (f":{t['plan']}" if t["kind"] == "sell"
                                                     else "")
            key = (series, t["date"], filing["filed"])
            sums[key] = sums.get(key, 0.0) + t["value"]
            owners.setdefault(key, set()).add(filing["owner"])
    rows = []
    for (series, day, filed), value in sums.items():
        rows.append({"source": SOURCE, "series_id": series, "ts": day, "ts_release": filed,
                     "value": value})
        rows.append({"source": SOURCE, "series_id": f"{series}:owners", "ts": day,
                     "ts_release": filed, "value": float(len(owners[(series, day, filed)]))})
    return rows


class InsiderActivityIngester(Ingester):
    """Form 4 purchases and sales of the researched and held companies."""

    source = SOURCE

    def __init__(self, settings: Settings) -> None:
        sec = settings.source("sec")
        cfg = dict(settings.source("insider_activity"))
        self._base_url = str(sec.get("base_url", "https://data.sec.gov")).rstrip("/")
        self._user_agent = settings.secret(str(sec.get("user_agent_env", "SEC_USER_AGENT")))
        self._cache = Path(str(sec.get("cache_dir", ".cache/sec")))
        self._ticker_map_url = str(sec.get("ticker_map_url",
                                           "https://www.sec.gov/files/company_tickers.json"))
        self._delay_s = 1.0 / float(sec.get("rate_limit_rps", 8) or 8)
        self._years = int(cfg.get("years", 3))
        self._companies = [(str(c["cik"]).zfill(10), str(c["ticker"]))
                           for c in settings.researched_companies if c.get("cik")]
        self._failures: list[str] = []

    @staticmethod
    def is_available(settings: Settings) -> bool:
        sec = settings.source("sec")
        agent = settings.secret(str(sec.get("user_agent_env", "SEC_USER_AGENT")))
        return bool(agent) and bool(settings.source("insider_activity"))

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

    def _filing(self, cik: str, accession: str, document: str) -> dict[str, Any]:
        """One Form 4, parsed once and cached."""
        path = self._cache / "form4" / f"{accession}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        url = ARCHIVE.format(cik=int(cik), folder=accession.replace("-", ""),
                             document=document.split("/")[-1])

        def call() -> str:
            response = requests.get(url, headers={"User-Agent": self._user_agent}, timeout=60)
            response.raise_for_status()
            return response.text

        xml = retry(call, exceptions=(requests.RequestException,))
        time.sleep(self._delay_s)
        parsed = {"owner": _field(xml, "rptOwnerCik") or accession,
                  "transactions": parse_form4(xml)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(parsed), encoding="utf-8")
        return parsed

    def fetch(self) -> pd.DataFrame:
        self._failures = []
        since = (dt.date.today() - dt.timedelta(days=365 * self._years)).isoformat()
        records: list[dict[str, Any]] = []
        for cik, ticker in self._companies:
            try:
                recent = cached_json(SUBMISSIONS_URL.format(base=self._base_url, cik=cik),
                                     self._cache / f"submissions_CIK{cik}.json",
                                     user_agent=self._user_agent, delay_s=self._delay_s,
                                     max_age_days=1)["filings"]["recent"]
            except Exception as exc:  # noqa: BLE001 — one company must not sink the rest
                self._failures.append(f"{ticker} submissions: {exc}")
                continue
            filings = []
            for i, form in enumerate(recent["form"]):
                if form != "4" or recent["filingDate"][i] < since:
                    continue
                try:
                    parsed = self._filing(cik, recent["accessionNumber"][i],
                                          recent["primaryDocument"][i])
                except Exception as exc:  # noqa: BLE001 — one filing must not sink the rest
                    self._failures.append(f"{ticker} {recent['accessionNumber'][i]}: {exc}")
                    continue
                filings.append({"filed": recent["filingDate"][i], **parsed})
            rows = observation_rows(cik, filings)
            records += rows
            log.info("Form 4 %s: %d filing(s) since %s, %d value(s).", ticker, len(filings),
                     since, len(rows), extra={"source": SOURCE})
        return self.validate(pd.DataFrame(records)) if records else empty_observations()

    def partial_failures(self) -> list[str]:
        return list(self._failures)
