"""SEC submissions: the filing history, the amendment flag and the earnings calendar.

Two things come out of the same document, which is why they live in one module.

**The filing history** (section 8, phase 2 point 3) feeds the restatement flag. A ``10-K/A``
or ``10-Q/A`` means the company re-filed financial statements it had already published —
that is a governance red flag, not a technical detail (section 9.6). Note that most ``/A``
forms are *not* that: a ``4/A`` corrects an insider-transaction report and says nothing
about the accounts, so :func:`restatement_filings` narrows to the financial forms.

**The earnings calendar** (point 6) is the analogue of a token unlock: a binary event on a
date everyone knows, and section 2 makes checking it mandatory before opening a position.
There is no free feed of *future* earnings dates, so this module builds one honestly:

- **Past dates are facts.** A company announces results through an ``8-K`` carrying item
  **2.02** ("Results of Operations and Financial Condition"). The SEC records the date, so
  those rows are stored with ``is_estimated = 0``.
- **The next date is an estimate, and says so.** Companies report on a stable calendar, so
  the estimate is the announcement four quarters back plus 364 days — 52 weeks, which
  preserves the weekday, because results land on a Tuesday-to-Thursday and a plain
  "+1 year" would drift across the weekend. The method used is written into the event's
  payload so the panel can show it and nobody has to trust a bare date.

Estimated is never rendered as confirmed (section 12): ``is_estimated`` exists in the
schema precisely so the difference survives all the way to the screen.

fetch() -> empty observations; nothing here is a time series.
fetch_tables() -> {"filings": [...], "events": [...]}
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import requests

from core.config import Settings
from core.logging_setup import get_logger
from ingest.base import Ingester, empty_observations, retry
from ingest.sec_xbrl import SecRequestError

log = get_logger(__name__)

SOURCE = "sec"
SUBMISSIONS_URL = "{base}/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

# Item 2.02 of an 8-K is "Results of Operations and Financial Condition" — the earnings
# announcement itself.
EARNINGS_ITEM = "2.02"

# Forms whose amendment means the financial statements were re-filed (section 9.6). A
# `4/A` is an insider-transaction correction and does not belong here.
FINANCIAL_FORMS = frozenset({"10-K", "10-Q", "20-F", "40-F", "6-K"})

# 52 weeks. Preserves the weekday, which "+1 year" does not — and results land midweek.
YEAR_OF_WEEKS = dt.timedelta(days=364)
QUARTERS_PER_YEAR = 4


def _archive_url(cik: str, accession: str, document: str) -> str | None:
    """Public URL of a filing's primary document."""
    if not document:
        return None
    return ARCHIVE_URL.format(
        cik=int(cik), accession=accession.replace("-", ""), document=document
    )


def filing_rows(
    submissions: Mapping[str, Any], cik: str, keep_forms: Sequence[str]
) -> list[dict[str, Any]]:
    """Rows for the ``filings`` table, restricted to the configured forms.

    Every form is kept in its amended and unamended shape: passing ``10-K`` keeps ``10-K/A``
    too, because the amendment is the interesting one.

    ⚠️ ``submissions`` serves the most recent ~1000 filings inline and older ones in
    separate files. Only the inline block is read, which for a company that files hundreds
    of Form 4s a year can be a shorter window than it looks. The count is logged rather
    than assumed away.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    accessions = recent.get("accessionNumber", [])
    wanted = {form.upper() for form in keep_forms}
    rows: list[dict[str, Any]] = []

    for index, accession in enumerate(accessions):
        form = (recent["form"][index] or "").upper()
        base_form = form[:-2] if form.endswith("/A") else form
        if base_form not in wanted:
            continue
        rows.append({
            "accession": accession,
            "cik": cik,
            "form": form,
            "period_end": recent["reportDate"][index] or None,
            "filed_date": recent["filingDate"][index],
            "is_amended": 1 if form.endswith("/A") else 0,
            "url": _archive_url(cik, accession, recent["primaryDocument"][index]),
        })
    return rows


def restatement_filings(filings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The amendments that actually mean re-filed financial statements (section 9.6).

    A governance red flag, and the reason the schema keeps every version of a fact: the
    panel shows the latest, a backtest uses the one in force on its date.
    """
    return [
        filing for filing in filings
        if filing["is_amended"] and filing["form"][:-2].upper() in FINANCIAL_FORMS
    ]


def earnings_dates(submissions: Mapping[str, Any]) -> list[dict[str, str]]:
    """Confirmed earnings announcements, newest first: ``[{date, accession}]``.

    Read from 8-K filings carrying item 2.02. These are facts with a date the SEC recorded,
    not a guess and not a scrape (section 4.3).
    """
    recent = submissions.get("filings", {}).get("recent", {})
    out: list[dict[str, str]] = []
    for index, form in enumerate(recent.get("form", [])):
        if (form or "").upper() != "8-K":
            continue
        items = recent.get("items", [])[index] or ""
        if EARNINGS_ITEM not in [item.strip() for item in items.split(",")]:
            continue
        out.append({
            "date": recent["filingDate"][index],
            "accession": recent["accessionNumber"][index],
        })
    return sorted(out, key=lambda row: row["date"], reverse=True)


def estimate_next_earnings(
    announcements: Sequence[Mapping[str, str]]
) -> tuple[str, str] | None:
    """Estimate the next earnings date from the company's own cadence.

    Returns ``(iso_date, method)``, or ``None`` when there is not enough history to say
    anything — in which case the panel shows no date at all rather than a guess dressed as
    a schedule (section 12).

    Preferred method: the announcement four quarters back plus 364 days. Companies report
    on a stable calendar, and 52 weeks preserves the weekday where "+1 year" drifts into
    the weekend. With fewer than five announcements it falls back to the last date plus the
    median gap, which is coarser and is labelled as such.
    """
    if len(announcements) < 2:
        return None
    dates = [dt.date.fromisoformat(row["date"]) for row in announcements]  # newest first

    if len(dates) > QUARTERS_PER_YEAR:
        anchor = dates[QUARTERS_PER_YEAR - 1]  # same fiscal quarter, one year earlier
        return (anchor + YEAR_OF_WEEKS).isoformat(), "misma fecha del año anterior + 364 días"

    gaps = [(dates[i] - dates[i + 1]).days for i in range(len(dates) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2]
    return (dates[0] + dt.timedelta(days=median_gap)).isoformat(), (
        f"última fecha + {median_gap} días (mediana de {len(gaps)} intervalos)"
    )


def event_rows(
    submissions: Mapping[str, Any], cik: str, ticker: str, history: int = 8
) -> list[dict[str, Any]]:
    """Earnings events for the ``events`` table: the recent confirmed ones plus the next.

    The estimate is written under a fixed ``event_id`` so that each run moves it forward
    instead of leaving a trail of stale predictions.
    """
    announcements = earnings_dates(submissions)
    if not announcements:
        return []

    rows: list[dict[str, Any]] = []
    for announcement in announcements[:history]:
        rows.append({
            "event_id": f"{cik}:earnings:{announcement['date']}",
            "category": "earnings",
            "cik": cik,
            "ts": announcement["date"],
            "is_estimated": 0,
            "label": f"{ticker}: resultados publicados",
            "payload": json.dumps({"accession": announcement["accession"],
                                   "source": "8-K item 2.02"}),
        })

    estimate = estimate_next_earnings(announcements)
    if estimate:
        date, method = estimate
        rows.append({
            "event_id": f"{cik}:earnings:next",
            "category": "earnings",
            "cik": cik,
            "ts": date,
            "is_estimated": 1,
            "label": f"{ticker}: próximos resultados (estimado)",
            # The method travels with the date: a panel that shows an estimated date
            # without saying how it was produced is asking to be trusted blindly.
            "payload": json.dumps({"method": method,
                                   "last_confirmed": announcements[0]["date"]}),
        })
    return rows


class SecFilingsIngester(Ingester):
    """Filing history, amendment flags and the earnings calendar, per tracked company."""

    source = SOURCE

    def __init__(self, settings: Settings) -> None:
        cfg = settings.source("sec")
        self._base_url = str(cfg.get("base_url", "https://data.sec.gov")).rstrip("/")
        self._user_agent = settings.secret(str(cfg.get("user_agent_env", "SEC_USER_AGENT")))
        self._keep_forms = list(cfg.get("filing_forms", ["10-K", "10-Q", "8-K"]))
        self._history = int(cfg.get("earnings_history", 8))
        self._cache_dir = Path(str(cfg.get("cache_dir", ".cache/sec")))
        self._cache_ttl = dt.timedelta(hours=float(cfg.get("cache_ttl_hours", 24)))
        retry_cfg = settings.raw.get("ingest", {}).get("retry", {})
        self._retry_kwargs = {
            "attempts": retry_cfg.get("attempts", 4),
            "base_delay_s": retry_cfg.get("base_delay_s", 1.0),
            "max_delay_s": retry_cfg.get("max_delay_s", 30.0),
        }
        from ingest.sec_xbrl import SecXbrlIngester  # noqa: PLC0415 — shared company list

        self._companies = SecXbrlIngester.companies_for(settings)
        self._tables: dict[str, list[dict[str, Any]]] = {}
        self._failures: list[str] = []

    @staticmethod
    def is_available(settings: Settings) -> bool:
        """Same prerequisites as the XBRL ingester: a real User-Agent and a CIK."""
        from ingest.sec_xbrl import SecXbrlIngester  # noqa: PLC0415

        return SecXbrlIngester.is_available(settings)

    def _download(self, cik: str) -> dict[str, Any]:
        """Submissions document, from cache when fresh, from EDGAR otherwise."""
        path = self._cache_dir / f"submissions_CIK{cik}.json"
        if path.exists():
            age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
            if age < self._cache_ttl:
                return json.loads(path.read_text(encoding="utf-8"))

        url = SUBMISSIONS_URL.format(base=self._base_url, cik=cik)

        def call() -> dict[str, Any]:
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
                log.warning("CIK %s: download failed (%s); using the expired cache copy.",
                            cik, exc, extra={"source": SOURCE})
                return json.loads(path.read_text(encoding="utf-8"))
            raise
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fetch(self) -> pd.DataFrame:
        """Collect filings and events. Nothing here is a time series, so no observations."""
        filings: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []

        for cik, ticker in self._companies:
            try:
                submissions = self._download(cik)
            except Exception as exc:
                log.exception("CIK %s (%s) failed.", cik, ticker, extra={"source": SOURCE})
                self._failures.append(f"CIK {cik} ({ticker}): {exc}")
                continue

            company_filings = filing_rows(submissions, cik, self._keep_forms)
            company_events = event_rows(submissions, cik, ticker, self._history)
            filings.extend(company_filings)
            events.extend(company_events)

            restatements = restatement_filings(company_filings)
            if restatements:
                # Not a technical detail: the company re-filed statements it had already
                # published (section 9.6). The panel raises this as a red flag.
                log.warning(
                    "CIK %s (%s): %d amended financial filing(s) — possible restatement: %s",
                    cik, ticker, len(restatements),
                    ", ".join(f"{r['form']} {r['filed_date']}" for r in restatements[:5]),
                    extra={"source": SOURCE},
                )
            estimated = [e for e in company_events if e["is_estimated"]]
            log.info(
                "CIK %s (%s): %d filings, %d earnings dates%s.",
                cik, ticker, len(company_filings),
                len(company_events) - len(estimated),
                f", next estimated {estimated[0]['ts']}" if estimated else "",
                extra={"source": SOURCE},
            )

        self._tables = {"filings": filings, "events": events}
        return empty_observations()

    def fetch_tables(self) -> dict[str, list[dict[str, Any]]]:
        return {table: rows for table, rows in self._tables.items() if rows}

    def partial_failures(self) -> list[str]:
        return list(self._failures)
