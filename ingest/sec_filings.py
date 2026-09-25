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

**Held positions are fetched too**, even without a card in config. What is held is read
from the account (section 5.1), and a position is exactly where an earnings date or an
amended 10-Q matters most — the phase-4 alerts look at positions. The ticker is resolved to
a CIK through the SEC's own ``company_tickers.json`` (section 4.2); a ticker that does not
resolve (an ETF, typically) stays unresolved with a warning, never guessed (section 9.3).

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
from ingest.base import Ingester, empty_observations, held_tickers, retry
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
# No quarterly report follows the previous one this soon; an announcement closer than this
# is an off-cycle item 2.02 (preliminary figures, a conference), not the next quarter.
MIN_REPORT_GAP = dt.timedelta(days=45)
QUARTERS_PER_YEAR = 4


def normalize_ticker(ticker: str) -> str:
    """One spelling for a ticker across sources: the SEC writes ``BRK-B``, IBKR ``BRK B``."""
    return str(ticker).strip().upper().replace(".", "-").replace(" ", "-")


def ticker_map(payload: Mapping[str, Any]) -> dict[str, str]:
    """``{TICKER: ten-digit CIK}`` from the SEC's ``company_tickers.json``.

    The file maps *today's* tickers, which is exactly what a current holding carries.
    It would be the wrong tool for a historical ticker (section 9.3): tickers are reused.
    """
    out: dict[str, str] = {}
    for row in payload.values():
        out.setdefault(normalize_ticker(row["ticker"]), str(int(row["cik_str"])).zfill(10))
    return out


def company_row(
    submissions: Mapping[str, Any], cik: str, ticker: str, card: Mapping[str, Any] | None,
    today: str,
) -> dict[str, Any]:
    """A ``companies`` row: the CIK with the ticker it is known by today.

    The registry is what lets any later reader go from a held ticker to its CIK without
    guessing (section 9.3). Sector and thesis category come only from a written card: the
    SEC's SIC code is not GICS, and copying it into ``sector`` would pass one off as the
    other — it has its own column, used to find a company's peers.
    """
    card = card or {}
    return {
        "cik": cik,
        "ticker": ticker,
        "name": submissions.get("name"),
        "sector": card.get("sector"),
        "thesis_category": card.get("thesis_category"),
        "first_seen": today,
        "status": "active",
        "sic": str(submissions.get("sic") or "") or None,
        "sic_description": submissions.get("sicDescription") or None,
    }


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


# A foreign issuer (20-F / 6-K) files no 8-K. Its results come as SEVERAL 6-Ks on one day —
# press release, financial statements, presentation — all for the quarter just closed
# (Nu Holdings: 3-4 on each results day, 2025-2026). That pattern is the fact the SEC records.
FOREIGN_MIN_REPORTS = 3
FOREIGN_LAG_DAYS = (15, 100)     # a results filing lands weeks after the quarter, not years


def foreign_earnings_dates(submissions: Mapping[str, Any]) -> list[dict[str, str]]:
    """Results days of a foreign issuer, newest first: ``[{date, accession}]``.

    A day with at least ``FOREIGN_MIN_REPORTS`` 6-Ks whose report date is the same quarter
    end, filed ``FOREIGN_LAG_DAYS`` after it. Inferred from a pattern, not read from an item
    code — the event's payload says so.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    groups: dict[tuple[str, str], list[str]] = {}
    for index, form in enumerate(recent.get("form", [])):
        if (form or "").upper() != "6-K":
            continue
        filed = recent["filingDate"][index]
        period = (recent.get("reportDate") or [""] * (index + 1))[index] or ""
        if len(period) != 10 or period[5:] not in ("03-31", "06-30", "09-30", "12-31"):
            continue
        lag = (dt.date.fromisoformat(filed) - dt.date.fromisoformat(period)).days
        if FOREIGN_LAG_DAYS[0] <= lag <= FOREIGN_LAG_DAYS[1]:
            groups.setdefault((filed, period), []).append(recent["accessionNumber"][index])
    out = [{"date": filed, "accession": accessions[0]}
           for (filed, _period), accessions in groups.items()
           if len(accessions) >= FOREIGN_MIN_REPORTS]
    return sorted(out, key=lambda row: row["date"], reverse=True)


def estimate_next_earnings(
    announcements: Sequence[Mapping[str, str]]
) -> tuple[str, str] | None:
    """Estimate the next earnings date from the company's own cadence.

    Returns ``(iso_date, method)``, or ``None`` when there is not enough history to say
    anything — in which case the panel shows no date at all rather than a guess dressed as
    a schedule (section 12).

    Preferred method: the earliest anniversary (+364 days) of last year's announcements that
    is at least ``MIN_REPORT_GAP`` after the latest one — robust to an off-cycle item 2.02,
    which "four announcements back" was not. Companies report
    on a stable calendar, and 52 weeks preserves the weekday where "+1 year" drifts into
    the weekend. With fewer than five announcements it falls back to the last date plus the
    median gap, which is coarser and is labelled as such.
    """
    if len(announcements) < 2:
        return None
    dates = [dt.date.fromisoformat(row["date"]) for row in announcements]  # newest first

    if len(dates) > QUARTERS_PER_YEAR:
        # The anniversary of each announcement of the last year, and the earliest one that
        # can still be the next report. Counting four announcements back instead broke on an
        # off-cycle item 2.02 — preliminary results at a January conference (Duolingo
        # 2026-01-12; ImmunityBio three in a year): it counted as a quarter and the estimate
        # skipped November for January (2026-09-25).
        last = dates[0]
        anniversaries = sorted(d + YEAR_OF_WEEKS for d in dates
                               if last - d < YEAR_OF_WEEKS + dt.timedelta(days=30))
        upcoming = [d for d in anniversaries if d > last + MIN_REPORT_GAP]
        if upcoming:
            return upcoming[0].isoformat(), "misma fecha del año anterior + 364 días"

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
    source = "8-K item 2.02"
    if not announcements:
        announcements = foreign_earnings_dates(submissions)
        source = f"6-K: {FOREIGN_MIN_REPORTS}+ el mismo día sobre el trimestre (inferido)"
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
                                   "source": source}),
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
        self._cards = {
            str(card["cik"]).zfill(10): card
            for card in settings.researched_companies if card.get("cik")
        }
        self._ticker_map_url = str(
            cfg.get("ticker_map_url", "https://www.sec.gov/files/company_tickers.json")
        )
        self._tables: dict[str, list[dict[str, Any]]] = {}
        self._failures: list[str] = []

    @staticmethod
    def is_available(settings: Settings) -> bool:
        """Same prerequisites as the XBRL ingester: a real User-Agent and a CIK."""
        from ingest.sec_xbrl import SecXbrlIngester  # noqa: PLC0415

        return SecXbrlIngester.is_available(settings)

    def _download(self, cik: str) -> dict[str, Any]:
        """Submissions document, from cache when fresh, from EDGAR otherwise."""
        return self._cached_json(
            self._cache_dir / f"submissions_CIK{cik}.json",
            SUBMISSIONS_URL.format(base=self._base_url, cik=cik),
            f"CIK {cik}",
        )

    def _cached_json(self, path: Path, url: str, label: str) -> dict[str, Any]:
        """A SEC JSON document: fresh cache, else download, else the expired copy (warned)."""
        if path.exists():
            age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
            if age < self._cache_ttl:
                return json.loads(path.read_text(encoding="utf-8"))

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
                self._failures.append(f"{label}: using expired cache ({exc})")
                log.warning("%s: download failed (%s); using the expired cache copy.",
                            label, exc, extra={"source": SOURCE})
                return json.loads(path.read_text(encoding="utf-8"))
            raise
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def attach_database(self, conn: Any) -> None:
        """Add the held positions that no card in config names (see the module docstring)."""
        known = {normalize_ticker(ticker) for _, ticker in self._companies}
        missing = [t for t in held_tickers(conn) if normalize_ticker(t) not in known]
        if not missing:
            return
        try:
            mapping = ticker_map(self._cached_json(
                self._cache_dir / "company_tickers.json", self._ticker_map_url,
                "company_tickers.json",
            ))
        except Exception as exc:
            # Without the map the held positions get no calendar, and the earnings alert
            # would go quiet on them. That must reach the exit code, not just the log.
            log.exception("Could not load the SEC ticker map.", extra={"source": SOURCE})
            self._failures.append(f"company_tickers.json: {exc}")
            return
        known_ciks = {cik for cik, _ in self._companies}
        added, unresolved = [], []
        for ticker in missing:
            cik = mapping.get(normalize_ticker(ticker))
            if cik is None:
                unresolved.append(ticker)
            elif cik not in known_ciks:
                self._companies.append((cik, ticker))
                known_ciks.add(cik)
                added.append(ticker)
        if added:
            log.info("Adding %d held position(s) to the filings download: %s",
                     len(added), added, extra={"source": SOURCE})
        if unresolved:
            # An ETF has no company CIK, and that is not a failure. Guessing one would be
            # (section 9.3), so it is only said.
            log.warning("Held ticker(s) without a SEC company CIK, left unresolved: %s",
                        unresolved, extra={"source": SOURCE})

    def fetch(self) -> pd.DataFrame:
        """Collect filings and events. Nothing here is a time series, so no observations."""
        filings: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        companies: list[dict[str, Any]] = []
        today = dt.date.today().isoformat()

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
            companies.append(company_row(submissions, cik, ticker, self._cards.get(cik), today))

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

        self._tables = {"filings": filings, "events": events, "companies": companies}
        return empty_observations()

    def fetch_tables(self) -> dict[str, list[dict[str, Any]]]:
        return {table: rows for table, rows in self._tables.items() if rows}

    def partial_failures(self) -> list[str]:
        return list(self._failures)
