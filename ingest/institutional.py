"""Institutional holdings of the watched companies, from the SEC's Form 13F data sets (§15.5,
user request 2026-09-29: "how the whales move").

Every manager with more than 100 M USD in US equities files a 13F each quarter, 45 days after
it closes, listing its long positions. The SEC publishes all of them as one zip per filing
window (~100 MB: tables SUBMISSION, COVERPAGE, INFOTABLE). Verified on 2026-09-29:

- A window groups filings by **filing date**, not by quarter: it carries late filings of old
  quarters (a manager filed its 2001 reports in 2026) and **amendments**. So each file is
  reduced to the rows of the watched CUSIPs (``.cache/sec/form13f/{name}.csv.gz``; the zip is
  not kept) and the quarters are rebuilt across all files.
- **Amendments, in filing order per manager and quarter:** a ``RESTATEMENT`` replaces the
  holdings, a ``NEW HOLDINGS`` amendment adds to them. Every state is stored as a version with
  its filing date (point-in-time, section 9.6); a position a restatement removes becomes 0.
- Only common shares (``SH``) held outright: options (``PUTCALL``) and principal amounts are
  not ownership.

CUSIPs come from the IBKR security master when the position is held, else from the SEC's
fails-to-deliver files (CUSIP and symbol side by side); when both exist and disagree, the
company is reported and left out, never guessed (section 9.3).

fetch() -> observations, source ``sec_13f``, series ``{cik}:inst:{filer_cik}`` (shares held,
``ts`` = quarter end, ``ts_release`` = filing date); table ``filers`` (manager names).
"""

from __future__ import annotations

import datetime as dt
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from core.config import Settings
from core.logging_setup import get_logger
from ingest.base import Ingester, empty_observations, held_tickers, retry
from ingest.screen import cached_json
from ingest.sec_filings import normalize_ticker, ticker_map

log = get_logger(__name__)

SOURCE = "sec_13f"
PAGE_13F = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
PAGE_FTD = "https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data"
ROW_COLUMNS = ["accession", "filed", "period", "filer_cik", "manager", "amendment_type",
               "cusip", "shares"]
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def window_end(name: str) -> dt.date | None:
    """The last filing date a data set covers: ``2023q4_form13f.zip`` → 2023-12-31,
    ``01jun2026-31aug2026_form13f.zip`` → 2026-08-31."""
    quarter = re.match(r"(\d{4})q([1-4])_", name)
    if quarter:
        year, q = int(quarter.group(1)), int(quarter.group(2))
        return (pd.Timestamp(year=year, month=3 * q, day=1) + pd.offsets.MonthEnd(0)).date()
    span = re.match(r"\d{2}[a-z]{3}\d{4}-(\d{2})([a-z]{3})(\d{4})_", name)
    if span:
        return dt.date(int(span.group(3)), _MONTHS[span.group(2)], int(span.group(1)))
    return None


def _date(text: str) -> str:
    return pd.to_datetime(text, format="%d-%b-%Y", errors="coerce").date().isoformat()


def member(archive: zipfile.ZipFile, name: str) -> str:
    """The archive path of table ``name``: at the root in most windows, inside a folder in
    some (01jun2025-31aug2025, found on the first run)."""
    for path in archive.namelist():
        if path.split("/")[-1].upper() == name.upper():
            return path
    raise KeyError(f"{name} not in the 13F data set: {archive.namelist()[:5]}")


def reduce_zip(archive: zipfile.ZipFile, cusips: set[str], chunk: int = 500_000) -> pd.DataFrame:
    """The watched CUSIPs' common-share rows of one data set, as :data:`ROW_COLUMNS`.

    INFOTABLE is read in chunks (~400 MB uncompressed); the small tables whole.
    """
    parts = []
    for block in pd.read_csv(archive.open(member(archive, "INFOTABLE.tsv")), sep="\t",
                             dtype=str, quoting=3,
                             usecols=["ACCESSION_NUMBER", "CUSIP", "SSHPRNAMT",
                                      "SSHPRNAMTTYPE", "PUTCALL"], chunksize=chunk):
        block = block[block["CUSIP"].str.upper().isin(cusips)
                      & (block["SSHPRNAMTTYPE"] == "SH") & block["PUTCALL"].isna()]
        if len(block):
            parts.append(block)
    if not parts:
        return pd.DataFrame(columns=ROW_COLUMNS)
    info = pd.concat(parts)
    sub = pd.read_csv(archive.open(member(archive, "SUBMISSION.tsv")), sep="\t", dtype=str,
                      quoting=3)
    cover = pd.read_csv(archive.open(member(archive, "COVERPAGE.tsv")), sep="\t", dtype=str,
                        quoting=3,
                        usecols=["ACCESSION_NUMBER", "AMENDMENTTYPE", "FILINGMANAGER_NAME"])
    merged = info.merge(sub, on="ACCESSION_NUMBER").merge(cover, on="ACCESSION_NUMBER", how="left")
    out = pd.DataFrame({
        "accession": merged["ACCESSION_NUMBER"],
        "filed": merged["FILING_DATE"].map(_date),
        "period": merged["PERIODOFREPORT"].map(_date),
        "filer_cik": merged["CIK"].astype(str).str.zfill(10),
        "manager": merged["FILINGMANAGER_NAME"].fillna(""),
        "amendment_type": merged["AMENDMENTTYPE"].fillna("").str.upper(),
        "cusip": merged["CUSIP"].str.upper(),
        "shares": pd.to_numeric(merged["SSHPRNAMT"], errors="coerce"),
    })
    return out[ROW_COLUMNS]


def versions(rows: pd.DataFrame, cik_of: dict[str, str]) -> list[dict[str, Any]]:
    """Observation rows: each manager's shares of each company per quarter, one version per
    filing that changed them (see the module docstring for the amendment rule)."""
    out = []
    if rows.empty:
        return out
    per_filing = rows.groupby(["filer_cik", "period", "accession", "filed", "amendment_type",
                               "cusip"])["shares"].sum().reset_index()
    for (filer, period), group in per_filing.groupby(["filer_cik", "period"]):
        state: dict[str, float] = {}
        filings = group.drop_duplicates("accession").sort_values(["filed", "accession"])
        for filing in filings.itertuples():
            held = group[group["accession"] == filing.accession].set_index("cusip")["shares"]
            before = dict(state)
            if filing.amendment_type == "NEW HOLDINGS":
                for cusip, shares in held.items():
                    state[cusip] = state.get(cusip, 0.0) + float(shares)
            else:                          # an original report or a restatement
                state = {c: float(v) for c, v in held.items()}
            for cusip in set(state) | set(before):
                value = state.get(cusip, 0.0)
                if before.get(cusip) != value and cusip in cik_of:
                    out.append({"source": SOURCE, "series_id": f"{cik_of[cusip]}:inst:{filer}",
                                "ts": period, "ts_release": filing.filed, "value": value})
    return out


def ftd_cusips(pages: Iterable[pd.DataFrame], symbols: set[str]) -> dict[str, str]:
    """``{symbol: cusip}`` from fails-to-deliver files (``SYMBOL`` and ``CUSIP`` columns)."""
    out: dict[str, str] = {}
    for page in pages:
        found = page[page["SYMBOL"].isin(symbols)].drop_duplicates("SYMBOL")
        for symbol, cusip in zip(found["SYMBOL"], found["CUSIP"]):
            out.setdefault(str(symbol), str(cusip).upper())
    return out


class InstitutionalIngester(Ingester):
    """13F holdings of the researched and held companies, from the SEC data sets."""

    source = SOURCE

    def __init__(self, settings: Settings) -> None:
        sec = settings.source("sec")
        cfg = dict(settings.source("institutional"))
        self._user_agent = settings.secret(str(sec.get("user_agent_env", "SEC_USER_AGENT")))
        self._cache = Path(str(sec.get("cache_dir", ".cache/sec")))
        self._ticker_map_url = str(sec.get("ticker_map_url",
                                           "https://www.sec.gov/files/company_tickers.json"))
        self._delay_s = 1.0 / float(sec.get("rate_limit_rps", 8) or 8)
        self._years = int(cfg.get("years", 3))
        self._ftd_files = int(cfg.get("ftd_files", 4))
        self._companies = [(str(c["cik"]).zfill(10), str(c["ticker"]))
                           for c in settings.researched_companies if c.get("cik")]
        self._ibkr_cusips: dict[str, str] = {}
        self._failures: list[str] = []
        self._filers: list[dict[str, Any]] = []
        self.force = False

    @staticmethod
    def is_available(settings: Settings) -> bool:
        sec = settings.source("sec")
        agent = settings.secret(str(sec.get("user_agent_env", "SEC_USER_AGENT")))
        return bool(agent) and bool(settings.source("institutional"))

    def attach_database(self, conn: Any) -> None:
        """Held positions without a card, and the CUSIPs IBKR already knows."""
        try:
            rows = conn.execute("SELECT ticker, cusip FROM securities WHERE cusip IS NOT NULL"
                                ).fetchall()
            self._ibkr_cusips = {str(t): str(c).upper() for t, c in rows if c}
        except Exception as exc:  # noqa: BLE001 — no security master yet
            log.debug("No securities table: %s", exc)
        known = {normalize_ticker(t) for _, t in self._companies}
        missing = [t for t in held_tickers(conn) if normalize_ticker(t) not in known]
        if missing:
            mapping = ticker_map(cached_json(self._ticker_map_url,
                                             self._cache / "company_tickers.json",
                                             user_agent=self._user_agent,
                                             delay_s=self._delay_s))
            for ticker in missing:
                cik = mapping.get(normalize_ticker(ticker))
                if cik and cik not in {c for c, _ in self._companies}:
                    self._companies.append((cik, ticker))

    def _get(self, url: str, **kw: Any) -> requests.Response:
        def call() -> requests.Response:
            response = requests.get(url, headers={"User-Agent": self._user_agent},
                                    timeout=300, **kw)
            response.raise_for_status()
            return response
        return retry(call, exceptions=(requests.RequestException,))

    def _cusips(self) -> dict[str, str]:
        """``{cusip: company cik}`` — IBKR first, the FTD files for the rest, and a
        disagreement between the two leaves the company out."""
        symbols = {t for _, t in self._companies}
        memo = self._cache / "form13f" / "ftd_cusips.json"
        ftd: dict[str, str] = {}
        if memo.exists():
            saved = json.loads(memo.read_text(encoding="utf-8"))
            fresh = saved.get("fetched", "") >= (dt.date.today()
                                                 - dt.timedelta(days=30)).isoformat()
            if fresh and symbols <= set(saved.get("cusips", {})):
                ftd = saved["cusips"]
        if not ftd:
            ftd = self._ftd(symbols)
            memo.parent.mkdir(parents=True, exist_ok=True)
            memo.write_text(json.dumps({"fetched": dt.date.today().isoformat(),
                                        "cusips": ftd}), encoding="utf-8")
        out = {}
        for cik, ticker in self._companies:
            ibkr, sec = self._ibkr_cusips.get(ticker), ftd.get(ticker)
            if ibkr and sec and ibkr != sec:
                self._failures.append(f"{ticker}: CUSIP {ibkr} (IBKR) ≠ {sec} (SEC FTD); left out")
                continue
            cusip = ibkr or sec
            if cusip:
                out[cusip] = cik
            else:
                self._failures.append(f"{ticker}: no CUSIP found; left out of the 13F view")
        return out

    def _ftd(self, symbols: set[str]) -> dict[str, str]:
        page = self._get(PAGE_FTD).text
        links = sorted(set(re.findall(r'href="([^"]*cnsfails(\d{6}[ab])\.zip)"', page)),
                       key=lambda x: x[1])[-self._ftd_files:]
        pages = []
        for url, _ in links:
            archive = zipfile.ZipFile(io.BytesIO(self._get("https://www.sec.gov" + url).content))
            pages.append(pd.read_csv(archive.open(archive.namelist()[0]), sep="|", dtype=str,
                                     encoding="latin-1"))
        return ftd_cusips(pages, symbols)

    def fetch(self) -> pd.DataFrame:
        self._failures, self._filers = [], []
        cik_of = self._cusips()
        wanted = set(cik_of)
        since = dt.date.today() - dt.timedelta(days=365 * self._years + 120)
        page = self._get(PAGE_13F).text
        links = sorted({link for link in re.findall(r'href="([^"]*form13f\.zip)"', page)
                        if (end := window_end(link.split("/")[-1])) and end >= since},
                       key=lambda link: window_end(link.split("/")[-1]))
        # Nothing new since the last good run — the same SEC files and the same watched
        # companies — means nothing to recompute: the versions are already in the database
        # (upserts never delete). Rebuilding them anyway cost ~156 s every morning for a
        # figure that changes once a quarter (measured 2026-09-29).
        stamp = {"files": [link.split("/")[-1] for link in links], "cusips": sorted(wanted)}
        memo = self._cache / "form13f" / "last_run.json"
        if not self.force and memo.exists() and \
                json.loads(memo.read_text(encoding="utf-8")) == stamp:
            log.info("13F: no new file and no new company since the last run; skipped.",
                     extra={"source": SOURCE})
            return empty_observations()
        frames = []
        for link in links:
            name = link.split("/")[-1]
            path = self._cache / "form13f" / f"{name}.csv.gz"
            covered = path.with_suffix(".cusips")
            if path.exists() and covered.exists() and wanted <= set(
                    covered.read_text(encoding="utf-8").split()):
                frames.append(pd.read_csv(path, dtype=str))
                continue
            raw = self._cache / "form13f_raw" / name
            raw.parent.mkdir(parents=True, exist_ok=True)
            try:
                if not raw.exists():
                    with self._get("https://www.sec.gov" + link, stream=True) as response:
                        with open(raw, "wb") as handle:
                            for block in response.iter_content(1 << 20):
                                handle.write(block)
                reduced = reduce_zip(zipfile.ZipFile(raw), wanted)
            except Exception as exc:  # noqa: BLE001 — one window must not sink the rest
                log.exception("13F %s failed.", name, extra={"source": SOURCE})
                self._failures.append(f"13F {name}: {exc}")
                continue
            finally:
                raw.unlink(missing_ok=True)          # the zip is not kept
            path.parent.mkdir(parents=True, exist_ok=True)
            reduced.to_csv(path, index=False, compression="gzip")
            covered.write_text("\n".join(sorted(wanted)), encoding="utf-8")
            frames.append(reduced.astype(str))
            log.info("13F %s: %d row(s) of the watched companies.", name, len(reduced),
                     extra={"source": SOURCE})
        rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ROW_COLUMNS)
        rows = rows[rows["cusip"].isin(wanted)].assign(
            shares=lambda f: pd.to_numeric(f["shares"], errors="coerce"),
            amendment_type=lambda f: f["amendment_type"].fillna("").replace("nan", ""))
        self._filers = [{"cik": c, "name": n} for c, n in
                        rows.drop_duplicates("filer_cik", keep="last")[["filer_cik", "manager"]]
                        .itertuples(index=False)]
        records = versions(rows, cik_of)
        log.info("13F: %d version(s) of %d manager(s) across %d file(s).", len(records),
                 len(self._filers), len(frames), extra={"source": SOURCE})
        if not self._failures:
            memo.parent.mkdir(parents=True, exist_ok=True)
            memo.write_text(json.dumps(stamp), encoding="utf-8")
        return self.validate(pd.DataFrame(records)) if records else empty_observations()

    def fetch_tables(self) -> dict[str, list[dict[str, Any]]]:
        return {"filers": self._filers} if self._filers else {}

    def partial_failures(self) -> list[str]:
        return list(self._failures)
