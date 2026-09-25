"""Open-market insider purchases from the SEC's Form 3/4/5 data sets (CLAUDE.md §15.1.6).

Only for the validation study (``validation/insiders.py``): the user decided the signal is
not shown until it is validated. One zip a quarter (~13 MB) holds every Form 3/4/5 filed;
only the **purchases** survive, projected onto an allow-list of columns — the data sets
carry each insider's name and street address, and none of that is needed or kept. Each
quarter is cached as a small ``.csv.gz`` in ``.cache/sec/form345``; the zip is not kept.

What a purchase is (config ``insider_study``): an original Form 4 (amendments would repeat
it), transaction code **P** (open-market or private purchase), shares **acquired**, by an
owner who is a director or an officer. A "10 % owner" that is neither is nearly always a
fund, whose buying is portfolio management, not inside knowledge.

``filing_date`` is when the purchase became public, and is the date the study uses
(section 9.4); ``trans_date`` is when it happened, kept only to measure the gap.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from core.logging_setup import get_logger
from ingest.base import retry

log = get_logger(__name__)

COLUMNS = ["accession", "filing_date", "trans_date", "issuer_cik", "symbol", "owner_cik",
           "is_director", "is_officer", "title", "shares", "price", "value"]


def quarters(first: str, today: dt.date) -> list[str]:
    """``2017q1`` … the last complete quarter before ``today``."""
    year, q = int(first[:4]), int(first[-1])
    last_year, last_q = (today.year, (today.month - 1) // 3)
    if last_q == 0:
        last_year, last_q = last_year - 1, 4
    out = []
    while (year, q) <= (last_year, last_q):
        out.append(f"{year}q{q}")
        year, q = (year + 1, 1) if q == 4 else (year, q + 1)
    return out


def purchases(submission: pd.DataFrame, owners: pd.DataFrame,
              transactions: pd.DataFrame) -> pd.DataFrame:
    """The insider purchases in one quarter's three tables, as :data:`COLUMNS`."""
    sub = submission[submission["DOCUMENT_TYPE"] == "4"][
        ["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"]]
    trans = transactions[(transactions["TRANS_CODE"] == "P")
                         & (transactions["TRANS_ACQUIRED_DISP_CD"] == "A")][
        ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"]]
    relation = owners["RPTOWNER_RELATIONSHIP"].fillna("")
    own = owners.assign(is_director=relation.str.contains("Director"),
                        is_officer=relation.str.contains("Officer"))
    own = own[own["is_director"] | own["is_officer"]][
        ["ACCESSION_NUMBER", "RPTOWNERCIK", "is_director", "is_officer", "RPTOWNER_TITLE"]]
    # A filing with several owners (a joint filing) credits the purchase to each: each is a
    # distinct insider for the cluster count, which is what a joint filing says.
    merged = trans.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER")
    if merged.empty:
        return pd.DataFrame(columns=COLUMNS)
    shares = pd.to_numeric(merged["TRANS_SHARES"], errors="coerce")
    price = pd.to_numeric(merged["TRANS_PRICEPERSHARE"], errors="coerce")
    out = pd.DataFrame({
        "accession": merged["ACCESSION_NUMBER"],
        "filing_date": pd.to_datetime(merged["FILING_DATE"], format="%d-%b-%Y",
                                      errors="coerce").dt.date.astype(str),
        "trans_date": pd.to_datetime(merged["TRANS_DATE"], format="%d-%b-%Y",
                                     errors="coerce").dt.date.astype(str),
        "issuer_cik": merged["ISSUERCIK"].astype(str).str.zfill(10),
        "symbol": merged["ISSUERTRADINGSYMBOL"].fillna("").str.upper().str.strip(),
        "owner_cik": merged["RPTOWNERCIK"].astype(str).str.zfill(10),
        "is_director": merged["is_director"], "is_officer": merged["is_officer"],
        "title": merged["RPTOWNER_TITLE"].fillna(""),
        "shares": shares, "price": price, "value": shares * price,
    })
    return out[COLUMNS]


def load_quarter(quarter: str, url_template: str, user_agent: str,
                 cache_dir: Path) -> pd.DataFrame | None:
    """One quarter's purchases, from the cache or the SEC. ``None`` if the SEC has not
    published that quarter yet (404) — the caller decides whether that is expected."""
    path = cache_dir / f"{quarter}_purchases.csv.gz"
    if path.exists():
        return pd.read_csv(path, dtype={"issuer_cik": str, "owner_cik": str, "accession": str})

    def call() -> requests.Response:
        return requests.get(url_template.format(quarter=quarter),
                            headers={"User-Agent": user_agent}, timeout=180)

    response = retry(call, exceptions=(requests.RequestException,))
    if response.status_code == 404:
        return None
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))

    def table(name: str) -> pd.DataFrame:
        return pd.read_csv(archive.open(name), sep="\t", dtype=str, quoting=3,
                           on_bad_lines="warn")

    frame = purchases(table("SUBMISSION.tsv"), table("REPORTINGOWNER.tsv"),
                      table("NONDERIV_TRANS.tsv"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")
    log.info("Form 4 %s: %d insider purchases.", quarter, len(frame),
             extra={"source": "sec_form345"})
    return frame
