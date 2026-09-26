"""companyfacts of the S&P 500 members, reduced to what a study needs (CLAUDE.md §15.4.9).

The factor study asks what ~590 companies had filed on each past date. SEC ``frames`` — the
screen's source — cannot answer that: it carries no filing date and returns the **last**
filed version of each period, restatements included. Fine for "what are the numbers now";
look-ahead for a backtest (sections 9.4, 9.6). ``companyfacts`` carries ``filed`` on every
fact, which is the point-in-time the study needs.

It is heavy (Apple: 265 KB compressed, ~5 MB expanded), so only the concepts asked for are
kept, compressed, in ``.cache/sec/member_facts/`` — **never in the database**: these are the
inputs of a study, not panel series, and they would multiply the database's size.

Kept indefinitely: a filed fact is never un-filed, and the study only asks about the past.
A file is downloaded again when the concepts asked for are not all in it, or with
``refresh=True`` (to extend the sample with newer filings).

Each company is one request, at the SEC's rate limit; a failure is reported and that company
left out of the study, never guessed (section 12).
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

from core.logging_setup import get_logger
from ingest.base import retry
from ingest.sec_xbrl import COMPANYFACTS_URL, TAXONOMY

log = get_logger(__name__)

SOURCE = "sec"


def subset(document: Mapping[str, Any], tags: Iterable[str]) -> dict[str, Any]:
    """``{"us-gaap": {tag: concept}}`` for the tags present — the shape
    :func:`ingest.sec_xbrl.extract_metric` reads."""
    concepts = (document.get("facts") or {}).get(TAXONOMY, {})
    return {TAXONOMY: {tag: concepts[tag] for tag in tags if tag in concepts}}


def load(cik: str, tags: Iterable[str], cache_dir: Path, *, base_url: str, user_agent: str,
         delay_s: float, refresh: bool = False) -> dict[str, Any]:
    """The company's facts for ``tags``, from the cache or the SEC.

    A company with no XBRL facts (404) caches an empty subset: asking again would give the
    same answer. Raises on any other failure — the caller counts it.
    """
    wanted = sorted(set(tags))
    path = cache_dir / f"{cik}.json.gz"
    if path.exists() and not refresh:
        stored = json.loads(gzip.decompress(path.read_bytes()))
        if set(wanted) <= set(stored.get("tags", [])):
            return stored["facts"]
    url = COMPANYFACTS_URL.format(base=base_url, cik=cik)

    def call() -> requests.Response:
        return requests.get(url, headers={"User-Agent": user_agent,
                                          "Accept-Encoding": "gzip"}, timeout=120)

    response = retry(call, exceptions=(requests.RequestException,))
    time.sleep(delay_s)
    if response.status_code == 404:
        facts: dict[str, Any] = {TAXONOMY: {}}
    else:
        response.raise_for_status()
        facts = subset(response.json(), wanted)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps({"tags": wanted, "facts": facts}).encode()))
    return facts


def load_all(ciks: Iterable[str], tags: Iterable[str], cache_dir: Path, *, base_url: str,
             user_agent: str, delay_s: float, refresh: bool = False
             ) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """``({cik: facts}, failures)`` for every CIK; progress logged every 100."""
    out: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    ciks = sorted(set(ciks))
    for i, cik in enumerate(ciks, 1):
        try:
            out[cik] = load(cik, tags, cache_dir, base_url=base_url, user_agent=user_agent,
                            delay_s=delay_s, refresh=refresh)
        except Exception as exc:  # noqa: BLE001 — one company must not sink the study
            failures.append(f"{cik}: {exc}")
        if i % 100 == 0:
            log.info("companyfacts: %d / %d.", i, len(ciks), extra={"source": SOURCE})
    return out, failures
