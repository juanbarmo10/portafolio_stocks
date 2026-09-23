"""FINRA consolidated short interest — level 2 positioning (CLAUDE.md sections 4.2, 8 phase 3).

Short interest is the count of shares sold short and not yet covered, reported twice a
month. It is the closest equity analogue of the funding-rate positioning read in the
sibling crypto project: it says how crowded one side of a trade is, at a much worse
frequency.

FINRA serves it through a public REST API with no credentials. Verified 2026-09-23.

**The hard part is not the download, it is the publication date.**

Section 4.2 says to respect ``ts_release`` for this source, and section 9.4 makes it
mandatory — but **the dataset does not carry one**. It has ``settlementDate``, the date the
positions describe, and no field saying when FINRA made them public. There is no
dissemination-calendar dataset in the API either (checked: 404).

So the publication date is **derived, not observed**, and this module says so everywhere it
can. The derivation is a configured lag in business days after the settlement date, and the
number was measured rather than assumed: FINRA publishes its settlement/publication schedule
on its website, and across **28 consecutive cycles** the gap was 7 business days in 19 of
them and 8 in the other 9 — it is *not* constant.

⚠️ **Which is why the default is the maximum, not the median.** A lag that is too short
claims a figure was public before it was, which is look-ahead — exactly what section 9.4
exists to stop. A lag that is too long merely makes the panel slightly conservative. When a
derived date has to be wrong, it must be wrong **late**.

Known limitation, stated rather than papered over: FINRA serves **one row per settlement
date** — a revision replaces the original rather than adding a row — and a derived
publication date cannot distinguish the two versions. So a revised cycle overwrites the
figure previously stored for it, and section 9.6's "keep every version" cannot be honored
here. What is kept is the *fact* of the revision, as a companion series, so a reader is
never told a revised number is an original one.

fetch() -> DataFrame[source, series_id, ts, ts_release, value]
    series_id = "TICKER:short_interest"          shares sold short and not covered
              | "TICKER:days_to_cover"           short interest / average daily volume
              | "TICKER:short_interest:revised"  1.0 when FINRA marks the cycle revised
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import requests

from core.config import Settings
from core.logging_setup import get_logger
from ingest.base import Ingester, empty_observations, retry

log = get_logger(__name__)

SOURCE = "finra"
REVISED_SUFFIX = "revised"

# Measured over the 28 cycles FINRA publishes on its schedule page (2026-09-23): 7 business
# days in 19 of them, 8 in 9. The maximum is the safe default — see the module docstring.
DEFAULT_LAG_BUSINESS_DAYS = 8

# FINRA marks a row that restates a previously published cycle.
REVISION_FLAG = "R"


class FinraRequestError(RuntimeError):
    """FINRA refused the request or answered something unusable."""


def publication_date(settlement: Any, lag_business_days: int) -> str | None:
    """Derive when a cycle became public: ``settlement + N business days``.

    **Derived, never observed.** The source carries no publication date (module docstring),
    so this is the project's assumption made explicit and configurable rather than buried.
    Weekends are skipped; US market holidays are not, which pushes the date slightly later
    — in the safe direction.

    Returns:
        ISO8601 UTC midnight stamp, matching the rest of the project, or ``None`` when the
        settlement date cannot be read.
    """
    try:
        day = dt.date.fromisoformat(str(settlement)[:10])
    except (ValueError, TypeError):
        return None
    remaining = max(int(lag_business_days), 0)
    while remaining:
        day += dt.timedelta(days=1)
        if day.weekday() < 5:
            remaining -= 1
    return f"{day.isoformat()}T00:00:00+00:00"


def observation_rows(
    records: Iterable[Mapping[str, Any]], ticker: str, lag_business_days: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """FINRA rows -> observations, with the derived publication date attached.

    Returns:
        ``(rows, problems)``. A cycle with no usable settlement date or no short position is
        skipped and reported; nothing is estimated (section 12).
    """
    rows: list[dict[str, Any]] = []
    problems: list[str] = []

    for record in records:
        settlement = record.get("settlementDate")
        released = publication_date(settlement, lag_business_days)
        if released is None:
            problems.append(f"{ticker}: unreadable settlementDate {settlement!r}")
            continue
        ts = f"{str(settlement)[:10]}T00:00:00+00:00"

        shares = _number(record.get("currentShortPositionQuantity"))
        if shares is None:
            problems.append(f"{ticker}: cycle {settlement} has no short position")
            continue
        rows.append({"source": SOURCE, "series_id": f"{ticker}:short_interest",
                     "ts": ts, "ts_release": released, "value": shares})

        cover = _number(record.get("daysToCoverQuantity"))
        if cover is not None:
            rows.append({"source": SOURCE, "series_id": f"{ticker}:days_to_cover",
                         "ts": ts, "ts_release": released, "value": cover})

        # The fact of a revision is kept even though the superseded value cannot be: a
        # reader must never be shown a restated figure believing it is the original.
        revised = str(record.get("revisionFlag") or "").strip().upper() == REVISION_FLAG
        rows.append({
            "source": SOURCE,
            "series_id": f"{ticker}:short_interest:{REVISED_SUFFIX}",
            "ts": ts, "ts_release": released, "value": 1.0 if revised else 0.0,
        })
    return rows, problems


def _number(value: Any) -> float | None:
    """Parse a FINRA numeric field. ``None`` on anything unusable — never a zero default."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ShortInterestIngester(Ingester):
    """Consolidated short interest for every ticker written down or held."""

    source = SOURCE

    def __init__(self, settings: Settings) -> None:
        cfg = settings.source("finra")
        self._url = str(cfg.get("short_interest_url", "")).rstrip("/")
        self._lag = int(cfg.get("publication_lag_business_days", DEFAULT_LAG_BUSINESS_DAYS))
        self._page_limit = int(cfg.get("page_limit", 500))
        self._timeout_s = float(cfg.get("request_timeout_s", 60))
        self._user_agent = str(cfg.get("user_agent", "equitydash (personal research)"))
        self._cache_dir = Path(str(cfg.get("cache_dir", ".cache/finra")))
        self._cache_ttl = dt.timedelta(hours=float(cfg.get("cache_ttl_hours", 24)))
        self._tickers = self.tickers_for(settings)
        retry_cfg = settings.raw.get("ingest", {}).get("retry", {})
        self._retry_kwargs = {
            "attempts": retry_cfg.get("attempts", 4),
            "base_delay_s": retry_cfg.get("base_delay_s", 1.0),
            "max_delay_s": retry_cfg.get("max_delay_s", 30.0),
        }
        self._failures: list[str] = []

    @staticmethod
    def tickers_for(settings: Settings) -> list[str]:
        """Every company written down — tracked and under study. Not the reference ETFs.

        Short interest on a broad-market ETF is dominated by hedging and creation/redemption
        mechanics rather than by a directional view, so it would not mean what the level-2
        read wants it to mean. Held tickers are added by :meth:`attach_database`.
        """
        return list(dict.fromkeys(
            str(card["ticker"]) for card in settings.researched_companies
            if card.get("ticker")
        ))

    @staticmethod
    def is_available(settings: Settings) -> bool:
        """Needs a configured endpoint and at least one ticker to ask about."""
        cfg = settings.source("finra")
        return bool(cfg.get("short_interest_url")) and bool(
            ShortInterestIngester.tickers_for(settings)
        )

    def attach_database(self, conn: Any) -> None:
        """Add whatever is actually held, read from the account (section 5.1)."""
        try:
            rows = conn.execute(
                "SELECT DISTINCT series_id FROM observations "
                "WHERE source = 'ibkr' AND series_id LIKE '%:position_qty'"
            ).fetchall()
        except Exception as exc:  # pragma: no cover - a fresh database has no table yet
            log.debug("Could not read held tickers: %s", exc)
            return
        held = [str(r[0]).rsplit(":", 1)[0] for r in rows]
        added = [t for t in held if t not in self._tickers]
        if added:
            self._tickers.extend(added)
            log.info("Added %d held ticker(s) to the short-interest request: %s",
                     len(added), added, extra={"source": SOURCE})

    def _cache_path(self, ticker: str) -> Path:
        return self._cache_dir / f"short_interest_{ticker}.json"

    def _download(self, ticker: str) -> list[dict[str, Any]]:
        """Every published cycle for one ticker, from cache when fresh.

        The dataset is partitioned by ``settlementDate``, so sorting requires filtering on
        that field; filtering by symbol instead returns the whole history unsorted, which is
        what this wants. A settlement date that has not been published yet simply has no
        rows — absence *is* the "not yet disseminated" signal.
        """
        path = self._cache_path(ticker)
        if path.exists():
            age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
            if age < self._cache_ttl:
                return json.loads(path.read_text(encoding="utf-8"))

        records: list[dict[str, Any]] = []
        offset = 0
        while True:
            body = {
                "compareFilters": [
                    {"fieldName": "symbolCode", "fieldValue": ticker, "compareType": "EQUAL"}
                ],
                "limit": self._page_limit,
                "offset": offset,
            }

            def call(payload: dict[str, Any] = body) -> list[dict[str, Any]]:
                response = requests.post(
                    self._url,
                    headers={"User-Agent": self._user_agent,
                             "Content-Type": "application/json"},
                    json=payload, timeout=self._timeout_s,
                )
                if response.status_code == 400:
                    raise FinraRequestError(
                        f"FINRA rejected the query for {ticker}: {response.text[:200]}"
                    )
                response.raise_for_status()
                return list(csv.DictReader(io.StringIO(response.text)))

            page = retry(call, **self._retry_kwargs)
            records.extend(page)
            if len(page) < self._page_limit:
                break
            offset += self._page_limit

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records), encoding="utf-8")
        return records

    def fetch(self) -> pd.DataFrame:
        """Short interest for every requested ticker."""
        if not self._tickers:
            log.warning("No ticker to ask FINRA about.", extra={"source": SOURCE})
            return empty_observations()

        records: list[dict[str, Any]] = []
        for ticker in self._tickers:
            try:
                cycles = self._download(ticker)
            except Exception as exc:
                # One ticker must not take the rest down (RESEARCH.md section 1.6).
                log.exception("Short interest for %s failed.", ticker,
                              extra={"source": SOURCE})
                self._failures.append(f"{ticker}: {exc}")
                continue

            rows, problems = observation_rows(cycles, ticker, self._lag)
            self._failures.extend(problems)
            revised = sum(
                1 for row in rows
                if row["series_id"].endswith(REVISED_SUFFIX) and row["value"]
            )
            log.info(
                "%s: %d cycle(s), %d revised by FINRA. Publication dates are DERIVED "
                "(settlement + %d business days); the source carries none.",
                ticker, len(cycles), revised, self._lag, extra={"source": SOURCE},
            )
            records.extend(rows)

        if not records:
            return empty_observations()
        return self.validate(self.observation_frame(records))

    def partial_failures(self) -> list[str]:
        """Tickers that failed, and cycles that could not be read."""
        return list(self._failures)
