"""S&P 500 membership over time, and the member closes market breadth needs (section 9.5).

Breadth — "what share of the index sits above its 200-session average" — needs to know who
was a member **on each date**. Today's list applied to the past is survivorship bias, and
it is worst exactly in the crises breadth exists to see: the companies that collapsed are
the ones missing from today's list (THEORY.md §0.3).

**Membership: free, and reconstructed.** From the ``fja05680/sp500`` dataset (MIT licence,
*Copyright (c) 2019-2020 Farrell J. Aultman*): 1996-2019 from the download that accompanies
Andreas Clenow's *Trading Evolved*, maintained by hand since then from Wikipedia with the
exact dates checked change by change. Its own README advises starting in 2001. Wikipedia's
API cannot be used directly: its ``robots.txt`` disallows ``/w/`` (RESEARCH.md §2.24).

**Prices: the bottleneck, and not solvable for free.** yfinance does not serve companies
that have stopped trading — worse, it *erases* their past when they delist. Measured on
2026-09-23: of the index members on each date, yfinance can price 63 % at the 2009 low,
84 % in mid-2019 and 86 % from early 2020. The missing ones are acquired, bankrupt or
renamed — not a random sample. So a breadth reading built this way **reduces** the bias
without removing it, and the coverage has to travel with every value
(``transform/breadth.py``). Nasdaq's WIKI prices, once the standard free answer, no longer
exist (checked).

**What makes the forward half pure** is not the membership list but the storage: every
member's close is stored the day it is fetched, and ``upsert`` never deletes. When a member
later delists and yfinance forgets it, this database does not.

Members yfinance cannot price are **not failures** — they are the coverage gap, expected
and counted. Only a download that errors counts as one.

fetch() -> DataFrame[source, series_id, ts, ts_release, value]
    series_id = "TICKER:close_raw"      raw close, split-reconstructed (section 9.1)
fetch_tables() -> {"universe_membership": [...], "corporate_actions": [...]}
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from core.config import Settings
from core.logging_setup import get_logger
from ingest.base import Ingester, empty_observations, retry
from ingest.prices import build_corporate_actions, build_observations, unadjust_history

log = get_logger(__name__)

SOURCE = "yfinance"                     # of the closes this ingester writes
MEMBERSHIP_SOURCE = "fja05680_sp500"
UNIVERSE = "sp500"
CLOSE = "close_raw"


class MembershipFormatError(RuntimeError):
    """The membership file no longer has the shape this parser was written against."""


def parse_snapshots(text: str) -> list[tuple[str, frozenset[str]]]:
    """``date,tickers`` change-point snapshots -> ``[(iso_date, members)]``, oldest first.

    Each row is the *complete* membership from that date until the next row. Fails loudly
    on a structural change rather than returning a thinner universe (section 10).
    """
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or {"date", "tickers"} - set(reader.fieldnames):
        raise MembershipFormatError(
            f"Expected columns 'date,tickers', got {reader.fieldnames!r}."
        )
    out: list[tuple[str, frozenset[str]]] = []
    for row in reader:
        try:
            day = dt.date.fromisoformat(row["date"].strip()).isoformat()
        except ValueError as exc:
            raise MembershipFormatError(f"Unreadable date {row['date']!r}.") from exc
        members = frozenset(t.strip() for t in row["tickers"].split(",") if t.strip())
        if len(members) < 400:
            # The index has ~500 names; a row with far fewer is a truncated file, and
            # computing breadth over it would silently describe a different universe.
            raise MembershipFormatError(
                f"{day}: only {len(members)} members — the file looks truncated."
            )
        out.append((day, members))
    if not out:
        raise MembershipFormatError("The membership file has no rows.")
    return sorted(out)


def membership_intervals(
    snapshots: Iterable[tuple[str, frozenset[str]]],
    *,
    universe: str = UNIVERSE,
    source: str = MEMBERSHIP_SOURCE,
    first_seen: str,
) -> list[dict[str, Any]]:
    """Turn change-point snapshots into ``[start, end)`` intervals, one per stay.

    A ticker that leaves and later re-enters gets **two** intervals: collapsing them into
    first-and-last dates would count it as a member through the years it was out, which is
    how a first/last summary quietly misstates the universe.
    """
    open_since: dict[str, str] = {}
    rows: list[dict[str, Any]] = []

    def close(ticker: str, end: str | None) -> None:
        rows.append({
            "universe": universe, "ticker": ticker, "start_date": open_since.pop(ticker),
            "end_date": end, "source": source, "first_seen": first_seen,
        })

    for day, members in snapshots:
        for ticker in [t for t in open_since if t not in members]:
            close(ticker, day)                      # end is exclusive
        for ticker in members:
            open_since.setdefault(ticker, day)
    for ticker in list(open_since):
        close(ticker, None)
    return sorted(rows, key=lambda r: (r["ticker"], r["start_date"]))


def members_since(intervals: Iterable[dict[str, Any]], since: str) -> list[str]:
    """Every ticker that was a member at any point on or after ``since``."""
    return sorted({
        row["ticker"] for row in intervals
        if row["end_date"] is None or row["end_date"] > since
    })


def yf_symbol(ticker: str) -> str:
    """Yahoo spells class shares with a dash: ``BRK.B`` -> ``BRK-B`` (THEORY.md §6.1)."""
    return ticker.replace(".", "-")


def incomplete_sessions(
    closes: pd.DataFrame, *, window: int = 21, floor: float = 0.5
) -> list[str]:
    """Sessions where far fewer tickers have a close than on the sessions around them.

    Found on the first real run (2026-09-23): a batch download near midnight New York time
    came back with the 2026-09-22 session empty for most tickers while the days either side
    were whole — Yahoo apparently mid-rebuild. The call succeeded, the frame had rows, and
    the hole was silent; stored as-is it dropped that day's breadth coverage to 12 % and
    split the "usable" series in two. A slow change in the count (tickers listing over the
    years) is normal; a one-day collapse against the surrounding median is not.

    Returns:
        ISO dates of the suspect sessions. Detection only — nothing is filled in, because
        a forward-filled close is an estimate presented as data (section 12). Measured
        later the same night: the hole was still at the source, so it is not always
        transient (``UniverseIngester._report_holes``).
    """
    if closes is None or closes.empty:
        return []
    counts = closes.notna().sum(axis=1)
    typical = counts.rolling(window, center=True, min_periods=5).median()
    suspect = counts[counts < floor * typical]
    return [pd.Timestamp(day).date().isoformat() for day in suspect.index]


def _batch_closes(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """The ``Close`` column of every ticker in a ``group_by='ticker'`` download."""
    frames = {}
    for ticker in tickers:
        try:
            frame = data[yf_symbol(ticker)] if len(tickers) > 1 else data
            frames[ticker] = frame["Close"]
        except (KeyError, TypeError):
            continue
    return pd.DataFrame(frames)


class UniverseIngester(Ingester):
    """Index membership over time plus every member's raw close, for breadth."""

    source = SOURCE

    def __init__(self, settings: Settings) -> None:
        cfg = settings.source("sp500_membership")
        self._url = str(cfg.get("url", ""))
        self._cache_dir = Path(str(cfg.get("cache_dir", ".cache/universe")))
        self._cache_ttl = dt.timedelta(hours=float(cfg.get("cache_ttl_hours", 168)))
        self._price_start = str(cfg.get("price_history_start", "2017-01-01"))
        self._batch = int(cfg.get("batch_size", 80))
        self._pause_s = float(cfg.get("batch_pause_s", 3))
        self._hole_retry_pause_s = float(cfg.get("hole_retry_pause_s", 5))
        self._hole_alert_sessions = int(cfg.get("hole_alert_sessions", 5))
        self._currency = settings.source("yfinance").get("currency", "USD")
        retry_cfg = settings.raw.get("ingest", {}).get("retry", {})
        self._retry_kwargs = {
            "attempts": retry_cfg.get("attempts", 4),
            "base_delay_s": retry_cfg.get("base_delay_s", 1.0),
            "max_delay_s": retry_cfg.get("max_delay_s", 30.0),
        }
        self._tables: dict[str, list[dict[str, Any]]] = {}
        self._failures: list[str] = []
        # Weekly by the user's decision (2026-09-25): ~1,4 M rows and ~5 min a day for a series
        # the sector breadth tracks at 0,91 correlation. Due when the last run is older than
        # this; `run_ingest.py --force` runs it anyway.
        self._every_days = int(cfg.get("run_every_days", 7))
        self.force = False
        self._skip_reason: str | None = None
        self.unpriced: list[str] = []

    @staticmethod
    def is_available(settings: Settings) -> bool:
        """Needs yfinance and a configured membership source."""
        try:
            import yfinance  # noqa: F401, PLC0415 — optional extra
        except ImportError:
            return False
        return bool(settings.source("sp500_membership").get("url"))

    def _membership_text(self) -> str:
        """The membership file, from a cache refreshed weekly — the source changes monthly."""
        path = self._cache_dir / "sp500_historical_components.csv"
        if path.exists():
            age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
            if age < self._cache_ttl:
                return path.read_text(encoding="utf-8")

        def call() -> str:
            response = requests.get(
                self._url, headers={"User-Agent": "equitydash (personal research)"},
                timeout=120,
            )
            response.raise_for_status()
            return response.text

        try:
            text = retry(call, **self._retry_kwargs)
        except Exception as exc:
            if path.exists():
                self._failures.append(f"membership: using expired cache ({exc})")
                return path.read_text(encoding="utf-8")
            raise
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text

    def _download_batch(self, tickers: list[str]) -> pd.DataFrame:
        import yfinance  # noqa: PLC0415 — optional extra

        def call() -> pd.DataFrame:
            # Split-adjusted closes with the split events alongside: the same inputs the
            # per-ticker price ingester reconstructs raw closes from (section 9.1).
            return yfinance.download(
                [yf_symbol(t) for t in tickers], start=self._price_start,
                auto_adjust=False, actions=True, group_by="ticker",
                threads=True, progress=False,
            )

        # yfinance logs an ERROR line for every ticker it cannot serve — here ~100 of
        # them, all expected (they are the coverage gap) and all counted in one summary
        # line by fetch(). A log that cries error a hundred times per run trains the
        # reader to ignore it, so its logger is quieted for the call only. A failure of
        # the batch itself still raises, and is still reported.
        yf_log = logging.getLogger("yfinance")
        previous = yf_log.level
        yf_log.setLevel(logging.CRITICAL)
        try:
            return retry(call, **self._retry_kwargs)
        finally:
            yf_log.setLevel(previous)

    def attach_database(self, conn: Any) -> None:
        """Decide whether the weekly run is due, from when the membership was last written."""
        if self.force or self._every_days <= 0:
            return
        try:
            row = conn.execute("SELECT MAX(ingested_at) FROM universe_membership").fetchone()
        except Exception as exc:  # noqa: BLE001 — no table yet: the first run is due
            log.debug("No membership table yet: %s", exc)
            return
        if not row or not row[0]:
            return
        last = dt.datetime.fromisoformat(str(row[0]))
        age = dt.datetime.now(dt.timezone.utc) - last
        if age < dt.timedelta(days=self._every_days):
            self._skip_reason = (f"last run {age.days} day(s) ago, due every "
                                 f"{self._every_days}; `--force` to run it now")

    def fetch(self) -> pd.DataFrame:
        """Membership intervals (for ``fetch_tables``) and the members' raw closes."""
        if self._skip_reason:
            log.info("Universe skipped: %s.", self._skip_reason, extra={"source": SOURCE})
            self._tables, self._failures = {}, []
            return empty_observations()
        self._failures = []
        self.unpriced = []
        today = dt.date.today().isoformat()

        intervals = membership_intervals(
            parse_snapshots(self._membership_text()), first_seen=today
        )
        tickers = members_since(intervals, self._price_start)
        log.info(
            "%d membership intervals; %d tickers were members at some point since %s.",
            len(intervals), len(tickers), self._price_start, extra={"source": SOURCE},
        )

        frames: list[pd.DataFrame] = []
        actions: list[dict[str, Any]] = []
        for i in range(0, len(tickers), self._batch):
            batch = tickers[i:i + self._batch]
            try:
                data = self._download_batch(batch)
                closes = _batch_closes(data, batch)
                holes = incomplete_sessions(closes)
                if holes:
                    # One short retry: a source mid-rebuild may come back whole. In the
                    # case measured it did not — the hole was still there hours later —
                    # so the retry is cheap on purpose rather than patient.
                    time.sleep(self._hole_retry_pause_s)
                    data = self._download_batch(batch)
                    closes = _batch_closes(data, batch)
                    holes = incomplete_sessions(closes)
                if holes:
                    self._report_holes(holes, closes, batch[0])
            except Exception as exc:
                # A whole batch failing is a real failure — network, rate limit — unlike
                # a single delisted ticker, which is the coverage gap and expected.
                log.exception("Price batch %d failed.", i // self._batch,
                              extra={"source": SOURCE})
                self._failures.append(f"batch starting {batch[0]}: {exc}")
                continue

            for ticker in batch:
                symbol = yf_symbol(ticker)
                try:
                    history = data[symbol] if len(batch) > 1 else data
                    history = history.dropna(subset=["Close"])
                except (KeyError, TypeError):
                    history = pd.DataFrame()
                if history.empty:
                    self.unpriced.append(ticker)
                    continue
                splits = _events(history, "Stock Splits")
                dividends = _events(history, "Dividends")
                raw = unadjust_history(history, splits)
                obs = build_observations(ticker, raw)
                frames.append(obs[obs["series_id"] == f"{ticker}:{CLOSE}"])
                actions.extend(
                    build_corporate_actions(ticker, splits, dividends, self._currency)
                )
            time.sleep(self._pause_s)

        log.info(
            "%d of %d members priced; %d have no prices in yfinance — expected: that is "
            "the survivorship gap breadth coverage reports (section 9.5), not a failure.",
            len(tickers) - len(self.unpriced), len(tickers), len(self.unpriced),
            extra={"source": SOURCE},
        )
        self._tables = {"universe_membership": intervals, "corporate_actions": actions}
        df = pd.concat(frames, ignore_index=True) if frames else empty_observations()
        return self.validate(df)

    def _report_holes(self, holes: list[str], closes: pd.DataFrame, first: str) -> None:
        """A hole is stored as it came — never filled — and reported by how fresh it is.

        A **recent** hole may still be transient, and the full window is re-fetched every
        run, so tomorrow may fill it: that is worth the exit code. An **old** hole that is
        still there is a known defect of the source, and failing every run over it would be
        an alarm that cries wolf until nobody reads it. It is logged, and the breadth
        transform flags the session on screen (``BreadthReading.source_hole``).
        """
        sessions = sorted({pd.Timestamp(d).date().isoformat() for d in closes.index})
        recent_from = sessions[-self._hole_alert_sessions] if sessions else ""
        recent = [h for h in holes if h >= recent_from]
        log.warning(
            "Batch from %s: source returned session(s) %s nearly empty; stored as they "
            "came, not filled.", first, holes, extra={"source": SOURCE},
        )
        if recent:
            self._failures.append(
                f"batch starting {first}: recent incomplete session(s) at source {recent}"
            )

    def fetch_tables(self) -> dict[str, list[dict[str, Any]]]:
        """Membership intervals and the members' splits and dividends."""
        return {table: rows for table, rows in self._tables.items() if rows}

    def partial_failures(self) -> list[str]:
        """Real failures only. Unpriceable members are the coverage gap, not errors."""
        return list(self._failures)


def _events(history: pd.DataFrame, column: str) -> dict[dt.date, float]:
    """Non-zero action rows of a history frame, keyed by calendar date."""
    if column not in history.columns:
        return {}
    return {
        (ts.date() if hasattr(ts, "date") else ts): float(value)
        for ts, value in history[column].items() if value
    }
