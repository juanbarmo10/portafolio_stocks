"""The laboratory's data: everything the panel already stored, read point-in-time.

No network. Prices and membership come from the database; fundamentals, insider purchases
and earnings dates from the caches the studies filled (``.cache/sec/member_facts``,
``form345``, ``earnings_history``). What is missing is said, never fetched behind your back:
``python run_validation.py --factors`` / ``--insiders`` / ``--pead`` fill those caches.

Disciplines built in, so an experiment cannot break them by accident:

- **Point-in-time macro** (section 9.4): :meth:`Lab.fred` gives, for each session, the value
  that was *published* by then — the latest reference date whose release is on or before
  the session, in its latest version known then. Never the revised series of today.
- **Total return** (section 9.1): prices are total-return indices built by the panel's
  tested ``adjustments.total_return_index`` (dividends added on their date), so a ratio of
  two dates is the return a holder got. :meth:`Lab.prices` gives split-adjusted closes for
  technical rules (moving averages, highs), where only ratios matter.
- **Membership of the day** (section 9.5): :meth:`Lab.members` says who was in the S&P 500
  on each session. The universe is the one the index had, not today's — with the coverage
  limit of §2.24: members that later disappeared from yfinance have no prices, which biases
  results **in favour** of any rule (survivorship). Every report says it.

Heavy frames are cached on disk in ``.cache/lab`` keyed by the database's modification
time: a new ingest rebuilds them once.
"""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import Settings, load_settings
from core.logging_setup import get_logger
from db.database import open_connection, read_observations, read_table

log = get_logger(__name__)

CACHE = Path(".cache/lab")
BENCHMARK = "SPY"


class Lab:
    """One handle on every dataset. Methods load lazily and memoize."""

    def __init__(self, settings: Settings | None = None, cache_dir: Path = CACHE) -> None:
        self.settings = settings or load_settings()
        self.cache_dir = Path(cache_dir)
        self._memo: dict[str, Any] = {}
        db = Path(self.settings.db_path)
        self._stamp = int(db.stat().st_mtime) if db.exists() else 0

    # --- plumbing ---------------------------------------------------------------------

    def _query(self, fn):
        conn = open_connection(self.settings.db_path)
        try:
            return fn(conn)
        finally:
            conn.close()

    def _disk(self, name: str, build):
        """``build()`` once per database version, pickled in the cache directory."""
        if name in self._memo:
            return self._memo[name]
        path = self.cache_dir / f"{name}_{self._stamp}.pkl"
        if path.exists():
            value = pickle.loads(path.read_bytes())
        else:
            log.info("Lab: building %s (cached afterwards).", name)
            value = build()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for old in self.cache_dir.glob(f"{name}_*.pkl"):
                old.unlink(missing_ok=True)
            path.write_bytes(pickle.dumps(value))
        self._memo[name] = value
        return value

    # --- universe and prices ------------------------------------------------------------

    def intervals(self) -> pd.DataFrame:
        """S&P 500 membership as ``[ticker, start_date, end_date, …]`` stays (§9.5)."""
        if "intervals" not in self._memo:
            self._memo["intervals"] = self._query(
                lambda c: read_table(c, "universe_membership"))
        return self._memo["intervals"]

    def actions(self) -> pd.DataFrame:
        if "actions" not in self._memo:
            self._memo["actions"] = self._query(lambda c: read_table(c, "corporate_actions"))
        return self._memo["actions"]

    def _raw_closes(self) -> pd.DataFrame:
        def build():
            tickers = sorted(set(self.intervals()["ticker"]) | set(self.reference_tickers()))
            return self._query(lambda c: read_observations(
                c, series_ids=[f"{t}:close_raw" for t in tickers]))
        return self._disk("raw_closes", build)

    def reference_tickers(self) -> list[str]:
        """Market references (SPY, sectors, metals…) plus the researched companies."""
        written = [str(c["ticker"]) for c in self.settings.researched_companies
                   if c.get("ticker")]
        return list(dict.fromkeys([*self.settings.market_references, *written]))

    def calendar(self) -> pd.DatetimeIndex:
        """The benchmark's sessions: the clock every frame is aligned to."""
        return self.total_return([BENCHMARK]).dropna(how="all").index

    def total_return(self, tickers: list[str] | None = None) -> pd.DataFrame:
        """``sessions × tickers`` total-return indices (dividends on their date, §9.1)."""
        from validation.factors import total_return_wide  # noqa: PLC0415

        def build():
            raw = self._raw_closes()
            spy = raw[raw["series_id"] == f"{BENCHMARK}:close_raw"]
            calendar = pd.DatetimeIndex(sorted(pd.to_datetime(spy["ts"].str[:10]).unique()))
            today = pd.Timestamp.today().date().isoformat()
            return total_return_wide(raw.assign(ts=raw["ts"].str[:10]), self.actions(),
                                     today, calendar)
        wide = self._disk("total_return", build)
        return wide if tickers is None else wide.reindex(columns=tickers)

    def prices(self, tickers: list[str] | None = None) -> pd.DataFrame:
        """``sessions × tickers`` split-adjusted closes, for technical rules (ratios only)."""
        from transform import breadth as br  # noqa: PLC0415

        def build():
            wide = br.adjusted_closes(br.wide_closes(self._raw_closes()),
                                      br.splits_by_ticker(self.actions()))
            return wide.reindex(self.calendar())
        wide = self._disk("prices", build)
        return wide if tickers is None else wide.reindex(columns=tickers)

    def members(self) -> pd.DataFrame:
        """``sessions × tickers``: in the S&P 500 that session (§9.5)."""
        from transform import breadth as br  # noqa: PLC0415

        def build():
            tickers = sorted(set(self.intervals()["ticker"]))
            return br.membership_mask(self.intervals(), self.calendar(), tickers)
        return self._disk("members", build)

    def members_on(self, day: Any) -> list[str]:
        """Members on ``day`` (the last session on or before it) that have a price."""
        mask = self.members()
        on = mask.index[mask.index <= pd.Timestamp(day)]
        if not len(on):
            return []
        row = mask.loc[on[-1]]
        priced = set(self.total_return().columns)
        return [t for t in row.index[row.to_numpy()] if t in priced]

    # --- macro ------------------------------------------------------------------------

    def fred(self, series_id: str) -> pd.Series:
        """The value **as published** on each session (§9.4): for every session, the latest
        reference date released by then, in its latest version known then."""
        key = f"fred:{series_id}"
        if key not in self._memo:
            rows = self._query(lambda c: read_observations(c, series_ids=[series_id]))
            self._memo[key] = known_as_of(rows, self.calendar())
        return self._memo[key]

    def regime(self):
        """The panel's regime light (``transform.regime.build``), point-in-time."""
        if "regime" not in self._memo:
            from transform import regime as rg  # noqa: PLC0415

            level2 = self.settings.raw["panel"]["level2"]
            fred = self._query(lambda c: read_observations(c, source="fred"))
            etfs = self._query(lambda c: read_observations(
                c, series_ids=[f"{t}:close_raw" for t in rg.regime_tickers(level2)]))
            self._memo["regime"] = rg.build(fred, etfs, self.actions(), level2)
        return self._memo["regime"]

    # --- companies --------------------------------------------------------------------

    def cik_of(self) -> dict[str, str]:
        """Today's ticker → CIK from the SEC map in cache. ⚠️ Today's tickers: a ticker
        reused by another company would map to the new one (§9.3) — the factor studies
        carry the same limit."""
        if "cik_of" not in self._memo:
            from ingest.sec_filings import normalize_ticker, ticker_map  # noqa: PLC0415

            path = self._sec_cache() / "company_tickers.json"
            if not path.exists():
                raise FileNotFoundError(f"{path}: run any SEC ingest once.")
            mapping = ticker_map(json.loads(path.read_text(encoding="utf-8")))
            tickers = set(self.intervals()["ticker"]) | set(self.reference_tickers())
            self._memo["cik_of"] = {t: mapping[normalize_ticker(t)] for t in tickers
                                    if normalize_ticker(t) in mapping}
        return self._memo["cik_of"]

    def _sec_cache(self) -> Path:
        return Path(str(self.settings.source("sec").get("cache_dir", ".cache/sec")))

    def facts(self) -> pd.DataFrame:
        """Every filed value of the factor metrics for the S&P 500 members in cache
        (``validation.factors.RECORD_COLUMNS``), each with its filing date."""
        from validation import factors  # noqa: PLC0415

        def build():
            sec = self.settings.source("sec")
            folder = self._sec_cache() / "member_facts"
            if not folder.exists():
                raise FileNotFoundError(f"{folder}: run `python run_validation.py --factors` "
                                        "once to fill it.")
            by_cik = {}
            for path in folder.glob("*.json.gz"):
                stored = json.loads(gzip.decompress(path.read_bytes()))
                by_cik[path.name.split(".")[0]] = stored.get("facts", {})
            concepts = {m: sec["concepts"][m] for m in factors.METRICS}
            return factors.fact_records(by_cik, concepts, sec["duration_windows"])
        return self._disk("facts", build)

    def insider_purchases(self) -> pd.DataFrame:
        """Form 4 open-market purchases of directors and officers, from the cached quarters."""
        if "insiders" not in self._memo:
            folder = self._sec_cache() / "form345"
            files = sorted(folder.glob("*_purchases.csv.gz"))
            if not files:
                raise FileNotFoundError(f"{folder}: run `python run_validation.py --insiders`.")
            self._memo["insiders"] = pd.concat(
                [pd.read_csv(f, dtype={"issuer_cik": str, "owner_cik": str, "accession": str})
                 for f in files], ignore_index=True)
        return self._memo["insiders"]

    def earnings_dates(self) -> pd.DataFrame:
        """``[ticker, date]``: every 8-K item 2.02 (results) of the cached members."""
        if "earnings" not in self._memo:
            folder = self._sec_cache() / "earnings_history"
            if not folder.exists():
                raise FileNotFoundError(f"{folder}: run `python run_validation.py --pead`.")
            ticker_of = {cik: t for t, cik in self.cik_of().items()}
            rows = []
            for path in folder.glob("*.json"):
                stored = json.loads(path.read_text(encoding="utf-8"))
                ticker = ticker_of.get(stored.get("cik", path.stem))
                if ticker:
                    rows += [{"ticker": ticker, "date": pd.Timestamp(d)}
                             for d in stored.get("dates", [])]
            self._memo["earnings"] = pd.DataFrame(rows, columns=["ticker", "date"])
        return self._memo["earnings"]


def known_as_of(rows: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    """Point-in-time series on ``calendar`` from long rows ``[ts, ts_release, value]``.

    For every session: of the observations released on or before it, the one with the latest
    reference date, in the latest version of that date released by then. A row with an
    unknown release date (``''``) is left out — it cannot be placed in time.
    """
    if rows is None or rows.empty:
        return pd.Series(dtype=float, index=calendar)
    frame = rows[rows["ts_release"].astype(str).str.len() > 0].copy()
    frame["release"] = pd.to_datetime(frame["ts_release"].astype(str).str[:10])
    frame["ref"] = pd.to_datetime(frame["ts"].astype(str).str[:10])
    frame = frame.sort_values(["release", "ref"])
    current: dict[pd.Timestamp, float] = {}
    points: dict[pd.Timestamp, float] = {}
    for release, group in frame.groupby("release", sort=True):
        for ref, value in zip(group["ref"], group["value"]):
            current[ref] = float(value)
        points[release] = current[max(current)]
    series = pd.Series(points).sort_index()
    # A release on a non-session day counts from the next session on (never before).
    return series.reindex(series.index.union(calendar)).ffill().reindex(calendar)
