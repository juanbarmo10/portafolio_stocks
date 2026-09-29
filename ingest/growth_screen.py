"""The growth screen's inputs: every SEC filer, quarterly (CLAUDE.md §15.5, point 2).

The quarterly screen (``ingest/screen``) reads the current S&P 500 on annual figures, and its
default filters (dilution ≤ 2 %, FCF ≥ 0, cap ≥ 2 bn) leave out by construction most growth
companies in their early years. This one reads **every company in the SEC ``frames``** — the
same payloads, which already carry every filer — to look for growth, its acceleration and
its quality.

What is fetched, and why this shape (verified against the API on 2026-09-28):

- **Revenue, gross profit, operating income and basic shares for the three most recent
  calendar quarters and the same three a year earlier.** Each company is then read at *its
  own* latest reported quarter (``transform/growth_screen``). A single calendar quarter
  would not do: the calendar Q4 barely exists as a 3-month fact (791 filers against ~3,900 in
  Q1-Q3; the Q4 lives inside the 10-K, section 9.12), and a company whose fiscal year ends in
  June lacks the calendar Q2 for the same reason.
- **Cash flow from the annual frame only.** A 3-month operating cash flow exists for 329
  filers: most present the statement as year-to-date sums (section 9.14). So FCF, its margin
  and SBC are the latest calendar year's.
- **Cash and short-term investments** at each recent quarter end, for the runway.
- **Price, median dollar volume, splits and the 6- and 12-month price returns** from
  yfinance, for the companies with revenue in a recent quarter. yfinance is a scraper
  (section 4.3): one summary per company, never a price history, and a missing price leaves
  the market cap unknown. The returns are the relative-strength context of §15.5 point 8:
  shown, never sorted by (the 12-1 momentum changed sign in the S&P 500, RESEARCH §2.49).

Every row is keyed by **CIK** (section 9.3), prices included: the ticker is only how
yfinance is asked. The companies go to the ``companies`` registry with their SIC code, as the
annual screen's do — the researched ones are left to ``sec_filings``, which owns their card's
sector. The public copy only carries researched companies, so this never travels.

Like the annual screen, ``frames`` gives no filing date: these rows serve today's screen and
**never** a backtest (section 9.4). One run a quarter (``screen.growth.run_every_days``).

fetch() -> observations, sources ``sec_growth`` and ``yfinance_screen``.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from core.config import Settings
from core.logging_setup import get_logger
from db.loader import TS_RELEASE_UNKNOWN
from ingest.base import Ingester, empty_observations, retry
from ingest.screen import (FRAMES_URL, cached_json, frame_rows, resolved_records,
                           screen_year, sic_map)
from ingest.universe import yf_symbol

log = get_logger(__name__)

SOURCE = "sec_growth"
PRICE_SOURCE = "yfinance_screen"
# A return's anchor close may sit this many days before its target date (weekends, holidays),
# no more: an older one would measure a longer span than the one named.
ANCHOR_SLACK_DAYS = 7

# Diluted shares only for the market cap of a company that files no total basic count (two
# share classes, such as Duolingo); dilution itself is measured on basic shares (§9.15).
QUARTERLY = ("revenue", "gross_profit", "operating_income", "basic_shares", "diluted_shares")
INSTANTS = ("cash", "short_term_investments")
ANNUAL = ("revenue", "operating_cash_flow", "capex", "sbc")


def latest_quarter(today: dt.date, lag_days: int) -> tuple[int, int]:
    """The newest calendar quarter that ended at least ``lag_days`` ago: 10-Qs are due 40
    to 45 days after the quarter, so by then nearly every filer is in the frame."""
    cutoff = pd.Timestamp(today) - pd.Timedelta(days=lag_days)
    year, quarter = cutoff.year, (cutoff.month - 1) // 3 + 1
    end = pd.Timestamp(year=year, month=3 * quarter, day=1) + pd.offsets.MonthEnd(0)
    if end > cutoff:
        year, quarter = (year, quarter - 1) if quarter > 1 else (year - 1, 4)
    return year, quarter


def shift(year: int, quarter: int, back: int) -> tuple[int, int]:
    """The calendar quarter ``back`` quarters before ``(year, quarter)``."""
    index = year * 4 + (quarter - 1) - back
    return index // 4, index % 4 + 1


def label(year: int, quarter: int) -> str:
    return f"CY{year}Q{quarter}"


def quarter_periods(year: int, quarter: int) -> tuple[list[str], list[str]]:
    """``(recent, year_ago)``: the three latest quarters and the same three a year before."""
    recent = [label(*shift(year, quarter, n)) for n in range(3)]
    ago = [label(*shift(year, quarter, n + 4)) for n in range(3)]
    return recent, ago


def first_tickers(raw_map: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """``{cik: (ticker, name)}``: the first entry of each CIK in ``company_tickers.json``,
    which lists a company's main listing before its other classes, warrants and units."""
    out: dict[str, tuple[str, str]] = {}
    for row in raw_map.values():
        cik = str(row["cik_str"]).zfill(10)
        out.setdefault(cik, (str(row["ticker"]).upper(), str(row["title"])))
    return out


def registry_rows(registry: dict[str, tuple[str, str]], tickers: dict[str, str],
                  sic: dict[str, list[Any]], researched: set[str],
                  today: str) -> list[dict[str, Any]]:
    """``companies`` rows for the screened companies, researched ones left out."""
    return [{"cik": cik, "ticker": registry[cik][0], "name": registry[cik][1],
             "sector": None, "thesis_category": None, "first_seen": today,
             "status": "active", "sic": (sic.get(cik) or [None])[0],
             "sic_description": (sic.get(cik) or [None, None])[1]}
            for cik in sorted(tickers) if cik not in researched]


def price_summary(history: pd.DataFrame, sessions: int,
                  horizons: Mapping[str, int] | None = None) -> dict[str, Any] | None:
    """Last close, median daily dollar volume over the last ``sessions``, the splits, and the
    price return over each ``{name: days}`` horizon, from one yfinance history
    (``auto_adjust=False, actions=True``). ``None`` without a close.

    The close is split-adjusted by the source, and the last one has no later split to be
    adjusted by, so it is the raw close (section 9.1). Price × volume is invariant to a
    split, so the dollar volume needs no reconstruction either; and a return is a ratio of
    two closes of the same download, adjusted by the same factors, so it is right across a
    split too. A return needs a close on or before ``last day − days`` and no more than
    ``ANCHOR_SLACK_DAYS`` before it: a younger listing, or a hole in the source, leaves it
    ``None`` rather than measured over a shorter span (section 12). Price only, without
    dividends.
    """
    if history is None or history.empty or "Close" not in history:
        return None
    frame = history.dropna(subset=["Close"])
    if frame.empty:
        return None
    tail = frame.tail(sessions)
    volume = (tail["Close"] * tail["Volume"]).median() if "Volume" in tail else None
    splits = frame["Stock Splits"] if "Stock Splits" in frame else pd.Series(dtype=float)
    splits = splits[splits.fillna(0) > 0]
    days = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    closes = pd.Series(frame["Close"].to_numpy(dtype=float), index=days)
    last_day, last = closes.index[-1], float(closes.iloc[-1])
    returns: dict[str, float | None] = {}
    for name, span in (horizons or {}).items():
        target = last_day - pd.Timedelta(days=int(span))
        before = closes[closes.index <= target]
        ok = (not before.empty and before.iloc[-1] > 0
              and (target - before.index[-1]).days <= ANCHOR_SLACK_DAYS)
        returns[name] = last / float(before.iloc[-1]) - 1 if ok else None
    return {"close": last,
            "day": last_day.date().isoformat(),
            "dollar_volume": None if volume is None or pd.isna(volume) else float(volume),
            "splits": {pd.Timestamp(d).date().isoformat(): float(r) for d, r in splits.items()},
            "returns": returns}


class GrowthScreenIngester(Ingester):
    """Every SEC filer's recent quarters, plus a price summary, once a quarter."""

    source = SOURCE

    def __init__(self, settings: Settings) -> None:
        cfg = settings.source("sec")
        growth = dict(settings.raw.get("screen", {}).get("growth") or {})
        self._base_url = str(cfg.get("base_url", "https://data.sec.gov")).rstrip("/")
        self._user_agent = settings.secret(str(cfg.get("user_agent_env", "SEC_USER_AGENT")))
        self._concepts = dict(cfg.get("concepts", {}))
        cache = Path(str(cfg.get("cache_dir", ".cache/sec")))
        self._cache_dir = cache / "frames"
        self._ticker_map_path = cache / "company_tickers.json"
        self._sic_path = cache / "sic_map.json"
        self._ticker_map_url = str(cfg.get("ticker_map_url",
                                           "https://www.sec.gov/files/company_tickers.json"))
        self._delay_s = 1.0 / float(cfg.get("rate_limit_rps", 8) or 8)
        self._every_days = int(growth.get("run_every_days", 90))
        self._lag_days = int(growth.get("quarter_lag_days", 50))
        self._sessions = int(growth.get("volume_sessions", 60))
        self._history_days = int(growth.get("price_history_days", 400))
        self._horizons = {str(k): int(v) for k, v in
                          (growth.get("return_horizons_days") or {}).items()}
        self._batch = int(growth.get("batch_size", 200))
        self._min_price_coverage = float(growth.get("min_price_coverage", 0.5))
        self._researched = {str(c["cik"]).zfill(10) for c in settings.researched_companies
                            if c.get("cik")}
        self.force = False
        self._skip_reason: str | None = None
        self._failures: list[str] = []
        self._companies: list[dict[str, Any]] = []

    @staticmethod
    def is_available(settings: Settings) -> bool:
        cfg = settings.source("sec")
        agent = settings.secret(str(cfg.get("user_agent_env", "SEC_USER_AGENT")))
        return bool(agent) and bool(settings.raw.get("screen", {}).get("growth"))

    def attach_database(self, conn: Any) -> None:
        """Whether a run is due: once every ``run_every_days`` unless forced."""
        if self.force or self._every_days <= 0:
            return
        try:
            last = conn.execute("SELECT MAX(ingested_at) FROM observations "
                                "WHERE source = ?", (SOURCE,)).fetchone()
        except Exception:  # noqa: BLE001 — no table yet: due
            return
        if last and last[0]:
            age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(str(last[0]))
            if age < dt.timedelta(days=self._every_days):
                self._skip_reason = (f"last run {age.days} day(s) ago, due every "
                                     f"{self._every_days}; `--force` to run it now")

    def _frames(self, metric: str, periods: list[str], candidates: dict) -> None:
        spec = self._concepts.get(metric) or {}
        unit = spec.get("unit", "USD")
        for period in periods:
            for preference, tag in enumerate(spec.get("tags", [])):
                url = FRAMES_URL.format(base=self._base_url, tag=tag, unit=unit, period=period)
                try:
                    payload = cached_json(url, self._cache_dir / f"{tag}_{unit}_{period}.json",
                                          user_agent=self._user_agent, delay_s=self._delay_s)
                    rows = frame_rows(payload, metric, period, None, preference, source=SOURCE)
                except Exception as exc:  # noqa: BLE001 — one tag must not sink the rest
                    log.exception("frames %s %s failed.", tag, period, extra={"source": SOURCE})
                    self._failures.append(f"{tag} {period}: {exc}")
                    continue
                for cik, row in rows.items():
                    candidates.setdefault((cik, metric), {}).setdefault(
                        preference, {})[period] = row

    def _download(self, symbols: list[str]) -> pd.DataFrame:
        import yfinance  # noqa: PLC0415 — optional extra

        yf_log = logging.getLogger("yfinance")
        previous = yf_log.level
        yf_log.setLevel(logging.CRITICAL)   # one ERROR per unserved ticker; counted below
        try:
            return retry(lambda: yfinance.download(
                symbols, start=(dt.date.today()
                                - dt.timedelta(days=self._history_days)).isoformat(),
                auto_adjust=False, actions=True,
                group_by="ticker", threads=True, progress=False), attempts=2)
        finally:
            yf_log.setLevel(previous)

    def _prices(self, tickers: dict[str, str]) -> list[dict[str, Any]]:
        """Price-summary observations for ``{cik: ticker}``."""
        rows: list[dict[str, Any]] = []
        ciks = sorted(tickers)
        priced = 0
        for i in range(0, len(ciks), self._batch):
            batch = ciks[i:i + self._batch]
            symbols = [yf_symbol(tickers[c]) for c in batch]
            try:
                data = self._download(symbols)
            except Exception as exc:  # noqa: BLE001 — a batch failing is a real failure
                log.exception("Price batch %d failed.", i // self._batch,
                              extra={"source": PRICE_SOURCE})
                self._failures.append(f"prices, batch starting {tickers[batch[0]]}: {exc}")
                continue
            for cik, symbol in zip(batch, symbols):
                try:
                    history = data[symbol] if len(batch) > 1 else data
                except (KeyError, TypeError):
                    continue
                summary = price_summary(history, self._sessions, self._horizons)
                if summary is None:
                    continue
                priced += 1
                stamp = f"{summary['day']}T00:00:00+00:00"
                base = {"source": PRICE_SOURCE, "ts_release": TS_RELEASE_UNKNOWN}
                rows.append({**base, "series_id": f"{cik}:close", "ts": stamp,
                             "value": summary["close"]})
                if summary["dollar_volume"] is not None:
                    rows.append({**base, "series_id": f"{cik}:dollar_volume", "ts": stamp,
                                 "value": summary["dollar_volume"]})
                for name, value in summary["returns"].items():
                    if value is not None:
                        rows.append({**base, "series_id": f"{cik}:return_{name}", "ts": stamp,
                                     "value": value})
                for day, ratio in summary["splits"].items():
                    rows.append({**base, "series_id": f"{cik}:split", "ts": f"{day}T00:00:00+00:00",
                                 "value": ratio})
            log.info("Growth screen prices: %d/%d batches, %d priced.", i // self._batch + 1,
                     -(-len(ciks) // self._batch), priced, extra={"source": PRICE_SOURCE})
        if ciks and priced < self._min_price_coverage * len(ciks):
            self._failures.append(f"prices for only {priced} of {len(ciks)} companies: the "
                                  "source may have changed")
        return rows

    def fetch(self) -> pd.DataFrame:
        if self._skip_reason:
            log.info("Growth screen skipped: %s.", self._skip_reason, extra={"source": SOURCE})
            return empty_observations()
        self._failures = []
        year, quarter = latest_quarter(dt.date.today(), self._lag_days)
        recent, ago = quarter_periods(year, quarter)
        annual = f"CY{screen_year(dt.date.today())}"
        candidates: dict[tuple[str, str], dict[int, dict[str, dict[str, Any]]]] = {}
        # Revenue in one pass over quarters and year, so one tag serves both (resolve_tags).
        self._frames("revenue", recent + ago + [annual], candidates)
        for metric in QUARTERLY[1:]:
            self._frames(metric, recent + ago, candidates)
        for metric in INSTANTS:
            self._frames(metric, [f"{p}I" for p in recent], candidates)
        for metric in ANNUAL[1:]:
            self._frames(metric, [annual], candidates)
        records = resolved_records(candidates)

        wanted = set(recent)
        with_revenue = {r["series_id"].split(":")[0] for r in records
                        if r["series_id"].split(":")[1] == "revenue"
                        and r["series_id"].split(":")[2] in wanted}
        raw_map = cached_json(self._ticker_map_url, self._ticker_map_path,
                              user_agent=self._user_agent, delay_s=self._delay_s)
        registry = first_tickers(raw_map)
        tickers = {cik: registry[cik][0] for cik in with_revenue if cik in registry}
        log.info("Growth screen: %d values, %d companies with a recent quarter's revenue, %d "
                 "with a ticker; quarters %s vs %s, annual %s.", len(records),
                 len(with_revenue), len(tickers), recent, ago, annual,
                 extra={"source": SOURCE})
        records += self._prices(tickers)
        sic = sic_map(set(tickers), self._sic_path, base_url=self._base_url,
                      user_agent=self._user_agent, delay_s=self._delay_s,
                      failures=self._failures)
        self._companies = registry_rows(registry, tickers, sic, self._researched,
                                        dt.date.today().isoformat())
        if not records:
            return empty_observations()
        return self.validate(pd.DataFrame(records))

    def fetch_tables(self) -> dict[str, list[dict[str, Any]]]:
        return {"companies": self._companies} if self._companies else {}

    def partial_failures(self) -> list[str]:
        return list(self._failures)
