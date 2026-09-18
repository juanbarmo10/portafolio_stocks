"""Daily price ingester: RAW OHLCV plus corporate actions (CLAUDE.md sections 4.3, 9.1).

Feeds portfolio valuation (level 4) and market structure (level 2).

The whole module exists because of one fact: **no free provider serves a raw close.**
Yahoo — reached through yfinance — back-adjusts prices, volume and dividends for splits,
retroactively, forever. So the ingester un-adjusts on the way in and stores the raw
figures, which is what section 9.1 requires and what keeps the stored history immutable.

Why raw matters, concretely:
    A split-adjusted series is not a fact, it is a fact *plus a viewpoint*: the split
    history known on the day it was downloaded. AAPL's close for 2020-08-28 reads 124.81
    today and read 499.23 before the 2020 split. The next split will rewrite it again, and
    every backtest run before that split silently stops reproducing. The raw close, 499.23,
    is true forever. It also makes the provider replaceable, which matters a lot for a
    library with no API contract (section 4.3).

Yahoo's adjustment conventions, verified empirically on 2026-08-08 (see RESEARCH.md §2.2):

    ============  ================================  =====================
    Field         Convention                        Reconstruction
    ============  ================================  =====================
    OHLC          split-adjusted                    multiply by factor
    Volume        split-adjusted                    **divide** by factor
    Dividends     split-adjusted                    multiply by factor
    ============  ================================  =====================

    ``auto_adjust=False`` only turns off *dividend* adjustment; the series stays
    split-adjusted. The parameter name is misleading and cost this project a full
    verification pass — do not trust it.

fetch() -> DataFrame[source, series_id, ts, ts_release, value]
    series_id = "TICKER:open_raw" | ":high_raw" | ":low_raw" | ":close_raw" | ":volume_raw"

fetch_tables() -> {"corporate_actions": [...]}   (splits and dividends, un-adjusted)
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping

import pandas as pd

from core.config import Settings
from core.logging_setup import get_logger
from ingest.base import Ingester, empty_observations, retry

log = get_logger(__name__)

SOURCE = "yfinance"

# Price fields multiplied by the split factor; volume is handled separately because
# Yahoo scales it the other way (see the module docstring).
_PRICE_FIELDS = {"Open": "open_raw", "High": "high_raw", "Low": "low_raw", "Close": "close_raw"}
_VOLUME_FIELD = "Volume"
_VOLUME_SERIES = "volume_raw"


def split_factor(bar_date: dt.date, splits: Mapping[dt.date, float]) -> float:
    """Cumulative factor that converts a split-adjusted price back to raw on ``bar_date``.

    Args:
        bar_date: Calendar date of the price bar.
        splits: Split ratios keyed by **ex-date** (e.g. ``{date(2020, 8, 31): 4.0}``).

    Returns:
        The product of every ratio whose ex-date is **strictly after** ``bar_date``.

    The strict comparison is the whole subtlety. The bar *on* the ex-date is already
    quoted post-split, so its factor is 1. Including it inflates that single bar by the
    ratio: for AAPL's 4:1 split, 2020-08-31 would read 516.16 instead of 129.04 — a 4x
    spike on one day, invisible in a yearly chart and ruinous in any return calculation.
    yfinance stamps splits at 09:30 and daily bars at 00:00, so comparing raw timestamps
    instead of calendar dates walks straight into it.
    """
    factor = 1.0
    for ex_date, ratio in splits.items():
        if ex_date > bar_date and ratio:
            factor *= float(ratio)
    return factor


def unadjust_history(history: pd.DataFrame, splits: Mapping[dt.date, float]) -> pd.DataFrame:
    """Reconstruct raw OHLCV from a split-adjusted frame. Pure function, no network.

    Args:
        history: Frame indexed by date, with Open/High/Low/Close/Volume columns as
            returned by ``yfinance.Ticker.history(auto_adjust=False)``.
        splits: Split ratios keyed by ex-date.

    Returns:
        A copy with prices multiplied and volume divided by each row's split factor,
        indexed by ``datetime.date``.
    """
    if history.empty:
        return history.copy()

    out = history.copy()
    out.index = [_as_date(ts) for ts in out.index]
    factors = pd.Series([split_factor(d, splits) for d in out.index], index=out.index)

    for column in _PRICE_FIELDS:
        if column in out.columns:
            out[column] = out[column] * factors
    if _VOLUME_FIELD in out.columns:
        # Yahoo scales historical volume UP so old bars are comparable to today's share
        # count; the raw traded share count is therefore the reported figure divided by
        # the factor. Verified via dollar-volume continuity across AAPL's 2020 split.
        out[_VOLUME_FIELD] = out[_VOLUME_FIELD] / factors
    return out


def _as_date(value: Any) -> dt.date:
    """Normalize any timestamp to a calendar date.

    Daily bars are calendar labels, not instants: yfinance returns them tz-aware in the
    exchange's timezone, and converting to UTC would shift a US close into the next day.
    The trading date is taken as-is, matching how FRED reference dates are stored.
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()


def _iso(day: dt.date) -> str:
    """Calendar date as ISO8601 UTC midnight (section 10)."""
    return f"{day.isoformat()}T00:00:00+00:00"


def build_observations(ticker: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Convert an un-adjusted OHLCV frame to the long loader contract.

    ``ts_release`` equals ``ts``: a daily close is public at the end of the session it
    describes, so there is no publication lag to model (unlike SEC or FRED data). It is
    stamped explicitly rather than left unknown, because it is known.
    """
    records: list[dict[str, Any]] = []
    for day, row in raw.iterrows():
        stamp = _iso(day)
        for column, suffix in _PRICE_FIELDS.items():
            if column in raw.columns:
                records.append({
                    "source": SOURCE, "series_id": f"{ticker}:{suffix}",
                    "ts": stamp, "ts_release": stamp, "value": row[column],
                })
        if _VOLUME_FIELD in raw.columns:
            records.append({
                "source": SOURCE, "series_id": f"{ticker}:{_VOLUME_SERIES}",
                "ts": stamp, "ts_release": stamp, "value": row[_VOLUME_FIELD],
            })

    if not records:
        return empty_observations()
    df = pd.DataFrame(records)
    # A bar with no trading is reported as NaN; it stays NULL rather than becoming 0.
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def build_corporate_actions(
    ticker: str, splits: Mapping[dt.date, float], dividends: Mapping[dt.date, float],
    currency: str = "USD",
) -> list[dict[str, Any]]:
    """Build corporate_actions rows, with dividends un-adjusted back to raw.

    Args:
        ticker: Symbol the actions belong to.
        splits: Ratios keyed by ex-date.
        dividends: Per-share amounts keyed by ex-date, as reported (split-adjusted).
        currency: Currency of the dividend amounts.

    Returns:
        Row dicts for ``db.loader.upsert_corporate_actions``. ``action_id`` is
        deterministic so re-ingesting updates in place instead of duplicating.

    Dividends get the same treatment as prices: Yahoo divides historical per-share
    amounts by later splits, so AAPL's 2019 dividend reads 0.1925 instead of the 0.77
    actually paid. Storing the reported figure would understate every pre-split payout
    and corrupt total-return calculations.
    """
    rows: list[dict[str, Any]] = []
    for ex_date, ratio in sorted(splits.items()):
        if not ratio:
            continue
        rows.append({
            "action_id": f"{ticker}:split:{ex_date.isoformat()}",
            "cik": None,  # resolved later where possible; never guessed (section 9.3)
            "ticker": ticker, "kind": "split", "ex_date": _iso(ex_date),
            "ratio": float(ratio), "amount": None, "currency": None, "source": SOURCE,
        })
    for ex_date, amount in sorted(dividends.items()):
        if not amount:
            continue
        rows.append({
            "action_id": f"{ticker}:dividend:{ex_date.isoformat()}",
            "cik": None,
            "ticker": ticker, "kind": "dividend", "ex_date": _iso(ex_date),
            "ratio": None,
            "amount": float(amount) * split_factor(ex_date, splits),
            "currency": currency, "source": SOURCE,
        })
    return rows


class PricesIngester(Ingester):
    """Daily raw OHLCV and corporate actions for the configured tickers."""

    source = SOURCE

    def __init__(self, settings: Settings) -> None:
        cfg = settings.source("yfinance")
        self._history_start = str(cfg.get("history_start", "2015-01-01"))
        self._currency = cfg.get("currency", "USD")
        self._tickers = self.tickers_for(settings)
        retry_cfg = settings.raw.get("ingest", {}).get("retry", {})
        self._retry_kwargs = {
            "attempts": retry_cfg.get("attempts", 4),
            "base_delay_s": retry_cfg.get("base_delay_s", 1.0),
            "max_delay_s": retry_cfg.get("max_delay_s", 30.0),
        }
        self._actions: list[dict[str, Any]] = []
        self._failures: list[str] = []

    @staticmethod
    def tickers_for(settings: Settings) -> list[str]:
        """Market references (section 5.1) plus any tracked company, deduplicated."""
        tracked = [
            str(c["ticker"]) for c in settings.tracked_companies if c.get("ticker")
        ]
        return list(dict.fromkeys([*settings.market_references, *tracked]))

    def attach_database(self, conn: Any) -> None:
        """Add whatever the account actually holds to the download list.

        A position with no price cannot be valued, and the panel refuses to estimate one
        (section 12), so the reconciliation against IBKR's NAV simply could not run for a
        holding outside the written universe. Which is the common case early on: the
        universe is a list of *analysis*, and a ticker can be held before its thesis card
        is written — section 5.3 flags that separately, it does not excuse leaving the
        position unpriced.
        """
        try:
            rows = conn.execute(
                "SELECT DISTINCT series_id FROM observations "
                "WHERE source = 'ibkr' AND series_id LIKE '%:position_qty'"
            ).fetchall()
        except Exception as exc:  # a fresh database has no rows, not a failure
            log.debug("Could not read held positions: %s", exc)
            return
        held = [str(row[0]).rsplit(":", 1)[0] for row in rows]
        added = [ticker for ticker in held if ticker not in self._tickers]
        if added:
            self._tickers.extend(added)
            log.info(
                "Adding %d held ticker(s) to the price download: %s",
                len(added), sorted(added), extra={"source": SOURCE},
            )

    @staticmethod
    def is_available(settings: Settings) -> bool:
        """Whether yfinance is installed and there is at least one ticker configured."""
        try:
            import yfinance  # noqa: F401, PLC0415 — optional extra
        except ImportError:
            return False
        return bool(PricesIngester.tickers_for(settings))

    def _download(self, ticker: str) -> pd.DataFrame:
        """One history request with actions, retried on transient failures."""
        import yfinance  # noqa: PLC0415 — optional extra, imported at call time

        def call() -> pd.DataFrame:
            # The full window is re-fetched every run rather than incrementally: a split
            # that lands tomorrow re-scales every past bar Yahoo serves, and only a
            # window that spans the split lets the un-adjustment restore the raw values.
            frame = yfinance.Ticker(ticker).history(
                start=self._history_start, auto_adjust=False, actions=True
            )
            if frame is None or frame.empty:
                raise RuntimeError(f"yfinance returned no rows for {ticker}.")
            return frame

        return retry(call, **self._retry_kwargs)

    def fetch(self) -> pd.DataFrame:
        """Return raw OHLCV for every configured ticker as one long DataFrame.

        Corporate actions found along the way are stashed for :meth:`fetch_tables`,
        which the runner calls straight after — they come from the same response, so
        fetching them separately would double the requests for no benefit.
        """
        frames: list[pd.DataFrame] = []
        self._actions = []
        self._failures = []

        for ticker in self._tickers:
            try:
                history = self._download(ticker)
                splits = self._events(history, "Stock Splits")
                dividends = self._events(history, "Dividends")
                raw = unadjust_history(history, splits)
                actions = build_corporate_actions(ticker, splits, dividends, self._currency)
            except Exception:
                # One delisted or renamed ticker must not cost the other thirteen. The
                # failure is logged and reported through partial_failures().
                log.exception(
                    "Ticker failed; continuing with the rest.",
                    extra={"source": self.source, "series_id": f"{ticker}:close_raw"},
                )
                self._failures.append(ticker)
                continue

            frames.append(build_observations(ticker, raw))
            self._actions.extend(actions)
            log.info(
                "%d bars, %d splits, %d dividends.",
                len(raw), len(splits), len(dividends),
                extra={"source": self.source, "series_id": f"{ticker}:close_raw"},
            )

        df = pd.concat(frames, ignore_index=True) if frames else empty_observations()
        return self.validate(df)

    def fetch_tables(self) -> dict[str, list[dict[str, Any]]]:
        """Corporate actions collected during :meth:`fetch`."""
        return {"corporate_actions": self._actions}

    def partial_failures(self) -> list[str]:
        """Tickers that failed during the last :meth:`fetch`."""
        return list(self._failures)

    @staticmethod
    def _events(history: pd.DataFrame, column: str) -> dict[dt.date, float]:
        """Extract non-zero action rows from a history frame, keyed by ex-date."""
        if column not in history.columns:
            return {}
        series = history[column]
        return {
            _as_date(ts): float(value) for ts, value in series.items() if value
        }
