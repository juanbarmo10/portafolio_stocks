"""Base class and shared retry/backoff for ingest modules (CLAUDE.md sections 7, 10).

Every concrete ingester subclasses :class:`Ingester` and implements ``fetch()``, returning
a DataFrame with the loader's column contract ``[source, series_id, ts, ts_release, value]``.
The ``retry`` helper centralizes exponential backoff so individual modules do not
reimplement it — needed for SEC rate limits (section 4.4) and for the IBKR Flex
two-step handshake, where the statement is generated asynchronously and the first
``GetStatement`` legitimately comes back not-ready (section 4.1).

Parsers fail loudly. A structural change upstream must surface as an exception, never as
a silently empty or half-parsed frame (sections 4.3, 10).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

import pandas as pd

from core.logging_setup import get_logger
from db.loader import OBSERVATION_COLUMNS, TS_RELEASE_UNKNOWN

log = get_logger(__name__)

T = TypeVar("T")


def retry(
    func: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Call ``func`` with exponential backoff.

    Args:
        func: Zero-argument callable to execute.
        attempts: Maximum number of attempts before re-raising.
        base_delay_s: Initial backoff delay; doubles each retry.
        max_delay_s: Upper bound on the delay between attempts.
        exceptions: Exception types that trigger a retry. Others propagate immediately.

    Returns:
        Whatever ``func`` returns on the first successful attempt.

    Raises:
        The last exception if every attempt fails.
    """
    delay = base_delay_s
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except exceptions as exc:
            if attempt == attempts:
                log.error("All %d attempts failed: %s", attempts, exc)
                raise
            log.warning(
                "Attempt %d/%d failed (%s); retrying in %.1fs.", attempt, attempts, exc, delay
            )
            time.sleep(delay)
            delay = min(delay * 2, max_delay_s)
    # Unreachable: the loop either returns or raises.
    raise RuntimeError("retry: exhausted attempts without returning")


def empty_observations() -> pd.DataFrame:
    """An empty frame with the observation contract, for the nothing-to-do path."""
    return pd.DataFrame(columns=OBSERVATION_COLUMNS)


class Ingester(ABC):
    """Abstract data-source ingester.

    Subclasses set :attr:`source` and implement :meth:`fetch`. Use :meth:`validate`
    to assert the output honors the loader column contract before handing it over.
    """

    #: Source label written to observations.source (e.g. 'fred', 'sec', 'stooq').
    source: str = ""

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Fetch data and return a DataFrame with :data:`OBSERVATION_COLUMNS`."""
        raise NotImplementedError

    def fetch_tables(self) -> dict[str, list[dict]]:
        """Rows destined for tables other than ``observations``, keyed by table name.

        Not every fact is a time series. Corporate actions, filings and trades have their
        own tables (section 6), and they usually arrive in the *same* upstream response as
        the observations — so they are returned by the same ingester rather than fetched
        twice.

        Called by the runner right after :meth:`fetch`. Default: nothing.
        """
        return {}

    def attach_database(self, conn: Any) -> None:
        """Optional: read from the database what the ingester needs to know *what* to fetch.

        Deliberately narrow: this is for deciding the work, never for transforming data
        (section 10 keeps transformations pure and out of ingest). The price ingester is
        the case that motivated it — the tickers actually held live in the account data,
        so without this it would download the market references and the written universe
        and still leave the real portfolio unpriced, which is exactly the hole the
        reconciliation found.

        Called by the runner before :meth:`fetch`. Default: nothing.
        """
        return None

    def partial_failures(self) -> list[str]:
        """Sub-units (series, tickers) that failed while the rest of the fetch succeeded.

        A source usually fetches many independent units. Letting one broken unit abort the
        whole source throws away good data for no reason — the same argument that makes
        run_ingest isolate sources from each other (RESEARCH.md section 1.6). But a failure
        must never be silent, so the ingester reports what it lost here and the runner turns
        the exit code non-zero while still loading what it got.

        Returns:
            Human-readable labels of the units that failed. Empty when all succeeded.
        """
        return []

    @classmethod
    def validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Assert the DataFrame honors the observation contract; fail loudly otherwise.

        Args:
            df: Candidate observations DataFrame.

        Returns:
            The same DataFrame (for chaining) if valid.

        Raises:
            ValueError: If a required column is missing, or if ``ts``/``series_id``
                contain nulls — those are keys, and a null key means the parser
                silently lost a row (sections 10, 12).
        """
        missing = [c for c in OBSERVATION_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"fetch() output missing required columns: {missing}. "
                f"Expected {OBSERVATION_COLUMNS}."
            )
        if df.empty:
            return df
        for key in ("series_id", "ts"):
            if df[key].isna().any():
                n_bad = int(df[key].isna().sum())
                raise ValueError(
                    f"fetch() produced {n_bad} row(s) with a null '{key}'. That column is "
                    "part of the primary key; a null there means the parser dropped data."
                )
        # `value` may legitimately be null ("not reported"), and `ts_release` may be
        # unknown — but it is a key column, so it is normalized rather than left null.
        return df

    @staticmethod
    def observation_frame(records: list[dict]) -> pd.DataFrame:
        """Build a contract-shaped frame from records, defaulting unknown ts_release.

        Args:
            records: Row dicts carrying at least source, series_id, ts and value.

        Returns:
            A DataFrame with exactly :data:`OBSERVATION_COLUMNS`.
        """
        df = pd.DataFrame(records)
        if df.empty:
            return empty_observations()
        if "ts_release" not in df.columns:
            df["ts_release"] = TS_RELEASE_UNKNOWN
        else:
            df["ts_release"] = df["ts_release"].fillna(TS_RELEASE_UNKNOWN)
        return df[OBSERVATION_COLUMNS]
