"""Official exchange rate of the owner's local currency against the USD (section 11).

The portfolio is in USD; the owner's life, and taxes, are in another currency (section 9.9).
Every figure of the local tax layer needs the official rate of a given day, so this
ingester stores that rate's daily history.

**Configured only in ``settings.local.yaml``** (``fiscal.local_fx``), never in the tracked
config: the source, the currency and the field names would reveal the owner's jurisdiction
(section 11). Without that block the ingester is unavailable and skipped, like any source
without its prerequisites.

The expected source is a Socrata-style JSON endpoint returning one row per validity period,
with a start date and a value; the field names are part of the config. A rate valid over a
weekend arrives as one row whose start date is the Friday or Saturday, and readers take the
latest rate at or before a date (as-of), so no row is invented for the days it covers.

``ts`` = the date the rate is valid from; ``ts_release`` = the same date. An official rate is
usually published the business day **before** it applies, so this errs late, which is the
safe side (section 9.4).

fetch() -> DataFrame[source, series_id, ts, ts_release, value]  (local currency per USD)
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import requests

from core.config import Settings
from core.logging_setup import get_logger
from ingest.base import Ingester, empty_observations, retry

log = get_logger(__name__)


def parse_rates(
    payload: Any, *, date_field: str, value_field: str, source: str, series_id: str,
) -> pd.DataFrame:
    """Rows of the endpoint → long observations, one per validity start date.

    Raises:
        ValueError: If the payload is not a list — a changed endpoint must fail loudly.
        KeyError: If a row lacks the configured fields, for the same reason.
    """
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list, got {type(payload).__name__}")
    by_day: dict[str, float] = {}
    for row in payload:
        by_day[str(row[date_field])[:10]] = float(row[value_field])
    if not by_day:
        return empty_observations()
    days = sorted(by_day)
    return pd.DataFrame({
        "source": source,
        "series_id": series_id,
        "ts": [f"{d}T00:00:00+00:00" for d in days],
        "ts_release": [f"{d}T00:00:00+00:00" for d in days],
        "value": [by_day[d] for d in days],
    })


def fx_config(settings: Settings) -> Mapping[str, Any]:
    """The ``fiscal.local_fx`` block, or empty when the owner has not configured it."""
    return (settings.raw.get("fiscal") or {}).get("local_fx") or {}


class LocalFxIngester(Ingester):
    """Daily official rate, local currency per USD."""

    def __init__(self, settings: Settings) -> None:
        cfg = fx_config(settings)
        self.source = str(cfg.get("source", "local_fx"))
        self._series_id = str(cfg.get("series_id", "FX:LOCAL_PER_USD"))
        self._url = str(cfg["url"])
        self._date_field = str(cfg.get("date_field", "date"))
        self._value_field = str(cfg.get("value_field", "value"))
        self._since = str(cfg.get("history_since", "2024-01-01"))
        self._timeout = int(cfg.get("request_timeout_s", 30))

    @staticmethod
    def is_available(settings: Settings) -> bool:
        """Only with the local config block: the source reveals the jurisdiction."""
        return bool(fx_config(settings).get("url"))

    def fetch(self) -> pd.DataFrame:
        params = {
            "$where": f"{self._date_field} >= '{self._since}T00:00:00'",
            "$order": f"{self._date_field} ASC",
            "$limit": 100000,
        }

        def call() -> Any:
            response = requests.get(self._url, params=params, timeout=self._timeout)
            response.raise_for_status()
            return response.json()

        payload = retry(call, exceptions=(requests.RequestException,))
        frame = parse_rates(payload, date_field=self._date_field, value_field=self._value_field,
                            source=self.source, series_id=self._series_id)
        log.info("Local FX: %d daily rates since %s.", len(frame), self._since,
                 extra={"source": self.source, "series_id": self._series_id})
        return self.validate(frame)
