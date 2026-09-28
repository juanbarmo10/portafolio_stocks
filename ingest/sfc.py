"""Colombia's financial supervisor: the CUIF of the Superintendencia Financiera (§15.5, p. 11).

For Nu Holdings the SEC reads nothing (IFRS, no quarterly XBRL), and the Banco Central do
Brasil covers only Brazil (``ingest/bcb``). Colombia is Nu's other growth market, and there
the **Superintendencia Financiera** publishes, every month and for free on datos.gov.co,
the CUIF — the *Catálogo Único de Información Financiera* — of every supervised entity,
account by account. Nu Colombia is there, next to the competitors that matter to the thesis
(Lulo Bank, RappiPay, Nequi, Bold) and a large incumbent (Bancolombia). Verified on
2026-09-28: dataset ``mxk5-ce6w``, latest month July 2026, one month and six entities in
~3 s.

What is kept, per entity and month (pesos, total of all currencies — ``moneda = 0``):

- the accounts in ``sources.sfc.accounts`` (assets, net loans, deposits, equity, result);
- the loan book by **risk category** (A normal … E uncollectible): every CUIF sub-account of
  class 14 whose name starts with ``CATEGORIA``, summed per letter. The letter comes from the
  account's *name*, which is not stored anywhere else, so the classification happens here —
  as ``sec_xbrl`` classifies durations — and is kept as raw sums, never as a ratio.

⚠️ Income-statement accounts (class 4, 5 and ``590000``) are **cumulative from January**
(the Colombian convention); ``transform/sfc`` annualizes them by the month, never sums
months.

The CUIF says nothing about when each month was published. ``ts_release`` follows the
BCB rule (``ingest.bcb.release_for``): the first time the panel sees a month (observed), and
for the history found on the first run, month end + ``derived_lag_days`` (derived, late on
purpose). One measure: July 2026 was published by 2026-09-18, 49 days after the close.

fetch() -> observations, source ``sfc_cuif``, series ``{tipo}-{codigo}:{key}``.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping

import pandas as pd
import requests

from core.config import Settings
from core.logging_setup import get_logger
from ingest.base import Ingester, empty_observations, retry
from ingest.bcb import release_for

log = get_logger(__name__)

SOURCE = "sfc_cuif"
CATEGORIES = ("A", "B", "C", "D", "E")


def month_ends(months: int, today: dt.date) -> list[dt.date]:
    """The last ``months`` month-ends strictly before ``today``, oldest first."""
    end = (pd.Timestamp(today).replace(day=1) - pd.Timedelta(days=1)).date()
    out = []
    for _ in range(months):
        out.append(end)
        end = (pd.Timestamp(end).replace(day=1) - pd.Timedelta(days=1)).date()
    return out[::-1]


def entity_key(tipo: Any, codigo: Any) -> str:
    return f"{int(tipo)}-{int(codigo)}"


def category_of(account_name: str) -> str | None:
    """``A``…``E`` for a ``CATEGORIA X RIESGO …`` sub-account, else ``None``."""
    words = str(account_name).upper().split()
    if len(words) >= 2 and words[0] == "CATEGORIA" and words[1] in CATEGORIES:
        return words[1]
    return None


def month_values(rows: list[Mapping[str, Any]], accounts: Mapping[str, str],
                 entities: set[str]) -> dict[str, dict[str, float]]:
    """``{entity: {key: value}}`` for one month's rows: the configured accounts as they come,
    and the loan book summed per risk category (``loans_cat_a`` … ``loans_cat_e``)."""
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        entity = entity_key(row["tipo_entidad"], row["codigo_entidad"])
        if entity not in entities or row.get("valor") is None:
            continue
        values = out.setdefault(entity, {})
        account = str(row["cuenta"])
        value = float(row["valor"])
        if account in accounts:
            values[accounts[account]] = value
        elif account.startswith("14") and (letter := category_of(row.get("nombre_cuenta", ""))):
            key = f"loans_cat_{letter.lower()}"
            values[key] = values.get(key, 0.0) + value
    return out


class SfcIngester(Ingester):
    """The CUIF of the configured Colombian entities, monthly."""

    source = SOURCE

    def __init__(self, settings: Settings) -> None:
        cfg = dict(settings.source("sfc"))
        self._url = str(cfg.get("url", ""))
        self._entities = list(cfg.get("entities", []))
        self._accounts = {str(k): str(v) for k, v in dict(cfg.get("accounts", {})).items()}
        self._history = int(cfg.get("history_months", 25))
        self._recent = int(cfg.get("recent_months", 3))
        self._lag = int(cfg.get("derived_lag_days", 60))
        self._every_days = int(cfg.get("run_every_days", 7))
        self.force = False
        self._skip_reason: str | None = None
        self._known: dict[str, list[tuple[str, float]]] = {}
        self._seen: set[str] = set()
        self._failures: list[str] = []

    @staticmethod
    def is_available(settings: Settings) -> bool:
        cfg = settings.source("sfc")
        return bool(cfg.get("url")) and bool(cfg.get("entities"))

    def attach_database(self, conn: Any) -> None:
        """Every stored version (so a month keeps the date it was first seen), and whether
        a run is due."""
        try:
            rows = conn.execute("SELECT series_id, ts, ts_release, value, ingested_at FROM "
                                "observations WHERE source = ?", (SOURCE,)).fetchall()
        except Exception as exc:  # noqa: BLE001 — a fresh database has none
            log.debug("No stored CUIF rows: %s", exc)
            rows = []
        last = None
        for series_id, ts, release, value, ingested in rows:
            self._known.setdefault(f"{series_id}|{str(ts)[:10]}", []).append(
                (str(release), float(value)))
            self._seen.add(str(series_id))
            last = max(last or str(ingested), str(ingested))
        if last and not self.force and self._every_days > 0:
            age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(last)
            if age < dt.timedelta(days=self._every_days):
                self._skip_reason = (f"last run {age.days} day(s) ago, due every "
                                     f"{self._every_days}; `--force` to run it now")

    def _where(self, month: dt.date) -> str:
        by_tipo: dict[int, list[str]] = {}
        for e in self._entities:
            by_tipo.setdefault(int(e["tipo"]), []).append(f"'{int(e['codigo'])}'")
        entities = " OR ".join(f"(tipo_entidad='{t}' AND codigo_entidad in ({','.join(c)}))"
                               for t, c in sorted(by_tipo.items()))
        accounts = ",".join(f"'{a}'" for a in self._accounts)
        return (f"fecha_corte='{month.isoformat()}T00:00:00.000' AND moneda='0' AND "
                f"({entities}) AND (cuenta in ({accounts}) OR (starts_with(cuenta,'14') AND "
                "starts_with(nombre_cuenta,'CATEGORIA')))")

    def _month(self, month: dt.date) -> list[Mapping[str, Any]]:
        def call() -> list[Mapping[str, Any]]:
            response = requests.get(self._url, params={
                "$select": "tipo_entidad,codigo_entidad,cuenta,nombre_cuenta,valor",
                "$where": self._where(month), "$limit": 50000}, timeout=300)
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, list):
                raise ValueError(f"CUIF answered something that is not a list: {str(body)[:200]}")
            return body
        return retry(call, attempts=3)

    def fetch(self) -> pd.DataFrame:
        if self._skip_reason:
            log.info("CUIF skipped: %s.", self._skip_reason, extra={"source": SOURCE})
            return empty_observations()
        self._failures = []
        today = dt.date.today()
        entities = {entity_key(e["tipo"], e["codigo"]) for e in self._entities}
        first_run = not any(s.split(":")[0] in entities for s in self._seen)
        months = month_ends(self._history if first_run else self._recent, today)
        records: list[dict[str, Any]] = []
        published = 0
        for month in months:
            try:
                rows = self._month(month)
            except Exception as exc:  # noqa: BLE001 — one month must not sink the rest
                log.exception("CUIF %s failed.", month, extra={"source": SOURCE})
                self._failures.append(f"CUIF {month}: {exc}")
                continue
            if rows:
                published += 1
            for entity, values in month_values(rows, self._accounts, entities).items():
                for key, value in values.items():
                    series = f"{entity}:{key}"
                    release = release_for(series, month, value, self._known, today,
                                          series not in self._seen, self._lag)
                    if release is None:
                        continue
                    records.append({"source": SOURCE, "series_id": series,
                                    "ts": month.isoformat(), "ts_release": release[0],
                                    "value": value})
        if months and not published:
            self._failures.append("CUIF returned no rows for any month asked: the dataset "
                                  "or its columns may have changed")
        log.info("CUIF: %d new or revised value(s) over %d month(s) with data.", len(records),
                 published, extra={"source": SOURCE})
        return self.validate(pd.DataFrame(records)) if records else empty_observations()

    def partial_failures(self) -> list[str]:
        return list(self._failures)
