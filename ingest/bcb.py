"""Banco Central do Brasil: supervisory data (IF.data) and the Selic (SGS).

For the companies the SEC does not let the panel read (RESEARCH.md §2.43). Nu Holdings files
IFRS on 20-F/6-K with no quarterly XBRL, but its Brazilian business is supervised by the
BCB, which publishes every institution's quarterly figures through an open OData API. They
are Brazilian regulatory accounting (COSIF), in **reais**, Brazil only — a different lens
from the company's IFRS reports, and one the company does not choose: that is its value.

**The publication date is not in the source** and the BCB publishes no calendar. So
``ts_release`` is (config ``sources.bcb``):

- **observed**: the first day the panel saw that quarter — an upper bound of the real date,
  so it can only err late; kept on every later run (read back in ``attach_database``);
- **derived**: quarter end + ``derived_lag_days`` for what already existed at the first
  ingest. The companion series ``{code}:release_observed`` says which (1/0) per quarter.

A value that changes after it was first seen is a revision: it is stored as a new row dated
the day it was seen, and the original stays (section 9.6).

``Lucro Líquido`` is **accumulated over the semester** (measured: Q2 = S1 − Q1). It is stored
as filed, under ``net_income_semester``; the quarter is derived in ``transform/``.

SGS series (the Selic target) are public the day they apply; the target is filled forward
to the next Copom meeting, so only dates up to the ingest day are stored.

fetch() -> observations, source ``bcb_ifdata`` / ``bcb_sgs``
    ``{code}:{key}``          e.g. ``C0084693:credit_portfolio``
    ``{code}:release_observed``
    ``BR:{key}``              e.g. ``BR:selic_target``
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping

import pandas as pd
import requests

from core.config import Settings
from core.logging_setup import get_logger
from ingest.base import Ingester, empty_observations, retry

log = get_logger(__name__)

IFDATA_SOURCE = "bcb_ifdata"
SGS_SOURCE = "bcb_sgs"
VALUES = ("IfDataValores(AnoMes=@AnoMes,TipoInstituicao=@TipoInstituicao,Relatorio=@Relatorio)"
          "?@AnoMes={am}&@TipoInstituicao={tipo}&@Relatorio='{report}'&$format=json"
          "&$filter=CodInst eq '{code}'")


def quarter_ends(first: str, today: dt.date) -> list[dt.date]:
    """Quarter-end dates from ``first`` (``2019q4``) up to the last one before ``today``."""
    year, q = int(first[:4]), int(first[-1])
    out = []
    while True:
        end = (pd.Timestamp(year=year, month=3 * q, day=1) + pd.offsets.MonthEnd(0)).date()
        if end >= today:
            return out
        out.append(end)
        year, q = (year + 1, 1) if q == 4 else (year, q + 1)


def column_values(rows: list[Mapping[str, Any]], wanted: Mapping[str, str]) -> dict[str, float]:
    """``{key: value}`` for the wanted columns of one report. The BCB's column names carry a
    formula label after a line break (``"Lucro Líquido \\n(z) = ..."``); the text before it is
    the name. An absent or empty value is left out, never read as zero."""
    out = {}
    for row in rows:
        name = str(row.get("NomeColuna") or "").split("\n")[0].strip()
        if name in wanted and row.get("Saldo") is not None:
            out[wanted[name]] = float(row["Saldo"])
    return out


def release_for(series_id: str, quarter_end: dt.date, value: float,
                known: Mapping[str, list[tuple[str, float]]], today: dt.date,
                first_run: bool, derived_lag_days: int) -> tuple[str, bool] | None:
    """``(ts_release, observed)`` for one value, or ``None`` if it is already stored as is.

    - Already stored with this value (any version): nothing new.
    - Stored with another value: a revision, dated today (observed).
    - Never seen, first run of this institution: derived, quarter end + lag — unless that is
      still in the future, in which case today (it is visible now, so it was published by now).
    - Never seen afterwards: first seen today (observed).
    """
    key = f"{series_id}|{quarter_end.isoformat()}"
    versions = known.get(key, [])
    if any(abs(v - value) <= 1e-6 * max(1.0, abs(value)) for _, v in versions):
        return None
    if versions or not first_run:
        return today.isoformat(), True
    derived = quarter_end + dt.timedelta(days=derived_lag_days)
    if derived >= today:
        return today.isoformat(), True
    return derived.isoformat(), False


class BcbIngester(Ingester):
    """IF.data for the configured institutions, and the SGS context series."""

    source = IFDATA_SOURCE

    def __init__(self, settings: Settings) -> None:
        cfg = dict(settings.source("bcb"))
        self._ifdata_url = str(cfg["ifdata_url"])
        self._sgs_url = str(cfg["sgs_url"])
        self._first = str(cfg.get("first_quarter", "2019q4"))
        self._institutions = list(cfg.get("institutions", []))
        self._reports = {str(k): dict(v) for k, v in dict(cfg.get("reports", {})).items()}
        self._lag = int(cfg.get("derived_lag_days", 120))
        self._sgs = {str(k): str(v) for k, v in dict(cfg.get("sgs_series", {})).items()}
        self._known: dict[str, list[tuple[str, float]]] = {}
        self._seen_series: set[str] = set()
        self._failures: list[str] = []

    @staticmethod
    def is_available(settings: Settings) -> bool:
        return bool(settings.source("bcb").get("institutions"))

    def attach_database(self, conn: Any) -> None:
        """Every stored version, so a quarter keeps the date it was first seen."""
        try:
            rows = conn.execute("SELECT series_id, ts, ts_release, value FROM observations "
                                "WHERE source = ?", (IFDATA_SOURCE,)).fetchall()
        except Exception as exc:  # noqa: BLE001 — a fresh database has none
            log.debug("No stored IF.data rows: %s", exc)
            rows = []
        for series_id, ts, release, value in rows:
            self._known.setdefault(f"{series_id}|{str(ts)[:10]}", []).append(
                (str(release), float(value)))
            self._seen_series.add(str(series_id))

    def _get(self, url: str, params: Mapping[str, str] | None = None) -> Any:
        def call() -> Any:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        return retry(call, exceptions=(requests.RequestException,))

    def _ifdata(self, today: dt.date) -> list[dict[str, Any]]:
        records = []
        for inst in self._institutions:
            code, tipo = str(inst["code"]), str(inst.get("tipo", 1))
            wanted_series = {f"{code}:{key}" for keys in self._reports.values()
                             for key in keys.values()}
            # The whole history while any configured series has never been stored (a first
            # run, or a column added to the config); afterwards only the last year: late
            # revisions and the new quarter.
            ends = quarter_ends(self._first, today)
            if wanted_series <= self._seen_series:
                ends = ends[-4:]
            for end in ends:
                am = f"{end.year}{end.month:02d}"
                observed_any: bool | None = None
                for report, wanted in self._reports.items():
                    url = self._ifdata_url + VALUES.format(am=am, tipo=tipo, report=report,
                                                           code=code)
                    try:
                        rows = self._get(url).get("value", [])
                    except Exception as exc:  # noqa: BLE001 — one report must not sink the rest
                        log.exception("IF.data %s %s report %s failed.", code, am, report,
                                      extra={"source": IFDATA_SOURCE})
                        self._failures.append(f"IF.data {code} {am} {report}: {exc}")
                        continue
                    for key, value in column_values(rows, wanted).items():
                        series = f"{code}:{key}"
                        release = release_for(series, end, value, self._known, today,
                                              series not in self._seen_series, self._lag)
                        if release is None:
                            continue
                        records.append({"source": IFDATA_SOURCE, "series_id": series,
                                        "ts": end.isoformat(), "ts_release": release[0],
                                        "value": value})
                        observed_any = release[1] if observed_any is None else observed_any
                if observed_any is not None:
                    records.append({"source": IFDATA_SOURCE,
                                    "series_id": f"{code}:release_observed",
                                    "ts": end.isoformat(), "ts_release": records[-1]["ts_release"],
                                    "value": 1.0 if observed_any else 0.0})
        return records

    def _sgs_rows(self, today: dt.date) -> list[dict[str, Any]]:
        records = []
        start = pd.Timestamp(self._first[:4] + "-01-01").strftime("%d/%m/%Y")
        for code, key in self._sgs.items():
            try:
                rows = self._get(self._sgs_url.format(code=code),
                                 {"formato": "json", "dataInicial": start,
                                  "dataFinal": today.strftime("%d/%m/%Y")})
            except Exception as exc:  # noqa: BLE001
                log.exception("SGS %s failed.", code, extra={"source": SGS_SOURCE})
                self._failures.append(f"SGS {code}: {exc}")
                continue
            for row in rows:
                day = dt.datetime.strptime(row["data"], "%d/%m/%Y").date()
                if day > today:
                    continue      # filled forward to the next Copom: not a fact yet
                records.append({"source": SGS_SOURCE, "series_id": f"BR:{key}",
                                "ts": day.isoformat(), "ts_release": day.isoformat(),
                                "value": float(row["valor"])})
        return records

    def fetch(self) -> pd.DataFrame:
        self._failures = []
        today = dt.date.today()
        supervisory = self._ifdata(today)
        context = self._sgs_rows(today)
        log.info("BCB: %d new or revised IF.data value(s); %d SGS row(s) (idempotent).",
                 len(supervisory), len(context), extra={"source": IFDATA_SOURCE})
        records = supervisory + context
        if not records:
            return empty_observations()
        return self.validate(pd.DataFrame(records))

    def partial_failures(self) -> list[str]:
        return list(self._failures)
