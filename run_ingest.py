"""Ingestion entry point (CLAUDE.md sections 7, 8).

Run daily after the US market close, orchestrated by a systemd timer locally and by
GitHub Actions later (section 3). Re-running is safe by construction: every write goes
through the idempotent upserts in db/loader.py.

Usage:
    python run_ingest.py --dry-run          # create/verify the DB, run no network calls
    python run_ingest.py                    # run the registered ingesters
    python run_ingest.py --only fred sec    # run a subset

An ingester whose prerequisites are missing (typically a secret) is **skipped with a
warning**, not treated as a failure: a pipeline that refuses to run because one optional
key is unset would block the sources that do work.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Any, Callable

from core.config import Settings, load_settings
from core.logging_setup import configure_logging, get_logger
from db import loader
from ingest.base import Ingester
from ingest.fred import FredIngester
from ingest.macro_calendar import MacroCalendarIngester
from ingest.ibkr_flex import (
    IbkrFlexIngester,
    cash_transactions_frame,
    trades_frame,
)
from ingest.prices import PricesIngester
from ingest.sec_filings import SecFilingsIngester
from ingest.short_interest import ShortInterestIngester
from ingest.universe import UniverseIngester
from ingest.sec_xbrl import SecXbrlIngester

log = get_logger(__name__)

# Registry: name -> (availability predicate, constructor). The predicate keeps the
# "skip, don't crash" decision next to the ingester that owns its prerequisites.
# Grows with each phase.
INGESTERS: dict[str, tuple[Callable[[Settings], bool], Callable[[Settings], Ingester]]] = {
    "fred": (FredIngester.is_available, FredIngester),
    "macro_calendar": (MacroCalendarIngester.is_available, MacroCalendarIngester),
    "prices": (PricesIngester.is_available, PricesIngester),
    "ibkr": (IbkrFlexIngester.is_available, IbkrFlexIngester),
    "sec": (SecXbrlIngester.is_available, SecXbrlIngester),
    "sec_filings": (SecFilingsIngester.is_available, SecFilingsIngester),
    "short_interest": (ShortInterestIngester.is_available, ShortInterestIngester),
    "universe": (UniverseIngester.is_available, UniverseIngester),
}

# Where non-observation rows go. An ingester returning a table name absent from this
# map is a programming error and fails loudly rather than dropping the rows.
TABLE_LOADERS: dict[str, Callable[[Any, list[dict]], int]] = {
    "corporate_actions": loader.upsert_corporate_actions,
    "companies": loader.upsert_companies,
    "filings": loader.upsert_filings,
    "events": loader.upsert_events,
    "securities": loader.upsert_securities,
    "universe_membership": loader.upsert_universe_membership,
    # The account tables take a frame rather than records, so they are adapted here
    # instead of bending either side of the contract.
    "trades": lambda conn, rows: loader.upsert_trades(conn, trades_frame(rows)),
    "cash_transactions": lambda conn, rows: loader.upsert_cash_transactions(
        conn, cash_transactions_frame(rows)
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="equitydash ingestion pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create/verify the database and report the plan without any network call.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="Run only the named ingesters (default: all registered).",
    )
    return parser.parse_args(argv)


def select_ingesters(only: list[str] | None) -> dict[str, tuple]:
    """Return the registry entries to run, failing loudly on an unknown name.

    Raises:
        SystemExit: If a requested ingester is not registered — a typo must not
            silently turn into "ran nothing" (section 10).
    """
    if not only:
        return dict(INGESTERS)
    unknown = [name for name in only if name not in INGESTERS]
    if unknown:
        available = sorted(INGESTERS) or ["<none registered yet>"]
        raise SystemExit(f"Unknown ingester(s): {unknown}. Available: {available}")
    return {name: INGESTERS[name] for name in only}


def available_ingesters(selected: dict[str, tuple], settings: Settings) -> dict[str, Ingester]:
    """Build the selected ingesters whose prerequisites are satisfied.

    A missing secret is a skip with a warning, not an error: the sources that can run
    should still run. What must never happen is a silent skip, so every one is logged.
    """
    built: dict[str, Ingester] = {}
    for name, (is_available, construct) in selected.items():
        if not is_available(settings):
            log.warning(
                "Skipping '%s': prerequisites not configured (see config/.env.example).",
                name, extra={"source": name},
            )
            continue
        built[name] = construct(settings)
    return built


def notify_failures(conn: Any, settings: Settings, failures: list[str]) -> None:
    """Send the failed sources to Telegram (section 8, phase 4: "any ingester failing").

    This is how an expired IBKR Flex token reaches the user (section 4.1) instead of only
    a log. Deduplicated per source per day; never allowed to mask the ingest's own exit
    code, so a broken bot cannot turn a failed run into an exception somewhere else.
    """
    if not settings.raw.get("alerts", {}).get("ingest_failure", True):
        return
    from alerts.rules import dispatch, ingest_failure_alerts  # noqa: PLC0415
    from alerts.telegram import TelegramSender  # noqa: PLC0415

    try:
        today = datetime.now(timezone.utc).date().isoformat()
        dispatch(conn, ingest_failure_alerts(failures, today), TelegramSender(settings))
    except Exception:  # noqa: BLE001 — the ingest result stands on its own
        log.exception("Could not send the ingest-failure alert.")


def run(args: argparse.Namespace) -> int:
    """Execute the pipeline. Returns a process exit code."""
    settings = load_settings()
    configure_logging(settings.log_level, settings.secrets.values())

    log.info("equitydash ingest starting (db=%s, dry_run=%s)", settings.db_path, args.dry_run)

    selected = select_ingesters(args.only)

    # The schema is applied even on a dry run: "the DB is created and it does not fail"
    # is exactly the phase-0 acceptance criterion (section 8).
    conn = loader.init_db(settings.db_path)
    try:
        if not selected:
            log.info(
                "No ingesters registered. Schema is applied and the database is ready at %s.",
                settings.db_path,
            )
            return 0

        ingesters = available_ingesters(selected, settings)

        if args.dry_run:
            log.info(
                "Dry run: would execute %s (skipped for missing prerequisites: %s). "
                "No network calls made.",
                sorted(ingesters), sorted(set(selected) - set(ingesters)),
            )
            return 0

        if not ingesters:
            log.error(
                "None of the selected ingesters (%s) has its prerequisites configured.",
                sorted(selected),
            )
            return 1

        failures: list[str] = []
        for name, ingester in ingesters.items():
            try:
                # Lets an ingester look up what to fetch (held tickers, last ingested
                # date) before it fetches it. Most do nothing here.
                ingester.attach_database(conn)
                df = ingester.fetch()
                rows = loader.upsert_observations(conn, df)
                log.info("Ingester '%s' upserted %d rows.", name, rows, extra={"source": name})
                for table, table_rows in ingester.fetch_tables().items():
                    if table not in TABLE_LOADERS:
                        raise KeyError(
                            f"Ingester '{name}' returned rows for unknown table '{table}'. "
                            f"Known: {sorted(TABLE_LOADERS)}."
                        )
                    TABLE_LOADERS[table](conn, table_rows)
                # Whatever the ingester managed to fetch is now loaded. A unit that failed
                # inside it (a series, a ticker) still has to reach the exit code, or a
                # partial run would look identical to a clean one.
                partial = ingester.partial_failures()
                if partial:
                    log.error(
                        "Ingester '%s' loaded its data but %d unit(s) failed: %s",
                        name, len(partial), partial, extra={"source": name},
                    )
                    failures.append(f"{name} ({', '.join(partial)})")
            except Exception:
                # One broken source must not abort the rest of the run, but the failure
                # is logged with a traceback and turns the exit code non-zero, so the
                # orchestrator notices (section 8, phase 4: "any ingester failing" alerts).
                log.exception("Ingester '%s' failed.", name, extra={"source": name})
                failures.append(name)

        if failures:
            log.error("Ingest finished with %d failed source(s): %s", len(failures), failures)
            notify_failures(conn, settings, failures)
            return 1
        log.info("Ingest finished successfully.")
        return 0
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
