"""Ingestion entry point (CLAUDE.md sections 7, 8).

Run daily after the US market close, orchestrated by a systemd timer locally and by
GitHub Actions later (section 3). Re-running is safe by construction: every write goes
through the idempotent upserts in db/loader.py.

Usage:
    python run_ingest.py --dry-run          # create/verify the DB, run no network calls
    python run_ingest.py                    # run the registered ingesters
    python run_ingest.py --only fred sec    # run a subset

Phase 0 registers no ingesters yet: the scaffold has to stand on its own before the
first data source lands (section 8, "no advancing without the previous phase working").
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

import pandas as pd

from core.config import load_settings
from core.logging_setup import configure_logging, get_logger
from db import loader

log = get_logger(__name__)

# Registry of ingesters: name -> zero-argument callable returning an observations
# DataFrame. Populated as each phase lands (phase 1: fred, prices, ibkr_flex).
INGESTERS: dict[str, Callable[[], pd.DataFrame]] = {}


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


def select_ingesters(only: list[str] | None) -> dict[str, Callable[[], pd.DataFrame]]:
    """Return the ingesters to run, failing loudly on an unknown name.

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


def run(args: argparse.Namespace) -> int:
    """Execute the pipeline. Returns a process exit code."""
    settings = load_settings()
    configure_logging(settings.log_level)

    log.info("equitydash ingest starting (db=%s, dry_run=%s)", settings.db_path, args.dry_run)

    selected = select_ingesters(args.only)

    # The schema is applied even on a dry run: "the DB is created and it does not fail"
    # is exactly the phase-0 acceptance criterion (section 8).
    conn = loader.init_db(settings.db_path)
    try:
        if not selected:
            log.info(
                "No ingesters registered yet (phase 0 scaffold). Schema is applied and "
                "the database is ready at %s.", settings.db_path
            )
            return 0

        if args.dry_run:
            log.info("Dry run: would execute %s. No network calls made.", sorted(selected))
            return 0

        failures: list[str] = []
        for name, fetch in selected.items():
            try:
                df = fetch()
                rows = loader.upsert_observations(conn, df)
                log.info("Ingester '%s' upserted %d rows.", name, rows, extra={"source": name})
            except Exception:
                # One broken source must not abort the rest of the run, but the failure
                # is logged with a traceback and turns the exit code non-zero, so the
                # orchestrator notices (section 8, phase 4: "any ingester failing" alerts).
                log.exception("Ingester '%s' failed.", name, extra={"source": name})
                failures.append(name)

        if failures:
            log.error("Ingest finished with %d failed source(s): %s", len(failures), failures)
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
